import os
import sys
import json
import math
import datetime
import argparse
import time
import re
import numpy as np

from pathlib import Path
from stanfordcorenlp import StanfordCoreNLP
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from Utilities import process_keyphrases

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data_files(data_path, T):

    patterns = [
        re.compile( r"^(.+)_MAX([A-Z0-9]+)_(.+)\.jsonl$"),
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_YAKE_{T}\.jsonl$"),
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_TopicRank\.jsonl$")
    ]

    data_files = [
        file.name.split('/')[-1]
        for file in Path(data_path).iterdir()
        if (file.is_file() and (
                                (patterns[0].match(file.name) and file.name not in ["YAKE", "TopicRank"])
                                or patterns[1].match(file.name)
                                or patterns[2].match(file.name)
                               )
        )
    ]

    return data_files


def build_doc_text(title, keyphrases):
    return f"TITLE: {title}. KEYWORDS: {'; '.join(keyphrases)}"


def tokenized_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_prompt_overhead_tokens(tokenizer, prompt_template, task_instruction):
    """
    Token length of everything surrounding the TITLE+KEYWORDS content: the
    task instruction plus all prompt template symbols/formatting, with the
    document slot left empty.
    """
    return tokenized_len(tokenizer, prompt_template.format(task_instruction, ""))


def detect_iqr_outliers(lengths):
    """
    Standard IQR-based upper-outlier detection.
    Returns: q1, q3, iqr, upper_threshold, outlier_mask (boolean np.array)
    """
    lengths_arr = np.array(lengths)
    q1 = float(np.percentile(lengths_arr, 25))
    q3 = float(np.percentile(lengths_arr, 75))
    iqr = q3 - q1
    upper_threshold = q3 + 1.5 * iqr
    outlier_mask = lengths_arr > upper_threshold
    return q1, q3, iqr, upper_threshold, outlier_mask


def prepare_file_plan(data_path, data_file, tokenizer, task_instruction, prompt_template, T, prompt_overhead_tokens):
    """
    Full (untimed) preprocessing for one data file:
      1) First pass: tokenize TITLE+KEYWORDS content only for every document.
      2) IQR-based outlier detection on those lengths.
      3) Budget = max length among non-outlier documents (falls back to max of
         all documents automatically when no outliers exist).
      4) Manual pre-truncation at the token level for any document exceeding
         the budget (content-level slicing -- partial keywords allowed).
      5) max_model_len = prompt_overhead_tokens + budget.

    Returns a dict describing the fully prepared file.
    """
    with open(os.path.join(data_path, data_file), "r", encoding='utf-8') as f:
        lines = f.readlines()
        data_list = [json.loads(line.strip()) for line in lines]

    lengths = []
    for j_data in data_list:
        keyphrases = j_data['keyphrases'][:T]
        doc_text = build_doc_text(j_data['title'], keyphrases)
        lengths.append(tokenized_len(tokenizer, doc_text))

    lengths_arr = np.array(lengths)
    q1, q3, iqr, upper_threshold, outlier_mask = detect_iqr_outliers(lengths)

    non_outlier_lengths = lengths_arr[~outlier_mask]
    if non_outlier_lengths.size > 0:
        budget = int(non_outlier_lengths.max())
    else:
        # Degenerate case: IQR flagged everything -- fall back to the max of all documents
        budget = int(lengths_arr.max())

    num_outliers = int(outlier_mask.sum())

    prepared_docs = []
    num_truncated = 0

    for j_data, length in zip(data_list, lengths):
        keyphrases = j_data['keyphrases'][:T]
        doc_text = build_doc_text(j_data['title'], keyphrases)

        if length > budget:
            token_ids = tokenizer(doc_text, add_special_tokens=False)["input_ids"]
            truncated_ids = token_ids[:budget]
            doc_text_final = tokenizer.decode(truncated_ids)
            was_truncated = True
            num_truncated += 1
            final_len = budget
        else:
            doc_text_final = doc_text
            was_truncated = False
            final_len = length

        prompt = prompt_template.format(task_instruction, doc_text_final)

        prepared_docs.append({
            "j_data": j_data,
            "doc_text": doc_text_final,
            "prompt": prompt,
            "was_truncated": was_truncated,
            "orig_len": length,
            "final_len": final_len
        })

    max_model_len = prompt_overhead_tokens + budget

    return {
        "data_file": data_file,
        "prepared_docs": prepared_docs,
        "budget": budget,
        "max_model_len": max_model_len,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper_threshold": upper_threshold,
        "num_outliers": num_outliers,
        "num_truncated": num_truncated
    }


def group_file_plans(file_plans, tolerance_frac=0.15):
    """
    Sorts file plans by their required max_model_len (ascending), then
    greedily groups consecutive files whose max_model_len stays within
    tolerance_frac (relative) of the group's current max. A new group starts
    whenever the next file's requirement exceeds that tolerance.

    Since these are one-dimensional values, this sequential/greedy approach
    is equivalent in spirit to the earlier bin-packing discussion, just
    applied to max_model_len instead of batch-size budget.

    Returns a list of dicts: {"plans": [...], "engine_max_model_len": int}
    """
    sorted_plans = sorted(file_plans, key=lambda p: p["max_model_len"])

    groups = []
    current_group = []
    group_max = None

    for plan in sorted_plans:
        if not current_group:
            current_group = [plan]
            group_max = plan["max_model_len"]
        elif abs(plan["max_model_len"] - group_max) / group_max <= tolerance_frac:
            current_group.append(plan)
            group_max = max(group_max, plan["max_model_len"])
        else:
            groups.append({"plans": current_group, "engine_max_model_len": group_max})
            current_group = [plan]
            group_max = plan["max_model_len"]

    if current_group:
        groups.append({"plans": current_group, "engine_max_model_len": group_max})

    return groups


if __name__ == '__main__':

    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--max_new_tokens', type=str, default='64', help="Maximum number of tokens to generate")
    parser.add_argument('--auth_token', type=str, default='', help="Authentication token for Llama")
    parser.add_argument('--T', type=str, default='10', help="Number of keywords to extract")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    parser.add_argument('--gpu_memory_utilization', type=str, default='0.85', help="Fraction of GPU VRAM vLLM is allowed to claim")
    parser.add_argument('--engine_group_tolerance', type=str, default='0.15',
                         help="Relative tolerance for grouping files to share a vLLM engine, based on max_model_len similarity")
    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    T = int(args.T)
    datasets_max_len = args.datasets_max_len
    gpu_memory_utilization = float(args.gpu_memory_utilization)
    engine_group_tolerance = float(args.engine_group_tolerance)

    data_files = get_data_files(data_path, T)

    print("The test datasets are:\n")
    for data_file in data_files:
        print(data_file)

    choice = input("\nContinue? (y/n): ").strip().lower()

    if choice in ("y", "yes"):
        pass
    elif choice in ("n", "no"):
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid input. Exiting...")
        sys.exit(1)

    task_instruction = (
        f"You are a keyphrase synthesis assistant. Given a document where the title appears after TITLE: "
        f"and the initial semicolon-separated list of keywords appears after KEYWORDS:, produce exactly {T} "
        f"concise, relevant, and informative keyphrases by refining, normalizing, combining, removing "
        f"irrelevant entries, and adding important concepts implied by the title or existing keywords, using "
        f"both as relevance anchors. The answer should be listed after 'Keyphrases: ' and separated by "
        f"semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)

    prompt_overhead_tokens = get_prompt_overhead_tokens(tokenizer, prompt_template, task_instruction)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop=["<|eot_id|>"]
    )

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # ================================================================================
    # Preprocessing for ALL files (NOT timed): IQR outlier detection, budget,
    # pre-truncation, and max_model_len -- done up front so files can be grouped
    # by engine compatibility before any engine is created.
    # ================================================================================

    file_plans = []
    for data_file in data_files:
        print(f"\nPreparing: {data_file}")
        plan = prepare_file_plan(
            data_path, data_file, tokenizer, task_instruction, prompt_template, T, prompt_overhead_tokens
        )
        print(f"  Q1={plan['q1']:.2f}  Q3={plan['q3']:.2f}  IQR={plan['iqr']:.2f}  "
              f"Upper threshold={plan['upper_threshold']:.2f}")
        print(f"  Outliers detected: {plan['num_outliers']}  |  Budget: {plan['budget']}  |  "
              f"max_model_len: {plan['max_model_len']}")
        print(f"  Documents truncated: {plan['num_truncated']} / {len(plan['prepared_docs'])}")
        file_plans.append(plan)

    groups = group_file_plans(file_plans, tolerance_frac=engine_group_tolerance)

    print(f"\n[ENGINE GROUPING] {len(groups)} engine group(s) for {len(file_plans)} file(s) "
          f"(tolerance={engine_group_tolerance})")
    for i, group in enumerate(groups):
        file_names = [p["data_file"] for p in group["plans"]]
        print(f"  Group {i}: engine_max_model_len={group['engine_max_model_len']}  files={file_names}")

    # ================================================================================
    # Generation, per group (engine created/reused across the group's files)
    # ================================================================================

    for group in groups:

        engine_max_model_len = group["engine_max_model_len"]

        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=engine_max_model_len,
        )

        for plan in group["plans"]:

            data_file = plan["data_file"]
            prepared_docs = plan["prepared_docs"]
            budget = plan["budget"]
            max_model_len = plan["max_model_len"]

            print(f"\nGenerating: {data_file}")

            prompts = [d["prompt"] for d in prepared_docs]

            # ============================ Timed region ============================
            # Only generation + output_list construction are timed.

            perdataset_start_time = time.perf_counter()

            outputs = llm.generate(prompts, sampling_params)

            output_list = []
            for prepared, output in zip(prepared_docs, outputs):
                j_data = prepared["j_data"]

                generated_output_str = output.outputs[0].text.strip()
                pred_keyphrases_seq = generated_output_str.lower().split('keyphrases:')[-1].strip().rstrip('.')

                log = {}
                log['final_pred_keyphrase'] = [pred.strip() for pred in pred_keyphrases_seq.split(';')]
                log['label'] = j_data['label']
                log['generated_output'] = generated_output_str
                log['normalized_label'] = j_data['normalized_label']
                log['was_truncated'] = prepared["was_truncated"]
                log['orig_len'] = prepared["orig_len"]
                log['final_len'] = prepared["final_len"]
                log['prompt'] = prepared["prompt"]
                log['doc'] = prepared["doc_text"]
                
                output_list.append(log)

            perdataset_end_time = time.perf_counter()

            # ============================ End timed region ============================

            total_time = perdataset_end_time - perdataset_start_time

            # ============================ Statistics (NOT timed) ============================

            total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases(
                [log['final_pred_keyphrase'] for log in output_list]
            )

            perkeyphrase_no_tokens = []
            perdoc_no_keyphrases = []

            for log in output_list:
                pred_keyphrases_list = log['final_pred_keyphrase']
                perdoc_no_keyphrases.append(len(pred_keyphrases_list))

                for kw in pred_keyphrases_list:
                    perkeyphrase_no_tokens.append(len(kw.split()))

            # ============================ Results path / settings ============================

            data_info = data_file.split('_')

            dataset = data_info[0]
            ke_method = data_info[2]

            if len(data_info) == 4 or len(data_info) == 3:
                clustering_technique = None
                similarity_technique = None
                similarity_threshold = None
            elif len(data_info) == 6:
                clustering_technique = data_info[3]
                similarity_technique = data_info[4]
                similarity_threshold = data_info[5].split('.')[0]
            else:
                raise ValueError(f"Unexpected filename format: {data_file}")

            settings = {
                'model_name': f"{model_name.split('/')[-1]}_{ke_method}" if clustering_technique is None else f"{model_name.split('/')[-1]}_{ke_method}_{clustering_technique}_{similarity_technique}_{int(float(similarity_threshold) * 100)}",
                'task_instruction': task_instruction,
                'max_new_tokens': int(max_new_tokens),
                'budget_method': 'IQR',
                'q1': plan['q1'],
                'q3': plan['q3'],
                'iqr': plan['iqr'],
                'upper_threshold': plan['upper_threshold'],
                'num_outliers': plan['num_outliers'],
                'budget': budget,
                'max_model_len': max_model_len,
                'engine_max_model_len': engine_max_model_len,
                'engine_group_tolerance': engine_group_tolerance,
                'gpu_memory_utilization': gpu_memory_utilization,
                'documents_truncated': plan['num_truncated'],
                'do_sample': False,
                'T': int(T),
                'ke_method': ke_method,
                'clustering_technique': clustering_technique,
                'similarity_technique': similarity_technique,
                'similarity_threshold': str(similarity_threshold),
            }

            if clustering_technique is None:
                result_path = os.path.join(
                    "results", model_name.split("/")[-1], ke_method, str(T),
                    f"{timestamp}_vllm_iqr",
                )
            else:
                result_path = os.path.join(
                    "results", model_name.split("/")[-1], ke_method, str(T),
                    clustering_technique, similarity_technique, str(similarity_threshold),
                    f"{timestamp}_vllm_iqr",
                )

            print(f"Results path: {result_path}")

            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print(f"Directory created: {result_path}")

            settings_path = os.path.join(result_path, 'settings.json')
            if not os.path.isfile(settings_path):
                with open(settings_path, 'w') as settings_file:
                    json.dump(settings, settings_file, indent=4)
                print(f"Settings saved to {settings_path}")

            # ============================ KE results ============================

            with open(os.path.join(result_path, f'{dataset}_result.json'), "w", encoding='utf-8') as f:
                for json_data in output_list:
                    f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

            # ============================ Statistics ============================

            stats = {
                "KE": f"Llama3vLLM_T{T}_{ke_method}" if clustering_technique is None else f"Llama3vLLM_T{T}_{ke_method}_{clustering_technique}_{similarity_technique}_{int(float(similarity_threshold) * 100)}",
                "Dataset": dataset,
                "T": T,
                "Timestamp": timestamp,
                "Datasets_Max_Length": datasets_max_len,
                "Budget_Method": "IQR",
                "Q1": plan['q1'],
                "Q3": plan['q3'],
                "IQR": plan['iqr'],
                "Upper_Threshold": plan['upper_threshold'],
                "Num_Outliers": plan['num_outliers'],
                "Budget": budget,
                "Max_Model_Len": max_model_len,
                "Engine_Max_Model_Len": engine_max_model_len,
                "Category": "F",

                "Runtime": {
                    "Per_Document": {
                        "Mean": -1.0,
                        "Median": -1.0,
                        "Min": -1.0,
                        "Max": -1.0
                    },
                    "Per_Dataset": float(total_time)
                },

                "Truncation": {
                    "Documents_Truncated": plan['num_truncated'],
                    "Total_Documents": len(prepared_docs),
                    "Truncation_Rate": plan['num_truncated'] / len(prepared_docs) if prepared_docs else 0.0
                },

                "Keywords": {
                    "Count": {
                        "Mean": float(np.mean(perdoc_no_keyphrases)),
                        "Median": float(np.median(perdoc_no_keyphrases)),
                        "Min": int(np.min(perdoc_no_keyphrases)),
                        "Max": int(np.max(perdoc_no_keyphrases))
                    },
                    "Length": {
                        "Mean": float(np.mean(perkeyphrase_no_tokens)),
                        "Median": float(np.median(perkeyphrase_no_tokens))
                    }
                },

                "Vocabulary": {
                    "Avg_Doc_Words": float(perdataset_avg_no_words),
                    "Non_Word_Ratio": float(total_non_word_count) / total_word_count if total_word_count else 0.0
                }
            }

            with open(os.path.join(result_path, f"{dataset}_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)

            print(f"Stats saved to {os.path.join(result_path, f'{dataset}_stats.json')}")

        # Engine released only when moving to the next group
        del llm