import os
import json
import math
import datetime
import argparse
import time
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from Utilities import process_keyphrases

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_datasets_max_len(raw_value):
    """
    Determines whether datasets_max_len should be treated as a fixed integer
    budget or as a mode string ('FULL', etc.) that triggers mean/median
    calculation.

    Returns: (mode, value)
        mode  = 'fixed' or 'stat'
        value = the int budget if mode == 'fixed', else the raw string
    """
    try:
        return 'fixed', int(raw_value)
    except ValueError:
        return 'stat', raw_value


def tokenized_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def get_prompt_overhead_tokens(tokenizer, prompt_template, task_instruction):
    """
    Token length of everything surrounding the document: the task
    instruction plus all prompt template symbols/formatting, with the
    document slot left empty.
    """
    return tokenized_len(tokenizer, prompt_template.format(task_instruction, ""))


def compute_document_lengths(tokenizer, data_list):
    """
    First pass: tokenizes the document text ONLY (no instruction, no prompt
    symbols) for every document in the dataset. Returns a list of lengths
    aligned with data_list.
    """
    return [tokenized_len(tokenizer, j_data['doc']) for j_data in data_list]


def truncate_documents(tokenizer, data_list, doc_lengths, budget):
    """
    Manual pre-truncation pass. For every document whose token length
    exceeds `budget`, truncates it at the TOKEN level:
        tokenizer.encode(full_text) -> slice to `budget` tokens -> decode back to text
    Documents at or below the budget are left unchanged.

    Returns a list of dicts: {"j_data", "doc_text", "was_truncated", "orig_len"}
    """
    prepared = []

    for j_data, length in zip(data_list, doc_lengths):

        if length > budget:
            token_ids = tokenizer(j_data['doc'], add_special_tokens=False)["input_ids"]
            truncated_ids = token_ids[:budget]
            truncated_text = tokenizer.decode(truncated_ids)
            was_truncated = True
        else:
            truncated_text = j_data['doc']
            was_truncated = False

        prepared.append({
            "j_data": j_data,
            "doc_text": truncated_text,
            "was_truncated": was_truncated,
            "orig_len": length
        })

    return prepared


if __name__ == '__main__':

    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--max_new_tokens', type=str, default='64', help="Maximum number of tokens to generate")
    parser.add_argument('--auth_token', type=str, default='', help="Authentication token for Llama")
    parser.add_argument('--T', type=str, default='10', help="Number of keyphrases to extract")

    parser.add_argument('--datasets_max_len', type=str, default='FULL',
                         help="If an integer string, used directly as a fixed document token budget. "
                              "If a non-integer string (e.g. 'FULL'), triggers mean/median-based budget calculation.")

    parser.add_argument('--length_stat', type=str, default='mean', choices=['mean', 'median'],
                         help="Which statistic to use as the document token budget when datasets_max_len is not an integer")

    parser.add_argument('--gpu_memory_utilization', type=str, default='0.85', help="Fraction of GPU VRAM vLLM is allowed to claim")
    
    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    T = int(args.T)
    datasets_max_len_raw = args.datasets_max_len  
    length_stat = args.length_stat
    gpu_memory_utilization = float(args.gpu_memory_utilization)

    mode, datasets_max_len_value = resolve_datasets_max_len(datasets_max_len_raw)

    task_instruction = (
        f"You are a keyphrase extractor. Extract {T} keyphrases from the text. The answer should be listed "
        f"after 'Keyphrases: ' and separated by semicolons (;). "
        f"'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)

    prompt_overhead_tokens = get_prompt_overhead_tokens(tokenizer, prompt_template, task_instruction) # Get number of tokens of task instruction and prompt symbols

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop=["<|eot_id|>"] # The final string segment in prompt_template that signals the end of the entire prompt
    )

    settings = {
        'model_name': "Llama3vLLM",
        'task_instruction': task_instruction,
        'max_new_tokens': max_new_tokens,
        'T': T,
        'datasets_max_len_mode': mode,
        'datasets_max_len_raw': datasets_max_len_raw,
        'length_stat': length_stat if mode == 'stat' else "N/A (fixed mode)",
        'gpu_memory_utilization': gpu_memory_utilization,
        'prompt_overhead_tokens': prompt_overhead_tokens,
        'do_sample': False,
    }

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join(
        'results',
        f"Llama3vLLM/Llama3vLLM_T{T}",
        f'{timestamp}_vllm_vanilla_{datasets_max_len_raw}'
    )

    print(f"Results path: {result_path}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")

    settings_path = os.path.join(result_path, 'settings.json')
    with open(settings_path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

    print(f"Settings saved to {settings_path}")

    dataset_list = [
        'MDPI',
        'SemEval2010',
        'DUC2001', 
        'nus',
        'krapivin'
    ]

    for dataset_name in dataset_list:

        print(f"\nDataset: {dataset_name}")

        # ============================ Preprocessing (NOT timed) =============================

        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len_raw}.jsonl'), "r", encoding='utf-8') as f:
            lines = f.readlines()
            data_list = [json.loads(line.strip()) for line in lines]

        if mode == 'fixed': 
            budget = datasets_max_len_value
            mean_len = None
            median_len = None
            doc_lengths = compute_document_lengths(tokenizer, data_list)
        else:
            doc_lengths = compute_document_lengths(tokenizer, data_list)
            mean_len = float(np.mean(doc_lengths))
            median_len = float(np.median(doc_lengths))
            chosen_len = mean_len if length_stat == 'mean' else median_len
            budget = int(math.ceil(chosen_len))

        prepared_docs = truncate_documents(tokenizer, data_list, doc_lengths, budget)
        num_truncated = sum(1 for d in prepared_docs if d["was_truncated"])

        max_model_len = prompt_overhead_tokens + budget

        print(f"  Mode: {mode}  |  Budget: {budget}  |  max_model_len: {max_model_len}")
        if mode == 'stat':
            print(f"  Mean doc length: {mean_len:.2f}  |  Median doc length: {median_len:.2f}  |  Using: {length_stat}")
        print(f"  Documents truncated: {num_truncated} / {len(prepared_docs)}")

        prompts = [prompt_template.format(task_instruction, d["doc_text"]) for d in prepared_docs] # Create the complete prompts to input the vLLM engine

        # Engine created/reloaded per dataset (intentional) -- NOT timed
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

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
            log['normalized_label'] = j_data['normalized_label']
            log['was_truncated'] = prepared["was_truncated"]
            log['orig_doc_len'] = prepared["orig_len"]
            log['generated_output'] = generated_output_str
            log['prompt'] = prompt_template.format(task_instruction, prepared["doc_text"])
            log['doc'] = prepared["doc_text"]
            
            output_list.append(log)

        perdataset_end_time = time.perf_counter()

        # ============================ End timed region ============================

        total_time = perdataset_end_time - perdataset_start_time

        del llm

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

        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f:
            for json_data in output_list:
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

        stats = {
            "KE": f"Llama3vLLM_T{T}",
            "Dataset": dataset_name,
            "T": T,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len_raw,
            "Budget": budget,
            "Max_Model_Len": max_model_len,
            "Length_Stat_Used": length_stat if mode == 'stat' else "N/A (fixed mode)",
            "Mean_Doc_Len": mean_len,
            "Median_Doc_Len": median_len,
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
                "Documents_Truncated": num_truncated,
                "Total_Documents": len(prepared_docs),
                "Truncation_Rate": num_truncated / len(prepared_docs) if prepared_docs else 0.0
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

        with open(os.path.join(result_path, f"{dataset_name}_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        print(f"  Stats saved to {os.path.join(result_path, f'{dataset_name}_stats.json')}")