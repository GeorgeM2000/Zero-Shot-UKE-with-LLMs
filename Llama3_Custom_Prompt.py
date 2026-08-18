import os
import sys
import json
import datetime
import argparse
import torch
import time
import re
import math
import numpy as np

from pathlib import Path
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from Utilities import process_keyphrases


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity_error()


def get_generated_output(str):
    split_prompt_output = str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
    return split_prompt_output[-1].strip().replace("<|eot_id|>", "")


# next_power_of_two() is used to determine the length of the tokenizer
# For example, if the maximum length of the test datasets is 400, next_power_of_two(400) will return 512
# next_power_of_two(512) will return 1024. Thus, the tokenizer length will be 1024
def next_power_of_two(x):
    return 1 << x.bit_length()


"""
 get_data_files() returns:
    1) The test datasets in data/processed that were created using a KE method and a clustering technique
    2) The maximum token length of all test datasets
    3) The mean token length of all test datasets

"""
def get_data_files(data_path, T):

    patterns = [
        re.compile( r"^(.+)_MAX([A-Z0-9]+)_(.+)\.jsonl$"), # Matches the datasets created using a KE method (except YAKE) with HAC
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_YAKE_{T}\.jsonl$"), # Matches the dataset created using YAKE for a given value of T
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_TopicRank\.jsonl$")
    ] 

    data_files = [
        file.name.split('/')[-1]
        for file in Path(data_path).iterdir()
        if (file.is_file() and (
                                (patterns[0].match(file.name) and file.name not in ["YAKE", "TopicRank"]) # For all other KE methods
                                or patterns[1].match(file.name) # For YAKE
                                or patterns[2].match(file.name) # For TopicRank
                               )
        )
    ]

    perdoc_len = [] # The length of each document 
    for data_file in data_files:

        with open(os.path.join(data_path, data_file), 'r', encoding='utf-8') as f: 
            lines = f.readlines() 
            data_list = [json.loads(line.strip()) for line in lines] 

        for j_data in data_list:
            doc_content = f"TITLE: {j_data['title']}. KEYWORDS: {'; '.join(j_data['keyphrases'][:T])}" # j_data['title'] + ". " + '; '.join(j_data['keyphrases'][:T]) # f"TITLE: {j_data['title']}. KEYWORDS: {'; '.join(j_data['keyphrases'][:T])}"
            perdoc_len.append(len(doc_content.split()))

    perdoc_len = np.array(perdoc_len)

    # Because perdoc_len is a numpy array we can calculate the average and the median values. This is very useful because max(perdoc_len) can be a large value, thereby increasing the input size of the LLM.
    # With np.mean(perdoc_len) or np.median(perdoc_len) we reduce the size of the LLM input while representing the text content size of most documents.
    return data_files, np.max(perdoc_len), np.mean(perdoc_len) #next_power_of_two(max(perdoc_len)), next_power_of_two(next_power_of_two(max(perdoc_len)))




if __name__ == '__main__':

    # We can use this prompt template without making any further modifications
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--max_new_tokens', type=str, default='64', help="Maximum number of tokens to generate")
    parser.add_argument('--cuda', type=str, default='0', help="GPU") # If there is a GPU, it is labeled as 0
    parser.add_argument('--auth_token', type=str, default='', help="Authentication token for Llama") 
    parser.add_argument('--T', type=str, default='10', help="Number of keywords to extract")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    args = parser.parse_args() # Parse the arguments

    model_name = args.model_name
    data_path = args.data_path
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    datasets_max_len = args.datasets_max_len if args.datasets_max_len == "FULL" else int(args.datasets_max_len)
    T = int(args.T)
    
    data_files, max_doc_len, mean_doc_len = get_data_files(data_path, T)

    print("The test datasets are:\n")
    for data_file in data_files: print(data_file)

    print(f"\nThe maximum document length of the test datasets is {max_doc_len}")
    print(f"The mean document length of the test datasets is {mean_doc_len}\n")

    choice = input("Continue? (y/n): ").strip().lower()

    if choice in ("y", "yes"):
        pass  # Continue execution
    elif choice in ("n", "no"):
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid input. Exiting...")
        sys.exit(1)
     
    



    # The task instruction is the most critical part of this evaluation. The instruction can be changed depending on the task
    #task_instruction = f"You are a keyphrase synthesis assistant. Given a document title and an initial semicolon-separated list of keywords, produce exactly {T} concise, relevant, and informative keyphrases by refining, normalizing, combining, removing irrelevant entries, and adding important concepts implied by the title or existing keywords, using both as relevance anchors. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"
    task_instruction = f"You are a keyphrase synthesis assistant. Given a document where the title appears after TITLE: and the initial semicolon-separated list of keywords appears after KEYWORDS:, produce exactly {T} concise, relevant, and informative keyphrases by refining, normalizing, combining, removing irrelevant entries, and adding important concepts implied by the title or existing keywords, using both as relevance anchors. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"


    # Loads a pretrained tokenizer associated with model_name. The tokenizer converts raw text → tokens (integer IDs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              token=auth_token) # AutoTokenizer: A factory class that automatically selects the correct tokenizer type. For LLaMA, this is typically a SentencePiece-based tokenizer
    
    # AutoModelForCausalLM loads a model for next-token prediction
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16, # Loads weights in FP16 (half precision). Reduces memory usage by ~50%, Speeds up inference on GPUs, Trade-off: slight numerical precision loss
                                                 output_attentions=False, # Tells the model to return attention weights. Useful for: interpretability, analyzing token relationships. Each attention tensor: Shape: (layers, heads, seq_len, seq_len)
                                                 token=auth_token, # Required for restricted models like LLaMA
                                                 attn_implementation="sdpa"
                                                 )


    if type(datasets_max_len) == int:
        tokenizer_max_len = len(
            tokenizer(
                prompt_template.format(task_instruction, ""),
                add_special_tokens=False
            )["input_ids"]
        ) + datasets_max_len

        if tokenizer_max_len + max_new_tokens > tokenizer.model_max_length:
            tokenizer_max_len = tokenizer.model_max_length - max_new_tokens

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu' 
    model.to(device) # Transfers all model parameters and buffers to the specified compute device. "cuda" → GPU (typical for FP16)
    model.eval() # Puts the model in inference mode. Ensures deterministic behavior (given fixed generation settings)

    model.generation_config # An object that stores default decoding parameters used by: model.generate(...)

    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")


    """
        This for loop will process all test datasets in data/processed created using: 
            1) a KE method 
            2) a clustering technique 
            3) a similarity technique and 
            4) a similarity threshold
    """

    for data_file in data_files:

        # =============================== Read Data File Contents ========================================
            
        print(f"Data file: {data_file}")
        
        with open(os.path.join(data_path, data_file), "r", encoding='utf-8') as f:
            lines  = f.readlines() 
            data_list = [json.loads(line.strip()) for line in lines] 

        # ================================================================================================


        output_list = [] 
        perkeyphrase_no_tokens = [] # Number of tokens (words, numbers, symbols) each keyphrase has
        perdoc_no_keyphrases = [] # Number of keyphrases extracted by a KE method for some dataset
        
        perdoc_times = []
        perdataset_start_time = time.perf_counter()

        for j_data in tqdm(data_list): 

            doc = f"TITLE: {j_data['title']}. KEYWORDS: {'; '.join(j_data['keyphrases'][:T])}"

            prompt = prompt_template.format(task_instruction, doc) 

            if type(datasets_max_len) == str:
                tokenizer_max_len = len(
                    tokenizer(
                        prompt,
                        add_special_tokens=False
                    )["input_ids"]
                )
        
                if tokenizer_max_len + max_new_tokens > tokenizer.model_max_length:
                    tokenizer_max_len = tokenizer.model_max_length - max_new_tokens
            

            # The input to the LLM will be the entire prompt (instruction and document). Uses the model’s tokenizer to convert prompt (text) into token IDs.
            inputs = tokenizer(prompt, 
                               return_tensors="pt", # Returns PyTorch tensors
                               max_length=tokenizer_max_len, # Caps the sequence at {tokenizer_max_len} tokens.
                               truncation=True # if the prompt exceeds {tokenizer_max_len} tokens, it is cut off (usually from the end, depending on tokenizer settings
                               ).to(device)
            
            perdoc_start_time = time.perf_counter()

            # =============================== KE Process ========================================

            # Give the input to the LLM and generate an output
            with torch.inference_mode(): # Disables gradient computation → faster and lower memory during inference
                outputs = model.generate(**inputs, # Moves tokenized inputs to the same device as the model (e.g., GPU)
                                         max_new_tokens=max_new_tokens, # Limits how many tokens the model generates 
                                         use_cache=True, # Reuses past key/value states → speeds up generation
                                         do_sample=False, # Disables randomness → greedy decoding (deterministic output)
                                         top_p=None,  # No sampling controls are applied (irrelevant since sampling is off)
                                         temperature=None,
                                         max_length=None)
             
            # ==================================================================================
            perdoc_times.append(time.perf_counter() - perdoc_start_time)
            
            # Takes the generated token IDs (outputs[0], the first sequence) and converts them back into human-readable text using the tokenizer
            outputs_str = tokenizer.decode(outputs[0]) 

            generated_output_str = get_generated_output(outputs_str) # The final string of the output will be the extracted keyphrases
            pred_keyphrases_seq = generated_output_str.lower().split('keyphrases:')[-1].strip().rstrip('.')

            log = {} 
            log['final_pred_keyphrase'] = [pred.strip() for pred in pred_keyphrases_seq.split(';')] 
            log['label'] = j_data['label']
            log['generated_output'] = generated_output_str
            log['normalized_label'] = j_data['normalized_label']
            log['prompt'] = prompt
            log['doc'] = doc

            output_list.append(log)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])

        for log in output_list:
            pred_keyphrases_list = log['final_pred_keyphrase']
            perdoc_no_keyphrases.append(len(pred_keyphrases_list))

            for kw in pred_keyphrases_list:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])


        # =============================== Determine Results Path and Create Settings File ========================================

        data_info = data_file.split('_')

        dataset = data_info[0]
        ke_method = data_info[2]

        if len(data_info) == 4 or len(data_info) == 3:
            # Example: nus_MAX4096_YAKE_5 or nus_MAX4096_TopicRank
            clustering_technique = None
            similarity_technique = None
            similarity_threshold = None

        elif len(data_info) == 6:
            # Example: nus_MAX4096_RAKE_HAC_NEb_25
            clustering_technique = data_info[3]
            similarity_technique = data_info[4]
            similarity_threshold = data_info[5].split('.')[0]

        else:
            raise ValueError(f"Unexpected filename format: {data_file}")

        settings = {
            'model_name': f"Llama3_{ke_method}" if clustering_technique is None else f"Llama3_{ke_method}_{clustering_technique}_{similarity_technique}_{int(similarity_threshold*100)}",
            'task_instruction': task_instruction,
            'max_new_tokens': int(max_new_tokens),
            'tokenizer_max_len': "Dynamic" if type(datasets_max_len) == str else tokenizer_max_len,
            'datasets_max_len': datasets_max_len,
            'do_sample': False,
            'T': int(T),
            'ke_method': ke_method,
            'clustering_technique': clustering_technique,
            'similarity_technique': similarity_technique,
            'similarity_threshold': str(similarity_threshold),
        }

        if clustering_technique is None:
            result_path = os.path.join(
                "results",
                "Llama3",
                ke_method,
                str(T),
                f"{timestamp}_custom_{datasets_max_len}",
            )
        else:
            result_path = os.path.join(
                "results",
                "Llama3",
                ke_method,
                str(T),
                clustering_technique,
                similarity_technique,
                str(similarity_threshold),
                f"{timestamp}_custom_{datasets_max_len}",
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


        # =============================== KE Results ========================================

        with open(os.path.join(result_path, f'{dataset}_result.json'), "w", encoding='utf-8') as f: 
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        # =============================== Statistics ========================================

        stats = {
            "KE": f"Llama3_T{T}_{ke_method}" if clustering_technique is None else f"Llama3_T{T}_{ke_method}_{clustering_technique}_{similarity_technique}_{int(similarity_threshold*100)}",
            "Dataset": dataset,
            "T": T,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len,
            "Category": "F",

            "Runtime": {
                "Per_Document": {
                    "Mean": float(np.mean(perdoc_times)),
                    "Median": float(np.median(perdoc_times)),
                    "Min": float(np.min(perdoc_times)),
                    "Max": float(np.max(perdoc_times))
                },
                "Per_Dataset": float(perdataset_end_time - perdataset_start_time)
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


        """
        with open(os.path.join(result_path, f'{dataset}_stats.txt'), "w", encoding='utf-8') as f:

            header = f"\n===== {model_name.split('/')[-1]} T{T}, {ke_method}"

            if clustering_technique is not None:
                header += f", {clustering_technique}, {similarity_technique}, {similarity_threshold}"

            header += f", {dataset}, {timestamp} =====\n"

            f.write(header)
            
            f.write(f'Time (mean) per document: {np.mean(perdoc_times)} sec\n')
            f.write(f'Time (median) per document: {np.median(perdoc_times)} sec\n')
            f.write(f'Time (maximum) per document: {max(perdoc_times)} sec\n')
            f.write(f'Time (minimum) per document: {min(perdoc_times)} sec\n')

            f.write(f'Total time for the entire dataset: {perdataset_end_time - perdataset_start_time} sec\n')

            f.write(f'Minimum number of keywords: {min(perdoc_no_keyphrases)}\n')
            f.write(f'Maximum number of keywords: {max(perdoc_no_keyphrases)}\n')
            f.write(f'Average number of keywords: {np.mean(perdoc_no_keyphrases)}\n')
            f.write(f'Median number of keywords: {np.median(perdoc_no_keyphrases)}\n')

            f.write(f"Average length of keywords: {np.mean(perkeyphrase_no_tokens)}\n")
            f.write(f"Median length of keywords: {np.median(perkeyphrase_no_tokens)}\n")


            f.write(f'Percentage (%) of non-words: {total_non_word_count / total_word_count}\n')
            f.write(f'Average number of words for the entire dataset: {perdataset_avg_no_words}\n')
            #f.write(f'Percentage (%) of non-words multiplied by the maximum number of keyphrases: {(total_non_word_count / total_word_count) * max(perdoc_no_keyphrases)}\n')
            #f.write(f'Average number of words for the entire dataset multiplied by the maximum number of keyphrases: {perdataset_avg_no_words * max(perdoc_no_keyphrases)}\n')
        """