import os
import json
import datetime
import argparse
import torch
import time
import numpy as np

from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from Utilities import process_keyphrases

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity_error()

"""
Some information:

output_attentions=True could let you: 
    analyze which keywords influenced generation and potentially rank importance (advanced idea)

tokenizer(prompt, max_length=4096, truncation=True)

    What this actually means
        You are telling the tokenizer:
            ➤ “If the input exceeds 4096 tokens, cut it down to 4096.”
    
    What it does NOT mean
        ❌ It does not change the tokenizer’s inherent max length
        ❌ It does not guarantee the model supports 4096 tokens


    The real limits

        There are two separate concepts:

            Tokenizer limit

                Often stored as: tokenizer.model_max_length

            Model context window (critical)
                
                Defined by the model (e.g., LLaMA variants)
                Typical values: 2048, 4096, 8192, etc.

"""



def get_generated_output(str):
    split_prompt_output = str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
    return split_prompt_output[-1].strip().replace("<|eot_id|>", "")



if __name__ == '__main__':

    
    # We can use this prompt template without making any further modifications
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--max_new_tokens', type=str, default='64', help="Maximum number of tokens to generate")
    parser.add_argument('--cuda', type=str, default='0', help="GPU") # If there is a GPU, it is labeled as 0
    parser.add_argument('--auth_token', type=str, help="Authentication token for Llama") 
    parser.add_argument('--T', type=str, default='10', help="Number of keyphrases to extract")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    args = parser.parse_args() 

    model_name = args.model_name
    data_path = args.data_path
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    T = int(args.T)
    datasets_max_len = args.datasets_max_len if args.datasets_max_len == "FULL" else int(args.datasets_max_len)
    
    # The task instruction is the most critical part of this process. The instruction can be changed depending on the task
    task_instruction = f"You are a keyphrase extractor. Extract {T} keyphrases from the text. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"

    # Loads a pretrained tokenizer associated with model_name. The tokenizer converts raw text → tokens (integer IDs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              token=auth_token) # AutoTokenizer: A factory class that automatically selects the correct tokenizer type. For LLaMA, this is typically a SentencePiece-based tokenizer
    
    # AutoModelForCausalLM Loads a model for next-token prediction
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


    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu' # No way we are using CPU
    model.to(device) # Transfers all model parameters and buffers to the specified compute device. "cuda" → GPU (typical for FP16)
    model.eval() # Puts the model in inference mode. Ensures deterministic behavior (given fixed generation settings)

    model.generation_config # An object that stores default decoding parameters used by: model.generate(...)



    settings = {
        'model_name': "Llama3",
        'task_instruction': task_instruction,
        'max_new_tokens': max_new_tokens, # How long the output will be
        'tokenizer_max_len': tokenizer_max_len if type(datasets_max_len) == int else "Dynamic", # Alternatives: 8192 4096
        'datasets_max_len': datasets_max_len, # Original: 512 # Alternatives: 4096 2048
        'do_sample': False, # do_sample = False → deterministic (greedy / beam search). do_sample = True → stochastic (sampling)
        'T': T
    }

    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join('results', 
                               f"Llama3/Llama3_T{T}", 
                               f'{timestamp}_vanilla_{datasets_max_len}') # Create a folder like: results/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_T5/{timestamp}_vanilla_{datasets_max_len}
    
    print(f"Results path: {result_path}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
    # Create a settings file
    settings_path = os.path.join(result_path, 'settings.json') # results/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct_T5/{timestamp}_vanilla_{datasets_max_len}/settings.json
    with open(settings_path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

    print(f"Settings saved to {settings_path}")


    dataset_list = [#'Inspec', 
                    #'SemEval2017', 
                    'MDPI',
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin'
                    ] 

    for dataset_name in dataset_list:
            
        print(f"Dataset: {dataset_name}")
        
        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}.jsonl'), "r", encoding='utf-8') as f: # data/processed/{dataset_name}_MAX{datasets_max_len}.jsonl
            lines  = f.readlines() 
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)


        output_list = [] 
        perkeyphrase_no_tokens = [] # Number of tokens (words, numbers, symbols) each keyphrase has
        perdoc_no_keyphrases = [] # Number of keyphrases extracted by a KE method for some dataset
        
        perdoc_times = []
        perdataset_start_time = time.perf_counter()
        
        for j_data in tqdm(data_list): 

            perdoc_start_time = time.perf_counter()

            doc = j_data['doc'] 
            prompt = prompt_template.format(task_instruction, doc) # Use the prompt template to insert the document into the prompt and the task instruction
            

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

            # =============================== KE Process ========================================

            # Give the input to the LLM and generate an output
            with torch.inference_mode(): # with torch.no_grad(): # Disables gradient computation → faster and lower memory during inference
                outputs = model.generate(**inputs, # Moves tokenized inputs to the same device as the model (e.g., GPU)
                                         max_new_tokens=max_new_tokens, # Limits how many tokens the model generates 
                                         use_cache=True, # Reuses past key/value states → speeds up generation
                                         do_sample=False, # Disables randomness → greedy decoding (deterministic output)
                                         top_p=None,  # No sampling controls are applied (irrelevant since sampling is off)
                                         max_length=None,
                                         temperature=None)
             
            # ==================================================================================

            
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
            perdoc_times.append(time.perf_counter() - perdoc_start_time)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])

        for log in output_list:
            pred_keyphrases_list = log['final_pred_keyphrase']
            perdoc_no_keyphrases.append(len(pred_keyphrases_list))

            for kw in pred_keyphrases_list:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])


        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f: # The results file is located in: results/Meta-Llama-3-8B-Instruct/{timestamp}/{dataset_name}_result.json
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        stats = {
            "KE": f"Llama3_T{T}",
            "Dataset": dataset_name,
            "T": T,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len,
            "Category": "E",
            

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

        with open(os.path.join(result_path, f"{dataset_name}_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        """
        with open(os.path.join(result_path, f'{dataset_name}_stats.txt'), "w", encoding='utf-8') as f:
            
            f.write(f'\n===== {model_name.split('/')[-1] + " T" + str(T)},{dataset_name},{timestamp} =====\n')
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