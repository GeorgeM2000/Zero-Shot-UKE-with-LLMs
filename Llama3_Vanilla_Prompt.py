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
from Utilities import process_keyphrases

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
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path") # So if i use/download the Llama LLM, it will be located in a folder named meta-llama
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    #parser.add_argument('--task_instruction', type=str, default=task_instruction, help="Vanilla prompt")
    parser.add_argument('--max_new_tokens', type=str, default='128', help="Maximum number of tokens to generate")
    parser.add_argument('--cuda', type=str, default='0', help="GPU") # If there is a GPU, it is labeled as 0
    parser.add_argument('--auth_token', type=str, default='', help="Authentication token for Llama") # Very important. Question: If llama is downloaded locally, why does it need an auth. token?
    parser.add_argument('--T', type=str, default='5', help="Number of keywords to extract")
    parser.add_argument('--datasets_max_len', type=str, default='512', help="Maximum length of test datasets")
    parser.add_argument('--tokenizer_max_len', type=str, default='4096', help="Maximum length of tokenizer")
    args = parser.parse_args() # Parse the arguments

    model_name = args.model_name
    data_path = args.data_path
    #task_instruction = args.task_instruction
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    T = int(args.T)
    datasets_max_len = int(args.datasets_max_len)
    tokenizer_max_len = int(args.tokenizer_max_len)

    # The task instruction is the most critical part of this evaluation. The instruction can be changed depending on the task
    task_instruction = f"You are a keyphrase extractor. Extract {T} keyphrases from the text. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase N'"
    
    # Loads a pretrained tokenizer associated with model_name. The tokenizer converts raw text → tokens (integer IDs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token) # AutoTokenizer: A factory class that automatically selects the correct tokenizer type. For LLaMA, this is typically a SentencePiece-based tokenizer
    
    # AutoModelForCausalLM Loads a model for next-token prediction
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16, # Loads weights in FP16 (half precision). Reduces memory usage by ~50%, Speeds up inference on GPUs, Trade-off: slight numerical precision loss
                                                 output_attentions=True, # Tells the model to return attention weights. Useful for: interpretability, analyzing token relationships. Each attention tensor: Shape: (layers, heads, seq_len, seq_len)
                                                 token=auth_token # Required for restricted models like LLaMA
                                                 )

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu' # No way we are using CPU
    model.to(device) # Transfers all model parameters and buffers to the specified compute device. "cuda" → GPU (typical for FP16)
    model.eval() # Puts the model in inference mode. Ensures deterministic behavior (given fixed generation settings)

    model.generation_config # An object that stores default decoding parameters used by: model.generate(...)

    settings = {
        'model_name': model_name,
        'task_instruction': task_instruction,
        'max_new_tokens': max_new_tokens, # How long the output will be
        'tokenizer_max_len': tokenizer_max_len, # Alternatives: 8192 4096
        'max_len': datasets_max_len, # Original: 512 # Alternatives: 4096 2048
        'do_sample': False, # do_sample = False → deterministic (greedy / beam search). do_sample = True → stochastic (sampling)
        'T': T
    }

    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join('results', 
                               f"{model_name.split('/')[-1]}/{model_name.split('/')[-1]}_T{T}", 
                               f'{timestamp}_vanilla') # Create a folder like: results/Meta-Llama-3-8B-Instruct/{timestamp}
    
    print(f"Results path: {result_path}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
    # Create a settings file
    settings_path = os.path.join(result_path, 'settings.json') # results/Meta-Llama-3-8B-Instruct/{timestamp}/settings.json
    with open(settings_path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

    print(f"Settings saved to {settings_path}")


    dataset_list = [#'Inspec', 
                    #'SemEval2017', 
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin'
                    ] 

    for dataset_name in dataset_list:
            
        print(f"Dataset: {dataset_name}")
        
        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}.jsonl'), 'r', encoding='utf-8') as f: # data/processed/{dataset_name}_MAX512.jsonl
            lines  = f.readlines() # Each line is a document of a specific dataset
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)


        output_list = [] # Keeps all available information for each document of a dataset
        perkeyphrase_no_tokens = []
        perdoc_no_keyphrases = []
        perdoc_times = []
        perdataset_start_time = time.perf_counter()
        
        for j_data in tqdm(data_list): # For JSON data in data_list (for each document)

            doc = j_data['doc'] # Take the document (size of 512 tokens) labeled as 'doc'
            prompt = prompt_template.format(task_instruction, doc) # Use the prompt template to insert the document into the prompt and the task instruction
            

            # The input to the LLM will be the entire prompt (instruction and document). Uses the model’s tokenizer to convert prompt (text) into token IDs.
            inputs = tokenizer(prompt, 
                               return_tensors="pt", # Returns PyTorch tensors
                               max_length=tokenizer_max_len, # Caps the sequence at 4096 tokens.
                               truncation=True # if the prompt exceeds 4096 tokens, it is cut off (usually from the end, depending on tokenizer settings
                               ) 
            
            perdoc_start_time = time.perf_counter()

            # =============================== KE Process ========================================

            # Give the input to the LLM and generate an output
            with torch.no_grad(): # Disables gradient computation → faster and lower memory during inference
                outputs = model.generate(**inputs.to(device), # Moves tokenized inputs to the same device as the model (e.g., GPU)
                                         max_new_tokens=max_new_tokens, # Limits how many tokens the model generates 
                                         use_cache=True, # Reuses past key/value states → speeds up generation
                                         do_sample=False, # Disables randomness → greedy decoding (deterministic output)
                                         top_p=None,  # No sampling controls are applied (irrelevant since sampling is off)
                                         temperature=None)
             
            # ==================================================================================

            perdoc_end_time = time.perf_counter()
            perdoc_times.append(perdoc_end_time - perdoc_start_time)
            
            

            # Takes the generated token IDs (outputs[0], the first sequence) and converts them back into human-readable text using the tokenizer
            outputs_str = tokenizer.decode(outputs[0]) # Use the tokenizer to decode the output

            generated_output_str = get_generated_output(outputs_str) # The final string of the output will be the extracted keyphrases
            #print(generated_output_str)

            pred_keyphrases_seq = generated_output_str.lower().split('keyphrases:')[-1].strip().rstrip('.')
            pred_keyphrases_list = [ pred.strip() for pred in pred_keyphrases_seq.split(';') ] # For each keyphrase (splitted by ;) store the keyphrase in pred_keyphrases_list

            perdoc_no_keyphrases.append(len(pred_keyphrases_list))

            for kw in pred_keyphrases_list:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])

            log = {} # log keeps all the necessary information of the current document
            log['prompt'] = prompt
            log['generated_output'] = generated_output_str
            log['final_pred_keyphrase'] = pred_keyphrases_list
            log['doc'] = doc
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']

            output_list.append(log)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])



        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f: # The results file is located in: results/Meta-Llama-3-8B-Instruct/{timestamp}/{dataset_name}_result.json
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')



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
            f.write(f'Percentage (%) of non-words multiplied by the maximum number of keyphrases: {(total_non_word_count / total_word_count) * max(perdoc_no_keyphrases)}\n')
            f.write(f'Average number of words for the entire dataset multiplied by the maximum number of keyphrases: {perdataset_avg_no_words * max(perdoc_no_keyphrases)}\n')
