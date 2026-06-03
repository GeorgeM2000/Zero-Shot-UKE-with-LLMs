import os
import re
import json
import argparse
from tqdm import tqdm
import datetime
import nltk
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

nltk.download('stopwords')
stopword_dict = set(stopwords.words('english'))

GRAMMAR = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


def extract_candidates(tokens_tagged, no_subset=False, enable_filter=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    cans_count = dict()
    
    np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            
            if len(np.split()) == 1:
                if np not in cans_count.keys():
                    cans_count[np] = 0
                cans_count[np] += 1
                
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1
        
    if enable_filter == True:
        i = 0
        while i < len(keyphrase_candidate):
            can, pos = keyphrase_candidate[i]
            #pos[0] > 50 and
            if can in cans_count.keys() and cans_count[can] == 1:
                keyphrase_candidate.pop(i)
                continue
            i += 1
    
    return keyphrase_candidate


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extract_candidates(self.tokens_tagged, en_model)

def remove_candidate (text):
    text_len = len(text.split())
    remove_chars = '[’!"#$%&\'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, '', text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True 
    else:
        return False

def get_generated_output(str):
    split_prompt_output = str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
    return split_prompt_output[-1].strip().replace("<|eot_id|>", "")



if __name__ == '__main__':

    task1_instruction = "You are a keyphrase extractor. Extract keyphrases from the text. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase N'"
    task2_instruction = "You are a keyphrase extractor. Extract top 5 keyphrases from the 'Keyphrase candidates' consisting of noun phrases extracted from the text. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase N'"
    task3_instruction = "You are a keyphrase extractor. Extract keyphrases from the 'Keyphrase candidates'. The answer should be listed after 'Keyphrases: ' and separated by semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase N'"
    task1_prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"
    task2_prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}\n\nKeyphrase candidates: {}<|eot_id|>"
    task3_prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}\n\nKeyphrase candidates: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--task1_instruction', type=str, default=task1_instruction, help="Vanilla prompt")
    parser.add_argument('--task2_instruction', type=str, default=task2_instruction, help="Candidate-based prompt")
    parser.add_argument('--task3_instruction', type=str, default=task3_instruction, help="Hybrid prompt")
    parser.add_argument('--max_new_tokens', type=str, default='128', help="Maximum number of tokens to generate")
    parser.add_argument('--cuda', type=str, default='0', help="GPU")
    parser.add_argument('--core_nlp_path', type=str, default='stanford-corenlp-full-2018-02-27', help="Your StanfordCoreNLP path")
    parser.add_argument('--auth_token', type=str, default='', help="auth_token for Llama")

    args = parser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    task1_instruction = args.task1_instruction
    task2_instruction = args.task2_instruction
    task3_instruction = args.task3_instruction
    max_new_tokens = int(args.max_new_tokens)
    other_max_new_tokens = 512
    StanfordCoreNLP_path = args.core_nlp_path
    auth_token = args.auth_token


    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, output_attentions=True, token=auth_token)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    model.generation_config

    settings = {
        'model_name': model_name,
        'task1_instruction': task1_instruction,
        'task2_instruction': task2_instruction,
        'task3_instruction': task3_instruction, 
        'max_new_tokens': max_new_tokens,
        'tokenizer_max_len': 4096,
        'max_len': 512,
        'do_sample': False
    }

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join('results', model_name.split('/')[-1], f'{timestamp}_hybrid')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
    settings_path = os.path.join(result_path, 'settings.json')
    with open(settings_path, 'w') as settings_file:
        json.dump(settings, settings_file, indent=4)

    print(f"Settings saved to {settings_path}")

    en_model = StanfordCoreNLP(StanfordCoreNLP_path, quiet=True)

    dataset_list = ['Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin'] 

    for dataset_name in dataset_list:
            
        file_path = os.path.join(data_path,f'{dataset_name}_MAX512.jsonl')

        print(f"Dataset: {dataset_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines()
            data_list = [json.loads(line.strip()) for line in lines]


        output_list = []

        for j_data in tqdm(data_list):

            doc = j_data['doc']
            prompt1 = task1_prompt_template.format(task1_instruction, doc)

            inputs = tokenizer(prompt1, return_tensors="pt", max_length=4096, truncation=True)

            with torch.no_grad():
                outputs = model.generate(**inputs.to(device), max_new_tokens=max_new_tokens, use_cache=True, do_sample=False, top_p=None, temperature=None )
            outputs_str = tokenizer.decode(outputs[0])

            generated_output1_str = get_generated_output(outputs_str)
            #print(generated_output1_str)

            pred_keyphrases_seq = generated_output1_str.lower().split('keyphrases:')[-1].strip().rstrip('.')
            pred_keyphrases_list = [ pred.strip() for pred in pred_keyphrases_seq.split(';') ]



            text_obj = InputTextObj(en_model, doc)

            cans = text_obj.keyphrase_candidate
            candidates = []
            for can, _ in cans:
                if remove_candidate(can):
                    continue
                candidates.append(can.lower())

            candidates = list(set(candidates))

            candidates_seq = ' ; '.join(candidates)
            
            prompt2 = task2_prompt_template.format(task2_instruction, doc, candidates_seq)
            
            inputs = tokenizer(prompt2, return_tensors="pt", max_length=4096, truncation=True)

            with torch.no_grad():
                outputs = model.generate(**inputs.to(device), max_new_tokens=max_new_tokens, use_cache=True, do_sample=False, top_p=None, temperature=None )
            outputs_str = tokenizer.decode(outputs[0])


            generated_output2_str = get_generated_output(outputs_str)
           
            #print(generated_output2_str)

            pred_candidates_seq = generated_output2_str.lower().split('keyphrases:')[-1].strip().rstrip('.')
            pred_candidates_list = [ pred.strip() for pred in pred_candidates_seq.split(';') ]


            pred_with_candidates = pred_keyphrases_list + pred_candidates_list


            pred_with_candidates_seq = ' ; '.join(pred_with_candidates)
            
            prompt3 = task3_prompt_template.format(task3_instruction, doc, pred_with_candidates_seq)

            inputs = tokenizer(prompt3, return_tensors="pt", max_length=4096, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(**inputs.to(device), max_new_tokens=max_new_tokens, use_cache=True, do_sample=False, top_p=None, temperature=None )
            outputs_str = tokenizer.decode(outputs[0])


            generated_output3_str = get_generated_output(outputs_str)

            #print(generated_output3_str)

            final_pred_keyphrases_seq = generated_output3_str.lower().split('keyphrases:')[-1].strip().rstrip('.')
            final_pred_keyphrases_list = [ pred.strip() for pred in final_pred_keyphrases_seq.split(';') ]


            log = {}
            log['prompt1'] = prompt1
            log['prompt2'] = prompt2
            log['prompt3'] = prompt3
            log['generated_ourput1'] = generated_output1_str
            log['generated_ourput2'] = generated_output2_str
            log['generated_ourput3'] = generated_output3_str
            log['first_pred_keyphrase'] = pred_keyphrases_list
            log['pred_candidate'] = pred_candidates_list
            log['final_pred_keyphrase'] = final_pred_keyphrases_list
            log['doc'] = doc
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']

            output_list.append(log)

        file_path = os.path.join(result_path, f'{dataset_name}_result.json')
        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in output_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
