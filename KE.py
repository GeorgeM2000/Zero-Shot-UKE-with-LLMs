import os
import re
import json
import random
import datetime
import argparse
#import pke # Use this library when you want to run the KE methods in pke
import spacy
import yake
import string
import time
import numpy as np


from tqdm import tqdm
from RAKE import Rake, Metric
from Utilities import process_keywords

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ke_method', type=str, default='RAKE', help="The keyword/keyphrase extraction method") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--T', type=int, default=5, required=False, help="Number of keywords to extract")
    args = parser.parse_args() # Parse the arguments

    ke_method = args.ke_method
    data_path = args.data_path
    T = args.T # Set globally for KE methods that require it. Some KE methods extract only T keyphrases/keywords. Others output all candidates with their scores

    if ke_method == 'PositionRank':
        positionrank = spacy.load("en_core_web_sm-3.8.0-py3-none-any/en_core_web_sm/en_core_web_sm-3.8.0")
        positionrank.add_pipe("positionrank")

    elif ke_method == 'KPMiner':
        kpminer_weights_file = r'pke/models/df-semeval2010.tsv.gz'

    elif ke_method == 'YAKE':
        yake_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=T, features=None)

    elif ke_method == 'MPRank':
        stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))




    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join('results', 
                               ke_method if ke_method != "YAKE" else ke_method + str(T), 
                               f'{timestamp}') # Create a folder like: results/RAKE/{timestamp}
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
    
    dataset_list = [#'Inspec', 
                    #'SemEval2017', 
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin'
                    ] 

    for dataset_name in dataset_list:
            
        file_path = os.path.join(data_path, f'{dataset_name}_MAX512.jsonl') # data/processed/{dataset_name}_MAX512.jsonl

        print(f"Dataset: {dataset_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines() # Each line is a document of a specific dataset
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)










        output_list = [] # Keeps all available information for each document of the dataset
        perkeyword_no_tokens = []
        perdoc_no_keywords = []
        perdoc_times = []
        perdataset_start_time = time.perf_counter()

        for j_data in tqdm(data_list): # For JSON data in data_list (for each document)

            doc = j_data['doc'] # Take the document (size of 512 tokens) labeled as 'doc'
            
            # =============================== KE Process ========================================
            perdoc_start_time = time.perf_counter()

            if ke_method == 'RAKE':
                rake_extractor = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
                rake_extractor.extract_keywords_from_text(doc)
                keyphrases = sorted(set(rake_extractor.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True)
                keyphrases = [kw for _,kw in keyphrases]
            
            elif ke_method == 'YAKE':
                keyphrases = yake_extractor.extract_keywords(doc)
                keyphrases = [kw for kw,_ in keyphrases]

            elif ke_method == 'PositionRank':
                positionrank_keyphrases = positionrank(doc)
                keyphrases = [kw.text for kw in positionrank_keyphrases._.phrases[:]]
            
            elif ke_method == 'KPMiner':
                kpminer_extractor = pke.unsupervised.KPMiner()
                kpminer_extractor.load_document(input = doc, language = 'en')
                kpminer_extractor.candidate_selection(lasf = 5, cutoff = 200)
                df = pke.load_document_frequency_file(input_file = kpminer_weights_file)
                kpminer_extractor.candidate_weighting(df = df, alpha = 2.3, sigma = 3.0)
                keyphrases = [kw for kw,_ in kpminer_extractor.get_n_best(n = T)]

            else:
                mprank_extractor = pke.unsupervised.MultipartiteRank()
                mprank_extractor.load_document(input = doc, stoplist = stoplist, language = 'en')
                mprank_extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
                mprank_extractor.candidate_weighting(alpha = 1.1, threshold = 0.74, method = 'average')
                keyphrases = [kw for kw,_ in mprank_extractor.get_n_best(n = T)]
            
            # ==================================================================================

            perdoc_end_time = time.perf_counter()
            perdoc_times.append(perdoc_end_time - perdoc_start_time)
            perdoc_no_keywords.append(len(keyphrases))

            for kw in keyphrases:
                perkeyword_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])







            pred_keyphrases_list = [pred.strip() for pred in keyphrases] # For each keyphrase (splitted by ;) store the keyphrase in pred_keyphrases_list

            log = {} # log keeps all the necessary information of the current document
            log['final_pred_keyphrase'] = pred_keyphrases_list
            log['doc'] = doc
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']

            output_list.append(log)




        perdataset_end_time = time.perf_counter()




        total_word_count, total_non_word_count, avg_word_count = process_keywords([log['final_pred_keyphrase'] for log in output_list])

        file_path = os.path.join(result_path, f'{dataset_name}_result.json') # The results file is located in: results/RAKE/{timestamp}/{dataset_name}_result.json
        
        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


        with open(os.path.join(result_path, f'{dataset_name}_stats.txt'),
                  "w", encoding='utf-8') as f:
            
            f.write(f'\n===== {ke_method if ke_method != "YAKE" else ke_method + str(T)},{dataset_name},{timestamp} =====\n')
            f.write(f'Time (mean) per document: {np.mean(perdoc_times)} sec\n')
            f.write(f'Time (median) per document: {np.median(perdoc_times)} sec\n')
            f.write(f'Time (maximum) per document: {max(perdoc_times)} sec\n')
            f.write(f'Time (minimum) per document: {min(perdoc_times)} sec\n')

            f.write(f'Total time for dataset: {perdataset_end_time - perdataset_start_time} sec\n')

            f.write(f'Minimum number of keywords: {min(perdoc_no_keywords)}\n')
            f.write(f'Maximum number of keywords: {max(perdoc_no_keywords)}\n')
            f.write(f'Average number of keywords: {np.mean(perdoc_no_keywords)}\n')
            f.write(f'Median number of keywords: {np.median(perdoc_no_keywords)}\n')

            f.write(f"Average length of keywords: {np.mean(perkeyword_no_tokens)}\n")


            f.write(f'Percentage (%) of non-words: {total_non_word_count / total_word_count}\n')
            f.write(f'Average word count: {avg_word_count}\n')

        