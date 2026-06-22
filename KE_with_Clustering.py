import os
import json
import datetime
import argparse
import pke # Use this library when you want to run the KE methods in pke
import spacy
import string
import time
import numpy as np


from tqdm import tqdm
from RAKE import Rake, Metric
from Utilities import process_keyphrases, cluster_keywords, cluster_keywords_embeddings
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer  




"""
Add the following function to the base.py file in the pke package. 

def get_all_sorted(self, redundancy_removal=False, stemming=False):
        \"""Returns all candidates sorted by descending weight.

        Args:
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the results, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), defaults to
                False.
        \"""

        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates
            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:

                # test whether candidate is redundant
                if self.is_redundant(candidate, non_redundant_best):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of all candidates as (lexical form, weight) tuples
        all_candidates = [(u, self.weights[u]) for u in best]

        # replace with surface forms if no stemming
        if not stemming:
            all_candidates = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                            self.weights[u]) for u in best]

        # return the sorted list of all candidates
        return all_candidates

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ke_method', type=str, default='RAKE', help="The keyword/keyphrase extraction method") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--similarity_technique', type=str, default='NEb', help="Similarity technique (embedding-based or non-embedding-based)") # Non-Embedding-based or Embedding-based
    parser.add_argument('--similarity_threshold', type=str, default='0.25', help="Similarity threshold for HAC")
    parser.add_argument('--datasets_max_len', type=str, default='512', help="Maximum length of test datasets")
    args = parser.parse_args() # Parse the arguments


    sim_threshold = float(args.similarity_threshold)
    sim_technique = args.similarity_technique
    ke_method = args.ke_method # The keyword/keyphrase extraction method with agglomerative clustering
    data_path = args.data_path
    datasets_max_len = int(args.datasets_max_len)

    if ke_method == 'PositionRank':
        positionrank = spacy.load("../en_core_web_sm-3.8.0-py3-none-any/en_core_web_sm/en_core_web_sm-3.8.0")
        positionrank.add_pipe("positionrank")

    elif ke_method == 'KPMiner':
        kpminer_weights_file = r'/home/georgematlis/Keyword_Extraction/lib/python3.12/site-packages/pke/models/df-semeval2010.tsv.gz' #r'../pke/models/df-semeval2010.tsv.gz'

    elif ke_method == 'MPRank': 
        stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))
        
    # Load only once
    if sim_technique == 'Eb': 
        embedding_model = SentenceTransformer('all-mpnet-base-v2')


    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join('results', 
                               f"{ke_method}/HAC_{sim_technique}_{sim_threshold*100}", 
                               f'{timestamp}') # Create a folder like: results/RAKE/{timestamp}
    

    print(f"Results path: {result_path}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory created: {result_path}")
    
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
        
        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}.jsonl'), 'r', encoding='utf-8') as f: # data/processed/{dataset_name}_MAX512.jsonl
            lines  = f.readlines() # Each line is a document of a specific dataset
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)

        output_list = [] # Keeps all available information for each document of the dataset
        perkeyphrase_no_tokens = []
        perdoc_no_keyphrases = []
        perdoc_times = []
        perdataset_start_time = time.perf_counter()

        for j_data in tqdm(data_list): # For JSON data in data_list (for each document)

            doc = j_data['doc'] # Take the document (size of 512 tokens) labeled as 'doc'

            perdoc_start_time = time.perf_counter()
            # =============================== KE Process ========================================

            if ke_method == 'RAKE':
                rake_extractor = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
                rake_extractor.extract_keywords_from_text(doc)
                keyphrases = sorted(set(rake_extractor.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True)
                keyphrases = [kw for _,kw in keyphrases]

            elif ke_method == 'PositionRank':
                positionrank_keyphrases = positionrank(doc)
                keyphrases = [kw.text for kw in positionrank_keyphrases._.phrases[:]]
            
            elif ke_method == 'KPMiner':
                kpminer_extractor = pke.unsupervised.KPMiner()
                kpminer_extractor.load_document(input = doc, language = 'en')
                kpminer_extractor.candidate_selection(lasf = 5, cutoff = 200)
                df = pke.load_document_frequency_file(input_file = kpminer_weights_file)
                kpminer_extractor.candidate_weighting(df = df, alpha = 2.3, sigma = 3.0)
                keyphrases = [kw for kw,_ in kpminer_extractor.get_all_sorted()]

            else:
                mprank_extractor = pke.unsupervised.MultipartiteRank()
                mprank_extractor.load_document(input = doc, stoplist = stoplist, language = 'en')
                mprank_extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
                mprank_extractor.candidate_weighting(alpha = 1.1, threshold = 0.74, method = 'average')
                keyphrases = [kw for kw,_ in mprank_extractor.get_all_sorted()]
     
            # ===================================================================================

        
            # =============================== Clustering Process ========================================
            if sim_technique == 'Eb': 
                # Generate an embedding for each keyword/keyphrase 
                keyphrase_embeddings = embedding_model.encode(keyphrases, 
                                                              convert_to_numpy=True, 
                                                              show_progress_bar=False) 
                
                # Optional: convert to a list of embeddings instead of a NumPy array 
                keyphrase_embeddings = list(keyphrase_embeddings)


                keyphrases = cluster_keywords_embeddings(
                    keyphrases,
                    keyphrase_embeddings,
                    sim_threshold
                )
            else:
                keyphrases = cluster_keywords(keyphrases, 
                                            [' '.join(PorterStemmer().stem(token.lower()) for token in word_tokenize(kw)) for kw in keyphrases],
                                            sim_threshold)

            # ===========================================================================================

            perdoc_end_time = time.perf_counter()
            perdoc_times.append(perdoc_end_time - perdoc_start_time)
            perdoc_no_keyphrases.append(len(keyphrases))

            for kw in keyphrases:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])


            pred_keyphrases_list = [pred.strip() for pred in keyphrases] # For each keyphrase (splitted by ;) store the keyphrase in pred_keyphrases_list


            log = {} # log keeps all the necessary information of the current document
            log['final_pred_keyphrase'] = pred_keyphrases_list
            log['doc'] = doc
            log['doc_title'] = j_data['title']
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']
            output_list.append(log)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])

        
        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f: # The results file is located in: results/RAKE/{timestamp}/{dataset_name}_result.json
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        jsonl_lines = []
        for log in output_list:
            line = {}
            line['doc_title'] = log['doc_title']
            line['doc_keyphrases'] = log['final_pred_keyphrase']
            jsonl_lines.append(line)


        with open(os.path.join(data_path, 
                               f'{dataset_name}_MAX{datasets_max_len}_{ke_method}_HAC_{sim_technique}_{sim_threshold*100}.jsonl'), 
                               "w", encoding='utf-8') as f:
            
            for json_data in jsonl_lines:
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        with open(os.path.join(result_path, f'{dataset_name}_stats.txt'), "w", encoding='utf-8') as f:
            
            f.write(f'\n===== {ke_method} + HAC(Technique={sim_technique}, Similarity={sim_threshold}),{dataset_name},{timestamp} =====\n')
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

        