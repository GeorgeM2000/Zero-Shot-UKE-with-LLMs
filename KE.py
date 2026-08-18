import os
import json
import datetime
import argparse
import pytextrank # Required if you want to use PositionRank, TextRank, TopicRank
import pke # Required if you want to use KPMiner, MPRank, and other KE methods 
import spacy
import yake
import string
import time
import numpy as np

from tqdm import tqdm
from RAKE import Rake, Metric
from Utilities import process_keyphrases


"""
Add the following method to the base.py file in the pke package after you download it using pip inside a venv (virtual enviroment). 

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
    parser.add_argument('--T', type=str, default='10', help="Number of keywords/keyphrases to extract")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    args = parser.parse_args() # Parse the arguments

    ke_method = args.ke_method
    data_path = args.data_path
    T = int(args.T) # Required for some KE methods. Some KE methods extract only T keyphrases/keywords. Others output all candidate keyphrases/keywords with their scores
    datasets_max_len = args.datasets_max_len
    spacy_model_path = "../en_core_web_sm-3.8.0-py3-none-any/en_core_web_sm/en_core_web_sm-3.8.0"

    if ke_method == 'PositionRank':
        positionrank = spacy.load(spacy_model_path)
        positionrank.add_pipe("positionrank")

    elif ke_method == 'TextRank':
        textrank = spacy.load(spacy_model_path)
        textrank.add_pipe("textrank")

    elif ke_method == 'TopicRank':
        topicrank = spacy.load(spacy_model_path)
        topicrank.add_pipe("topicrank")

    elif ke_method == 'KPMiner':
        kpminer_spacy = spacy.load(spacy_model_path)
        kpminer_weights_file = r'/home/georgematlis/Keyword_Extraction/lib/python3.12/site-packages/pke/models/df-semeval2010.tsv.gz' # Alternative: r'../pke/models/df-semeval2010.tsv.gz'

    elif ke_method == 'YAKE': # YAKE extracts 20 keywords/keyphrases by default
        yake_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=T, features=None)

    elif ke_method == 'MPRank':
        mprank_spacy = spacy.load(spacy_model_path)
        stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))


    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join('results', 
                               f"{ke_method}/{ke_method}" if ke_method != 'YAKE' else f"{ke_method}/{ke_method}_T{T}", 
                               f'{timestamp}_{datasets_max_len}') # Create a folder like: results/RAKE/RAKE/{timestamp}_{datasets_max_len} or results/YAKE/YAKE_T5/{timestamp}_{datasets_max_len}
    
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
        
        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}.jsonl'), "r", encoding='utf-8') as f: # data/processed/{dataset_name}_MAX{datasets_max_len}.jsonl
            lines = f.readlines() # Each line is a document of a specific dataset

            # Each line is a dictionary
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)


        output_list = []
        perkeyphrase_no_tokens = [] # Number of tokens (words, numbers, symbols) each keyphrase has
        perdoc_no_keyphrases = [] # Number of keyphrases extracted by a KE method for some dataset

        perdoc_times = []
        perdataset_start_time = time.perf_counter()

        for j_data in tqdm(data_list): 

            doc = j_data['doc'] 

            perdoc_start_time = time.perf_counter()

            # =============================== KE Process ========================================
            
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

            elif ke_method == 'TextRank':
                textrank_keyphrases = textrank(doc)
                keyphrases = [kw.text for kw in textrank_keyphrases._.phrases[:]]

            elif ke_method == 'TopicRank':
                topicrank_keyphrases = topicrank(doc)
                keyphrases = [kw.text for kw in topicrank_keyphrases._.phrases[:]]
            
            elif ke_method == 'KPMiner':
                kpminer_extractor = pke.unsupervised.KPMiner()
                kpminer_extractor.load_document(input = doc, language = 'en', spacy_model=kpminer_spacy)
                kpminer_extractor.candidate_selection(lasf = 5, cutoff = 200)
                df = pke.load_document_frequency_file(input_file = kpminer_weights_file)
                kpminer_extractor.candidate_weighting(df = df, alpha = 2.3, sigma = 3.0)
                keyphrases = [kw for kw,_ in kpminer_extractor.get_all_sorted()]

            else:
                mprank_extractor = pke.unsupervised.MultipartiteRank()
                mprank_extractor.load_document(input = doc, stoplist = stoplist, language = 'en', spacy_model=mprank_spacy)
                mprank_extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
                mprank_extractor.candidate_weighting(alpha = 1.1, threshold = 0.74, method = 'average')
                keyphrases = [kw for kw,_ in mprank_extractor.get_all_sorted()]
            
            # ==================================================================================

            perdoc_times.append(time.perf_counter() - perdoc_start_time)

            log = {} 
            log['final_pred_keyphrase'] = [pred.strip() for pred in keyphrases]
            log['label'] = j_data['label']
            log['normalized_label'] = j_data['normalized_label']
            log['doc'] = doc
            
            if ke_method in ['YAKE', 'TopicRank']: 
                log['title'] = j_data['title']
            
            output_list.append(log)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])

        for log in output_list:
            keyphrases = log['final_pred_keyphrase']
            perdoc_no_keyphrases.append(len(keyphrases))

            for kw in keyphrases:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])
        
        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f: # The results file is located in: results/RAKE/RAKE/{timestamp}_{datasets_max_len}/{dataset_name}_result.json
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        if ke_method in ['YAKE', 'TopicRank']:

            jsonl_lines = []
            for log in output_list:
                line = {}
                line['title'] = log['title']
                line['keyphrases'] = log['final_pred_keyphrase']
                line['label'] = log['label']
                line['normalized_label'] = log['normalized_label']
                jsonl_lines.append(line)

            with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}_{ke_method}_{T}.jsonl' if ke_method == 'YAKE' else f'{dataset_name}_MAX{datasets_max_len}_{ke_method}.jsonl'), "w", encoding='utf-8') as f:
                for json_data in jsonl_lines:
                    f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        stats = {
            "KE": ke_method,
            "Dataset": dataset_name,
            "T": T,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len,
            "Category": "A",

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
            
            f.write(f'\n===== {ke_method if ke_method not in ['YAKE', 'TopicRank'] else ke_method + str(T)},{dataset_name},{timestamp} =====\n')
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

        