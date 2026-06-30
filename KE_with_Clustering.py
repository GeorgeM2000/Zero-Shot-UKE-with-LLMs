import os
import json
import datetime
import argparse
import pytextrank
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
Add the following method to the base.py file in the pke package after you download it using pip inside a venv. 

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


# -----------------------------------------------------------------------
# Lemmatization helper (replaces PorterStemmer).
#
# spaCy lemmatization is linguistically correct (dictionary/POS-aware)
# rather than rule-based suffix stripping, so it avoids the occasional
# garbage stems Porter produces, while still collapsing inflected variants
# ("networks" -> "network", "optimizing"/"optimization" -> "optimize")
# onto a shared root for the word-overlap similarity metric to work well.
#
# Reuses the spaCy model already loaded elsewhere in the pipeline
# (spacy_model_path / nlp), so no extra model load cost.
# -----------------------------------------------------------------------
def lemmatize_keywords(keywords, nlp):
    """
    Lemmatize a list of keyword/keyphrase strings using spaCy.

    Args:
        keywords (list of str): Original keyword/keyphrase strings.
        nlp: A loaded spaCy Language object (e.g. spacy.load(spacy_model_path)).

    Returns:
        list of str: Space-joined lemmatized tokens per keyword, e.g.
            "neural networks" -> "neural network"
            "optimizing performance" -> "optimize performance"
    """
    # nlp.pipe batches the keywords through spaCy's pipeline efficiently,
    # rather than calling nlp(kw) once per keyword in a Python loop.
    lemmatized = []
    for doc in nlp.pipe(keywords):
        lemmas = [token.lemma_.lower() for token in doc if not token.is_space]
        lemmatized.append(" ".join(lemmas))
    return lemmatized



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ke_method', type=str, default='RAKE', help="The keyword/keyphrase extraction method") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--similarity_technique', type=str, default='NEb', help="Similarity technique (embedding-based or non-embedding-based)") 
    parser.add_argument('--similarity_threshold', type=str, default='0.25', help="Similarity threshold for HAC")
    parser.add_argument('--datasets_max_len', type=str, default='512', help="Maximum length of test datasets")
    args = parser.parse_args() # Parse the arguments


    sim_threshold = float(args.similarity_threshold)
    sim_technique = args.similarity_technique
    ke_method = args.ke_method # The keyword/keyphrase extraction method with HAC
    data_path = args.data_path
    datasets_max_len = int(args.datasets_max_len)
    spacy_model_path = "../en_core_web_sm-3.8.0-py3-none-any.whl_FILES/en_core_web_sm/en_core_web_sm-3.8.0"
    spacy_model = spacy.load(spacy_model_path)
    #stemmer = PorterStemmer()



    if ke_method == 'PositionRank':
        #positionrank = spacy.load(spacy_model_path)
        #positionrank.add_pipe("positionrank")
        spacy_model.add_pipe("positionrank")

    elif ke_method == 'KPMiner':
        #kpminer_spacy = spacy.load(spacy_model_path)
        kpminer_weights_file = r'/home/georgematlis/Keyword_Extraction/lib/python3.12/site-packages/pke/models/df-semeval2010.tsv.gz' # Alternative: r'../pke/models/df-semeval2010.tsv.gz'

    elif ke_method == 'MPRank': 
        #mprank_spacy = spacy.load(spacy_model_path)
        stoplist = list(string.punctuation) + list(pke.lang.stopwords.get('en'))
        
    # Load only once
    if sim_technique == 'Eb': # Embedding-based (Eb) Non-Embedding-based (NEb)
        embedding_model = SentenceTransformer('all-mpnet-base-v2')


    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join('results', 
                               f"{ke_method}/HAC_{sim_technique}_{int(sim_threshold)}", 
                               f'{timestamp}_{datasets_max_len}') # Create a folder like: results/RAKE/HAC_NEb_25/{timestamp}_{datasets_max_len}
    

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
            lines  = f.readlines() # Each line is a document of a specific dataset
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the document (doc, label, stemmed_label)

        output_list = [] # Keeps all available information for each document of the dataset
        perkeyphrase_no_tokens = []
        perdoc_no_keyphrases = []
        
        perdoc_times = []
        perdataset_start_time = time.perf_counter()

        for j_data in tqdm(data_list): # For JSON data in data_list (for each document)

            doc = j_data['doc'] # Take the document (size of {datasets_max_len} tokens) labeled as 'doc'

            perdoc_start_time = time.perf_counter()
            # =============================== KE Process ========================================

            if ke_method == 'RAKE':
                rake_extractor = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
                rake_extractor.extract_keywords_from_text(doc)
                keyphrases = sorted(set(rake_extractor.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True)
                keyphrases = [kw for _,kw in keyphrases]

            elif ke_method == 'PositionRank':
                positionrank_keyphrases = spacy_model(doc) # Alternative: positionrank(doc)
                keyphrases = [kw.text for kw in positionrank_keyphrases._.phrases[:]]
            
            elif ke_method == 'KPMiner':
                kpminer_extractor = pke.unsupervised.KPMiner()
                kpminer_extractor.load_document(input = doc, language = 'en', spacy_model=spacy_model)
                kpminer_extractor.candidate_selection(lasf = 5, cutoff = 200)
                df = pke.load_document_frequency_file(input_file = kpminer_weights_file)
                kpminer_extractor.candidate_weighting(df = df, alpha = 2.3, sigma = 3.0)
                keyphrases = [kw for kw,_ in kpminer_extractor.get_all_sorted()]

            else:
                mprank_extractor = pke.unsupervised.MultipartiteRank()
                mprank_extractor.load_document(input = doc, stoplist = stoplist, language = 'en', spacy_model=spacy_model)
                mprank_extractor.candidate_selection(pos = {'NOUN', 'PROPN', 'ADJ'})
                mprank_extractor.candidate_weighting(alpha = 1.1, threshold = 0.74, method = 'average')
                keyphrases = [kw for kw,_ in mprank_extractor.get_all_sorted()]
     
            # ===================================================================================

        
            # =============================== Clustering Process ========================================
            if sim_technique == 'Eb': 
                # Generate an embedding for each keyword/keyphrase 
                keyphrase_embeddings = embedding_model.encode(keyphrases, 
                                                              convert_to_numpy=True, 
                                                              show_progress_bar=True) 

                keyphrases = cluster_keywords_embeddings(
                    keyphrases,
                    keyphrase_embeddings,
                    sim_threshold
                )
            else:
                #keyphrases = cluster_keywords(keyphrases, 
                #                            [' '.join(stemmer.stem(token.lower()) for token in word_tokenize(kw)) for kw in keyphrases],
                #                            sim_threshold)

                keyphrases = cluster_keywords(keyphrases,
                                              lemmatize_keywords(keyphrases, spacy_model)
                                              sim_threshold)

            # ===========================================================================================

            perdoc_times.append(time.perf_counter() - perdoc_start_time)

            log = {} # log keeps all the necessary information of the current document
            log['doc'] = doc
            log['title'] = j_data['title']
            log['label'] = j_data['label']
            log['stemmed_label'] = j_data['stemmed_label']
            log['final_pred_keyphrase'] = [pred.strip() for pred in keyphrases] 
            output_list.append(log)


        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases([log['final_pred_keyphrase'] for log in output_list])

        for log in output_list:
            keyphrases = log['final_pred_keyphrase']
            perdoc_no_keyphrases.append(len(keyphrases))

            for kw in keyphrases:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])

        
        with open(os.path.join(result_path, f'{dataset_name}_result.json'), "w", encoding='utf-8') as f: # The results file is located in: results/RAKE/HAC_NEb_25/{timestamp}_{datasets_max_len}/{dataset_name}_result.json
            for json_data in output_list: # For each log in output_list
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')


        jsonl_lines = []
        for log in output_list:
            line = {}
            line['title'] = log['title']
            line['keyphrases'] = log['final_pred_keyphrase']
            line['label'] = log['label']
            line['stemmed_label'] = log['stemmed_label']
            jsonl_lines.append(line)


        with open(os.path.join(data_path, 
                               f'{dataset_name}_MAX{datasets_max_len}_{ke_method}_HAC_{sim_technique}_{int(sim_threshold)}.jsonl'), 
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
            #f.write(f'Percentage (%) of non-words multiplied by the maximum number of keyphrases: {(total_non_word_count / total_word_count) * max(perdoc_no_keyphrases)}\n')
            #f.write(f'Average number of words for the entire dataset multiplied by the maximum number of keyphrases: {perdataset_avg_no_words * max(perdoc_no_keyphrases)}\n')

        