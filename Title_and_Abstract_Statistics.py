import re
import sys
import codecs
import json
import os
import nltk
import argparse
import spacy
import numpy as np

from tqdm import tqdm
from Utilities import lemmatize_keywords
from Preprocessing import clean_text

def get_MDPI_data(file_path="data/MDPI/UMDPI_Abstracts.json"):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading MDPI Doc ..."):
            try:
                jsonl = json.loads(line)
                title = jsonl['title'] 
                abstract = jsonl['abstract']
                
                doc = title + ". " + abstract
                doc = doc.replace('\n', ' ')

                data[jsonl['name']] = doc

            except:
                raise ValueError

    return data



def get_long_data(dataset_name, file_path="data/nus/nus_test.json"):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc=f"Loading {dataset_name} Doc ..."):
            try:
                jsonl = json.loads(line)
                title = jsonl['title'] 
                abstract = jsonl['abstract']
                
                doc = title + ". " + abstract
                doc = re.sub(r'\. ', ' . ', doc)
                doc = re.sub(', ' , ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')

                data[jsonl['name']] = doc
        
            except:
                raise ValueError
            
    return data



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help="Directory path of test datasets")
    args = parser.parse_args()

    data_path = args.data_path # data/

    dataset_list = [
                    'MDPI',  
                    'nus', 
                    'krapivin'
                    ]

    for dataset_name in dataset_list:

        dataset_dir = os.path.join(data_path, dataset_name) # data/{dataset_name}

        if dataset_name == "nus" :
            data = get_long_data(dataset_name, file_path=dataset_dir + "/nus_test.json")

        elif dataset_name == "krapivin":
            data = get_long_data(dataset_name, file_path=dataset_dir + "/krapivin_test.json")

        elif dataset_name == "MDPI":
            data = get_MDPI_data()


        doc_lens = [] # Document lengths for dataset {dataset_name}

        for key, doc in data.items(): # For each document in data
 
            doc_lens.append(len(doc.split()))

        # ------------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------------
        print(f"===== DATASET: {dataset_name} =====\n\n")
        print(f"Number of documents : {len(doc_lens)}")
        print(f"Minimum length      : {min(doc_lens)}")
        print(f"Maximum length      : {max(doc_lens)}")
        print(f"Mean length         : {statistics.mean(doc_lens):.2f}")
        print(f"Median length       : {statistics.median(doc_lens):.2f}")
        print(f"Std. deviation      : {statistics.stdev(doc_lens):.2f}")
        print(f"25th percentile     : {statistics.quantiles(doc_lens, n=4)[0]:.2f}")
        print(f"75th percentile     : {statistics.quantiles(doc_lens, n=4)[2]:.2f}")
        print(f"90th percentile     : {statistics.quantiles(doc_lens, n=10)[8]:.2f}")
        print(f"95th percentile     : {statistics.quantiles(doc_lens, n=20)[18]:.2f}")
        print(f"99th percentile     : {statistics.quantiles(doc_lens, n=100)[98]:.2f}")

        
