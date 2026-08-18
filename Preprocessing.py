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



def clean_text(text="", database="Inspec"):

    # Special handling for the DUC2001 and SemEval2017 datasets.
    # In these datasets, some line breaks are preceded by a space or comma
    # (e.g., "word \n" or "word,\n"). These are converted into pure line breaks
    # to preserve paragraph boundaries.
    if database == "Duc2001" or database == "Semeval2017":
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}') # A whitespace character (\s) OR comma (,) followed by a newline (\n)
        while (True): # Keeps repeating until no such pattern remains
            if pattern2.search(text) is not None:
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                #start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:] 
                text = text_new
            else:
                break
    
    # Replace single line breaks occurring inside sentences with spaces.
    # This joins text that was artificially wrapped across multiple lines.
    # Example:
    #   "machine learning\nmethods"
    # becomes:
    #   "machine learning methods"
    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if pattern2.search(text) is not None:
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            #start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break


    # Collapse multiple consecutive whitespace characters into a single space.
    # This cleans up spacing artifacts introduced during previous replacements.
    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if pattern3.search(text) is not None:
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            #start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    # Remove markup-related brackets that may appear in the text.
    # Characters removed: < > [ ] { }
    pattern1 = re.compile(r'[<>[\]{}]') 
    text = pattern1.sub(' ', text) 
    
    text = text.replace("\t", " ")
    
    # Convert paragraph markers extracted from some document formats into actual line breaks.
    text = text.replace(' p ', '\n')
    text = text.replace(' /p \n', '\n')

    # Split the text into individual lines for further cleanup.
    lines = text.splitlines()
    
    # Remove empty lines and reconstruct the text.
    text_new = ""
    for line in lines:
        if line != '\n':
            text_new += line + '\n'

    return text_new



def get_MDPI_data(file_path="data/MDPI/MDPI_Articles.json"):
    data = {}
    titles = {}
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                title = jsonl['title'] 
                fulltxt = jsonl['fulltext']
                
                doc = title + ". " + fulltxt
                doc = doc.replace('\n', ' ')

                data[jsonl['name']] = doc
                titles[jsonl['name']] = title
                labels[jsonl['name']] = keywords

            except:
                raise ValueError

    return data,labels,titles



def get_long_data(file_path="data/nus/nus_test.json"):
    data = {}
    titles = {}
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                title = jsonl['title'] 
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                
                #doc = ' '.join([title, abstract, fulltxt])
                doc = title + ". " + abstract + ". " + fulltxt
                doc = re.sub(r'\. ', ' . ', doc)
                doc = re.sub(', ' , ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')

                data[jsonl['name']] = doc
                titles[jsonl['name']] = title
                labels[jsonl['name']] = keywords

            except:
                raise ValueError
            
    return data,labels,titles


def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    data = {}
    titles = {}
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                title = jsonl['title']
                abstract = jsonl['abstract']
                doc = title + ". " + abstract
                doc = re.sub(r'\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                
                data[i] = doc
                titles[i] = title
                labels[i] = keywords
            except:
                raise ValueError
            
    return data,labels,titles



def get_duc2001_data(file_path="data/DUC2001"):

    # ----------------------------------------------------------------------
    # Patterns
    # ----------------------------------------------------------------------

    # Add/remove supported title tags here
    TITLE_TAGS = ["HEAD", "HEADLINE", "HL"]

    title_pattern = re.compile(
        rf'<(?:{"|".join(TITLE_TAGS)})\b[^>]*>(.*?)</(?:{"|".join(TITLE_TAGS)})>',
        re.S | re.I
    )

    text_pattern = re.compile(
        r'<TEXT\b[^>]*>(.*?)</TEXT>',
        re.S | re.I
    )

    # Removes HTML/XML tags inside extracted title/text
    html_pattern = re.compile(r'<[^>]+>')

    data = {}
    titles = {}
    labels = {}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    total_files = 0

    files_with_title = 0
    files_without_title = 0

    files_with_text = 0
    files_without_text = 0

    valid_files = 0
    discarded_files = 0

    # ------------------------------------------------------------------
    # Read annotations first
    # ------------------------------------------------------------------

    all_labels = {}
    for dirname, _, filenames in os.walk(file_path):
        for fname in filenames:

            if fname == "annotations.txt":

                infile = os.path.join(dirname, fname)

                with open(infile, "rb") as f:
                    text = f.read().decode("utf8")

                for line in text.splitlines():
                    left, right = line.split("@", 1)

                    left = left.strip()
                    keywords = right.strip().split(";")[:-1]

                    all_labels[left] = keywords

                break

    # -------------------------------------------------------
    # Rest news articles
    # -------------------------------------------------------

    for dirname, _, filenames in os.walk(file_path):
        for fname in filenames:

            if fname == "annotations.txt":
                continue

            infile = os.path.join(dirname, fname)
            
            total_files += 1
            with open(infile, "rb") as f:
                text = f.read().decode("utf8")

            # ---------------------- TITLE -----------------------------

            title_match = title_pattern.search(text)

            if title_match:
                files_with_title += 1
                title = html_pattern.sub("", title_match.group(1))
                title = re.sub(r"\s+", " ", title).strip()

            else:
                files_without_title += 1
                discarded_files += 1
                continue

            # ----------------------- TEXT -----------------------------

            text_match = text_pattern.search(text)

            if text_match:
                files_with_text += 1
                article = text_match.group(1)

            else:
                files_without_text += 1
                discarded_files += 1
                continue

            # -------------------- VALID ARTICLE -----------------------

            valid_files += 1

            article = article.lower()
            title = title.lower()

            article = clean_text(title + ". " + article, database="Duc2001")

            data[fname] = article.strip("\n")
            titles[fname] = title
            

            # Keep labels only for retained articles
            if fname in all_labels:
                labels[fname] = all_labels[fname]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    print("\nDUC2001 statistics")
    print("------------------------------")
    print(f"Total article files      : {total_files}")
    print(f"Files with title         : {files_with_title}")
    print(f"Files without title      : {files_without_title}")
    print(f"Files with text          : {files_with_text}")
    print(f"Files without text       : {files_without_text}")
    print(f"Valid articles           : {valid_files}")
    print(f"Discarded articles       : {discarded_files}")
    print()

    return data,labels,titles



def get_inspec_data(file_path="data/Inspec"):

    data = {}
    labels = {}
    for dirname, _, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')

            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("%", '')
                text = clean_text(text)
                data[left] = text
                f.close()

            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("\n",' ')
                text = clean_text(text, database="Inspec")
                text = text.lower()
                label = text.split("; ")
                labels[left] = label
                f.close()

    return data,labels


def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data = {}
    labels = {}
    for dirname, _, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')

            infile = os.path.join(dirname, fname)
            #f = open(infile, 'rb')
            #text = f.read().decode('utf8')
            with open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text, database="Semeval2017")
            data[left] = text.lower()
            #f.close()

    for dirname, _, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls = text.splitlines()
            labels[left] = ls
            f.close()

    return data,labels



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help="Directory path of test datasets")
    parser.add_argument('--max_len', type=str, default='FULL', help="Maximum length of input document") # Alternatives: 1024 2048 4096
    parser.add_argument('--word_norm', type=str, default='Stem', help="Word normalization technique")
    args = parser.parse_args()

    data_path = args.data_path # data/
    MAX_LEN = args.max_len if args.max_len == "FULL" else int(args.max_len) # The maximum token length each document will have
    word_norm_technique = args.word_norm

    if word_norm_technique == "Lemma": 
        spacy_model_path = "../en_core_web_lg-3.8.0-py3-none-any/en_core_web_lg/en_core_web_lg-3.8.0"
        spacy_model = spacy.load(spacy_model_path)
    else:
        # With PorterStemmer, matching candidate keywords to reference keywords becomes easier (loose matching)
        porter = nltk.PorterStemmer()

    dataset_list = [#'Inspec', 
                    #'SemEval2017',
                    'MDPI', 
                    'SemEval2010', 
                    'DUC2001' 
                    'nus', 
                    'krapivin'
                    ]

    for dataset_name in dataset_list:

        dataset_dir = os.path.join(data_path, dataset_name) # data/{dataset_name}

        if dataset_name == "SemEval2017": 
            data, references = get_semeval2017_data(dataset_dir + "/docsutf8", dataset_dir + "/keys")

        elif dataset_name == "DUC2001":
            data, references, titles_dict = get_duc2001_data(dataset_dir)

        elif dataset_name == "nus" :
            data, references, titles_dict = get_long_data(dataset_dir + "/nus_test.json")

        elif dataset_name == "krapivin":
            data, references, titles_dict = get_long_data(dataset_dir + "/krapivin_test.json")

        elif dataset_name == "kp20k":
            data, references, titles_dict = get_short_data(dataset_dir + "/kp20k_valid200_test.json")

        elif dataset_name == "SemEval2010":
            data, references, titles_dict = get_short_data(dataset_dir + "/semeval_test.json")
            
        elif dataset_name == "Inspec":
            data, references = get_inspec_data(dataset_dir)

        elif dataset_name == "MDPI":
            data, references, titles_dict = get_MDPI_data()


        docs = []
        labels = []
        titles = []
        labels_normalized = []

        doc_lens = [] # Document lengths for dataset {dataset_name}

        # Number of {dataset_name} documents {MAX_LEN} has exceeded. 1 (True) = document length is greater than {MAX_LEN}. 0 (False) = {MAX_LEN} is greater than document length
        exceeded_max_doc_len = [] if MAX_LEN != "FULL" else None 

        for key, doc in data.items(): # For each document in data

            # Get normalized labels and document segments

            # {labels} are the true keywords
            # {ref} represents the keyword or the keyphrase
            # {references[key]} returns a list of keywords/keyphrases for a single document given the key
            labels.append([ref.replace(" \n", "") for ref in references[key]]) 

            if word_norm_technique == "Lemma":
                labels_n = lemmatize_keywords(references[key], spacy_model)

            else:
                labels_n = [] # {labels_n} are the true normalized keywords
                for l in references[key]: # {l} represents the keyword or the keyphrase
                    tokens = l.split()
                    labels_n.append(' '.join(porter.stem(t) for t in tokens))
            
            doc_lens.append(len(doc.split()))
            exceeded_max_doc_len.append( (len(doc.split()) > MAX_LEN) if MAX_LEN != "FULL" else False)

            # Split the document and take the first 512 tokens (words, numbers, characters, and pretty much everything that is separated with a whitespace)
            if MAX_LEN != "FULL":
                doc = ' '.join(doc.split()[:MAX_LEN]) 
            
            titles.append(titles_dict[key])
            labels_normalized.append(labels_n)
            docs.append(doc)
        
        assert len(docs) == len(labels) == len(labels_normalized) == len(titles), "The lengths of doc_list, labels, labels_normalized and titles are not equal."

        print(f"\nThe maximum document length for dataset {dataset_name} is {max(doc_lens)}")
        exceeded_max_doc_len = np.array(exceeded_max_doc_len)

        print(f"Document length is greater than {MAX_LEN}:", np.count_nonzero(exceeded_max_doc_len == True))
        print(f"{MAX_LEN} is greater than document length:", np.count_nonzero(exceeded_max_doc_len == False))
        print()

        jsonl_lines = []
        for doc, label, normalized_label, title in zip(docs, labels, labels_normalized, titles):
            line = {}
            line['title'] = title
            line['label'] = label
            line['normalized_label'] = normalized_label
            line['doc'] = doc
            jsonl_lines.append(line)

        result_path = os.path.join(data_path, 'processed') # data/processed
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"Directory created: {result_path}")

        file_path = os.path.join(result_path, f'{dataset_name}_MAX{MAX_LEN}.jsonl') # data/processed/{dataset_name}_MAX{MAX_LEN}.jsonl

        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in jsonl_lines:
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
