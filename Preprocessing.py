import re
import codecs
import json
import os
import nltk
import argparse

from tqdm import tqdm



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
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
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
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
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
                doc = re.sub('\. ', ' . ', doc)
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
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                title = jsonl['title']
                abstract = jsonl['abstract']
                doc = title + ". " + abstract
                doc = re.sub('\. ', ' . ', doc)
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
    text_pattern  = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    title_pattern = re.compile(r'<HEAD>(.*?)</HEAD>', re.S)
    
    data = {}
    titles = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if fname == "annotations.txt": # There is only one "annotations.txt" file 
                #left, right = fname.split('.')
                infile = os.path.join(dirname, fname) # Since file_path = "data/DUC2001", dirname = file_path. infile = "data/DUC2001/{file}"
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1] # The assigned keywords are separated with a ";". The variable "d" holds all the assigned keywords of an article
                    l = left # The left part works as the ID of the article 
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                title = re.findall(title_pattern, text)[0] # Find all patterns of text where it starts with "<HEAD>" and end with "</HEAD>". If there are > 1, take the first occurence
                text = re.findall(text_pattern, text)[0]

                text = text.lower()
                title = title.lower()

                text = clean_text(title + ". " + text, database="Duc2001")
                data[fname] = text.strip("\n")
                titles[fname] = title
                #data[fname] = text

    return data,labels,titles



def get_inspec_data(file_path="data/Inspec"):

    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("%", '')
                text = clean_text(text)
                data[left] = text

            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("\n",' ')
                text = clean_text(text, database="Inspec")
                text = text.lower()
                label = text.split("; ")
                labels[left] = label

    return data,labels


def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            #f = open(infile, 'rb')
            #text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text, database="Semeval2017")
            data[left] = text.lower()
            #f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
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
    parser.add_argument('--max_len', type=str, default='512', help="Maximum length of input document") # Alternatives: 1024 2048 4096
    args = parser.parse_args()

    data_path = args.data_path # data/
    MAX_LEN = int(args.max_len) # The maximum token length each document will have

    dataset_list = [#'Inspec', 
                    #'SemEval2017',
                    'MDPI', 
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin']

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
        labels_stemmed = []

        porter = nltk.PorterStemmer()

        for key, doc in data.items(): # for each document in data

            # Get stemmed labels and document segments

            # labels are the true keywords
            # ref represents the keyword or the keyphrase
            # references[key] returns a list of keywords/keyphrases for a single document given the key
            labels.append([ref.replace(" \n", "") for ref in references[key]]) 

            labels_s = [] # labels_s are the true stemmed keywords 
            for l in references[key]: # l represents the keyword or the keyphrase
                tokens = l.split()
                labels_s.append(' '.join(porter.stem(t) for t in tokens))

            doc = ' '.join(doc.split()[:MAX_LEN]) # Split the document and take the first 512 tokens (words, numbers, characters, and pretty much everything that is separated with a whitespace)
            
            titles.append(titles_dict[key])
            labels_stemmed.append(labels_s)
            docs.append(doc)
        
        assert len(docs) == len(labels) == len(labels_stemmed) == len(titles), "The lengths of doc_list, labels, labels_stemmed and titles are not equal."
        
        jsonl_lines = []
        for doc, label, stemmed_label, title in zip(docs, labels, labels_stemmed, titles):
            line = {}
            line['doc'] = doc
            line['label'] = label
            line['stemmed_label'] = stemmed_label
            line['title'] = title
            jsonl_lines.append(line)

        result_path = os.path.join(data_path, 'processed') # data/processed
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"Directory created: {result_path}")

        file_path = os.path.join(result_path, f'{dataset_name}_MAX{MAX_LEN}.jsonl') # data/processed/{dataset_name}_MAX{MAX_LEN}.jsonl

        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in jsonl_lines:
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
