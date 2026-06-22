import os
import re
import argparse
import nltk
import json
import logging

from Utilities import count_word_overlap_matches


"""
The evaluation process generates a single file containing the evaluation results for all datasets associated with one KE method (either Traditional UKE or LLM-based).
"""


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e!=0 else 0.0
    R = float(num_c) / float(num_s) if num_s!=0 else 0.0
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


# P: Precision R: Recall F1: F1-score N: Number of documents
def print_PRF(P, R, F1, N):
    logging.info("\nN=" + str(N))
    logging.info("P=" + str(P))
    logging.info("R=" + str(R))
    logging.info("F1=" + str(F1) + "\n")
    return 0










if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Directory path of prediction files")
    parser.add_argument('--T', type=str, default='5', required=False, help="Number of keywords and keyphrases to evaluate")
    args = parser.parse_args()


    T = int(args.T)
    preds_dir_path = args.path # Has to be something like this: results/Meta-Llama-3-8B-Instruct/{timestamp}/ --> This is the results folder created from the KE process
    # The file that contains the predictions is {dataset_name}_result.json for each dataset
    
    log_file_path  = os.path.join(preds_dir_path, 'experiment_results') # The file path to write the results for each dataset

    dataset_list = [#'Inspec', 
                    #'SemEval2017', 
                    'MDPI',
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin'
                    ] 


    # Sets up the logging system
    logging.basicConfig(level=logging.INFO, # Logs messages of level INFO and above
                        format='%(message)s' # Only print the message (no timestamps, etc.)
                        )
    
    logger = logging.getLogger() # Retrieves the root logger (central logging object)
    
    # Adds a handler that writes logs to a file:
    logger.addHandler(logging.FileHandler(log_file_path, 'w'))

    porter = nltk.PorterStemmer()

    files = os.listdir(preds_dir_path)    

    # Exact match score lists
    f_t_scores  = []
    
    # Partial match score lists (threshold=0.25)
    f_pc25_t_scores  = []
    
    # Partial match score lists (threshold=0.50)
    f_pc50_t_scores  = []
    
    for dataset_name in dataset_list:

        logging.info(f"Dataset Name: {dataset_name}")
        pred_file_path = os.path.join(preds_dir_path, f"{dataset_name}_result.json") # Each dataset will have a {dataset}_result.json file

        with open(pred_file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines() # Each line contains info about the document
            json_list = [json.loads(line.strip()) for line in lines] # json_list contains a dictionary for each document

        preds  = [j_data['final_pred_keyphrase'] for j_data in json_list] # preds is the extracted keyphrases
        labels = [j_data['label'] for j_data in json_list]                # labels is the true keywords

        if len(preds) != len(labels):
            raise ValueError("The lengths of the preds and labels are not equal.")
        
        
        # Exact match counters
        num_c_t = 0
        num_e_t = 0
        num_s = 0

        # Partial match counters (threshold=0.25)
        num_pc25_t = 0

        # Partial match counters (threshold=0.50)
        num_pc50_t = 0



        for pred_list, label_list in zip(preds, labels): 
            
            pred_list = [ p.replace('-'," ") for p in pred_list ]
            pred_list = [ p.replace('\n',"") for p in pred_list ]
            pred_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in pred_list ]
            pred_list = [ " ".join(pred.split()) for pred in pred_list ]
            pred_list = [ p.lower().strip() for p in pred_list ]

            label_list = [ l.replace('-'," ") for l in label_list ]
            label_list = [ l.replace('\n',"") for l in label_list ]
            label_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in label_list ]
            label_list = [ " ".join(l.split()) for l in label_list ]
            label_list = [ l.lower().strip() for l in label_list ]

            # Remove duplicates. Consider all unique keyphrases/keywords
            pred_set = []
            for pred in pred_list:
                if pred in pred_set or pred == '':
                    continue
                else:
                    pred_set.append(pred)

            pred_set_list = pred_set[:T] # Because the maximum number of keyphrases to evaluate is T

            # Apply stemming to the predicted/extracted keywords and the true keywords
            pred_s_list = []
            for p in pred_set_list:
                tokens = p.split()
                pred_s_list.append(' '.join(porter.stem(t) for t in tokens))

            label_s_list = []
            for l in label_list:
                tokens = l.split()
                label_s_list.append(' '.join(porter.stem(t) for t in tokens))

            # Count the number of True Positives (TP) for T keywords
            # EXACT MATCHING
            # ==================================================================
            j = 0
            for pred, pred_s in zip(pred_set_list, pred_s_list):
                if pred_s in label_s_list or pred in label_list:
                    if (j < T):
                        num_c_t  += 1

                j += 1
            # ==================================================================

            # PARTIAL MATCHING (threshold=0.25)
            # ==================================================================
            num_pc25_t  += count_word_overlap_matches(pred_s_list[:T],  pred_set_list[:T],  label_s_list, label_list, threshold=0.25)
            # ==================================================================

            # PARTIAL MATCHING (threshold=0.50)
            # ==================================================================
            num_pc50_t  += count_word_overlap_matches(pred_s_list[:T],  pred_set_list[:T],  label_s_list, label_list, threshold=0.50)
            # ==================================================================

            # Count the number of the extracted keywords for 5, 10, and 15 keywords
            if (len(pred_list[0:T]) == T):
                num_e_t += T
            else:
                num_e_t += len(pred_list[0:T])

            num_s += len(label_list)
        


        # EXACT MATCH: Calculate and log PRF scores
        logging.info("-- Exact Match --")

        p_t, r_t, f_t = get_PRF(num_c_t, num_e_t, num_s)
        print_PRF(p_t, r_t, f_t, T)

        f_t_scores.append(f_t*100)

        # PARTIAL MATCH (threshold=0.25): Calculate and log PRF scores
        logging.info("-- Partial Match (threshold=0.25) --")

        p_pc25_t, r_pc25_t, f_pc25_t = get_PRF(num_pc25_t, num_e_t, num_s)
        print_PRF(p_pc25_t, r_pc25_t, f_pc25_t, T)

        f_pc25_t_scores.append(f_pc25_t*100)


        # PARTIAL MATCH (threshold=0.50): Calculate and log PRF scores
        logging.info("-- Partial Match (threshold=0.50) --")

        p_pc50_t, r_pc50_t, f_pc50_t = get_PRF(num_pc50_t, num_e_t, num_s)
        print_PRF(p_pc50_t, r_pc50_t, f_pc50_t, T)

        f_pc50_t_scores.append(f_pc50_t*100)

        logging.info('---------------------')


    # EXACT MATCH: Average scores
    avg_f_t = sum(f_t_scores) / len(f_t_scores)
    logging.info(f"Exact Match F1@{T} Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_t_scores))
    logging.info(f"Exact Match Average F1@{T} Score: " + f"{avg_f_t:.2f}")


    # PARTIAL MATCH (threshold=0.25): Average scores
    avg_f_pc25_t = sum(f_pc25_t_scores) / len(f_pc25_t_scores)
    logging.info(f"Partial Match (0.25) F1@{T} Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_pc25_t_scores))
    logging.info(f"Partial Match (0.25) Average F1@{T} Score: " + f"{avg_f_pc25_t:.2f}")

    
    # PARTIAL MATCH (threshold=0.50): Average scores
    avg_f_pc50_t = sum(f_pc50_t_scores) / len(f_pc50_t_scores)
    logging.info(f"Partial Match (0.50) F1@{T} Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_pc50_t_scores))
    logging.info(f"Partial Match (0.50) Average F1@{T} Score: " + f"{avg_f_pc50_t:.2f}")

    
