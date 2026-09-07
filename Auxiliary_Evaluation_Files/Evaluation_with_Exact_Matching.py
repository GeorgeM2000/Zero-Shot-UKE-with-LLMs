import os
import re
import argparse
import nltk
import json
import logging



"""
The evaluation creates one file that contains the evaluations of all datasets for only one KE (Traditional UKE or LLM) method

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

def print_PRF(P, R, F1, N):
    logging.info("\nN=" + str(N))
    logging.info("P=" + str(P))
    logging.info("R=" + str(R))
    logging.info("F1=" + str(F1) + "\n")
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Directory path of pred files") # 
    args = parser.parse_args()

    preds_dir_path = args.path # Has to be something like this: results/Meta-Llama-3-8B-Instruct/{timestamp}/ --> This is the results folder created from the KE process
    # The file that contains the predictions is {dataset_name}_result.json for each dataset
    
    
    log_file_path  = os.path.join(preds_dir_path, 'experiment_results') # The file path to write the results

    dataset_list = ['Inspec', 
                    'SemEval2017', 
                    'SemEval2010', 
                    'DUC2001', 
                    'nus', 
                    'krapivin'
                    ] # 'Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin'

    # Sets up the logging system
    logging.basicConfig(level=logging.INFO, # Logs messages of level INFO and above
                        format='%(message)s' # Only print the message (no timestamps, etc.)
                        )
    
    logger = logging.getLogger() # Retrieves the root logger (central logging object)
    
    # Adds a handler that writes logs to a file:
    logger.addHandler(logging.FileHandler(log_file_path, 'w'))

    porter = nltk.PorterStemmer()

    files = os.listdir(preds_dir_path)    

    f_5_scores  = []
    f_10_scores = []
    f_15_scores = []

    for dataset_name in dataset_list:

        logging.info(f"Dataset Name: {dataset_name}")
        pred_file_path = os.path.join(preds_dir_path, f"{dataset_name}_result.json") # Each dataset will have a {dataset}_result.json file

        with open(pred_file_path, 'r', encoding='utf-8') as f:
            lines  = f.readlines() # Each line contains info about the document
            json_list = [json.loads(line.strip()) for line in lines] # json_list contains a dictionary for each document. The dictionary contains the document's info

        preds = [j_data['final_pred_keyphrase'] for j_data in json_list] # preds is the extracted keyphrases
        labels = [j_data['label'] for j_data in json_list] # labels is the true keywords


        #_, labels, _ = data.get_processed_data(data_name) 

        if len(preds) != len(labels):
            raise ValueError("The lengths of the preds and labels are not equal.")
        

        num_c_5 = num_c_10 = num_c_15 = 0
        num_e_5 = num_e_10 = num_e_15 = 0
        num_s = 0

        for pred_list, label_list in zip(preds, labels): 
            
            pred_list = [ p.replace('-'," ") for p in pred_list ] # Replace "-" with " "
            pred_list = [ p.replace('\n',"") for p in pred_list ] # Remove the newline characters
            pred_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in pred_list ] # Remove content inside parentheses/braces
            pred_list = [ " ".join(pred.split()) for pred in pred_list ] # Normalize whitespace
            pred_list = [ p.lower().strip() for p in pred_list ] # Convert all keyphrase characters into lowercase

            label_list = [ l.replace('-'," ") for l in label_list ]
            label_list = [ l.replace('\n',"") for l in label_list ]
            label_list = [ re.sub(r'\(.*?\)|\{.*?\}', '', kw).strip() for kw in label_list ]
            label_list = [ " ".join(l.split()) for l in label_list ]
            label_list = [ l.lower().strip() for l in label_list ]


            # Remove duplicates. Consider all unique keyphrases
            pred_set = []
            for pred in pred_list:
                if pred in pred_set or pred =='':
                    continue
                else:
                    pred_set.append(pred)


            # Change 15 to 10 
            pred_set_list = pred_set[:15] # Because the maximum number of keyphrases to evaluate is 15


            # Apply stemming to the predicted/extracted keywords and the true keywords
            pred_s_list = []
            for p in pred_set_list:
                tokens = p.split()
                pred_s_list.append(' '.join(porter.stem(t) for t in tokens))

            label_s_list = []
            for l in label_list:
                tokens = l.split()
                label_s_list.append(' '.join(porter.stem(t) for t in tokens))




            # Count the number of True Positives (TP) for 5, 10, and 15 keywords
            # EXACT MATCHING
            # ==================================================================
            j = 0
            for pred, pred_s in zip(pred_set_list, pred_s_list):
                if pred_s in label_s_list or pred in label_list:
                    if (j < 5):
                        num_c_5 += 1
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 10 and j >= 5):
                        num_c_10 += 1
                        num_c_15 += 1

                    elif (j < 15 and j >= 10):
                        num_c_15 += 1
                j += 1
            # ==================================================================





            # Count the number of the extracted keywords for 5, 10, and 15 keywords
            if (len(pred_list[0:5]) == 5):
                num_e_5 += 5
            else:
                num_e_5 += len(pred_list[0:5])

            if (len(pred_list[0:10]) == 10):
                num_e_10 += 10
            else:
                num_e_10 += len(pred_list[0:10])

            if (len(pred_list[0:15]) == 15):
                num_e_15 += 15
            else:
                num_e_15 += len(pred_list[0:15])

            num_s += len(label_list)
        
        # Calculate the Precision, Recall, and F1 scores for 5, 10, and 15 keywords and print them
        p_5, r_5, f_5 = get_PRF(num_c_5, num_e_5, num_s)
        print_PRF(p_5, r_5, f_5, 5)
        p_10, r_10, f_10 = get_PRF(num_c_10, num_e_10, num_s)
        print_PRF(p_10, r_10, f_10, 10)
        p_15, r_15, f_15 = get_PRF(num_c_15, num_e_15, num_s)
        print_PRF(p_15, r_15, f_15, 15)

        f_5_scores.append(f_5*100)
        f_10_scores.append(f_10*100)
        f_15_scores.append(f_15*100)

        logging.info('---------------------')

    avg_f_5 = sum(f_5_scores) / len(f_5_scores)
    logging.info("F1@5 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_5_scores))
    logging.info("Average F1@5 Score: " + f"{avg_f_5:.2f}")

    avg_f_10 = sum(f_10_scores) / len(f_10_scores)
    logging.info("F1@10 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_10_scores))
    logging.info("Average F1@10 Score: " + f"{avg_f_10:.2f}")

    avg_f_15 = sum(f_15_scores) / len(f_15_scores)
    logging.info("F1@15 Scores by Dataset: " + "\t".join(f"{f:.2f}" for f in f_15_scores))
    logging.info("Average F1@15 Score: " + f"{avg_f_15:.2f}")
  
