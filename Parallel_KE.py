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
import psutil
import numpy as np
import concurrent.futures

from tqdm import tqdm
from RAKE import Rake, Metric
from Utilities import process_keyphrases
from kneed import KneeLocator


INPUT_FOLDER = "/path/to/your/results/folder"



def find_knee_with_kneedle(runtimes):
    """
    Find the knee (elbow) in a list of sorted runtimes (descending) using Kneedle.
 
    Args:
        runtimes (list or np.array): Runtimes for 1, 2, 3, ... cores (decreasing).
 
    Returns:
        int: The optimal number of cores to use (1-based). Falls back to using
             all available cores (len(runtimes)) if no knee is detected.
    """
    runtimes = np.array(runtimes)
    x = np.arange(len(runtimes))
 
    # Kneedle works best if curve is convex and decreasing
    kneedle = KneeLocator(x, runtimes, curve="convex", direction="decreasing")
    knee = kneedle.knee
 
    if knee is None:
        # If no knee detected, fall back to using all available cores
        return len(runtimes)
    else:
        # `knee` is a 0-based index into `runtimes`; runtimes[i] corresponds
        # to (i + 1) cores, so we convert index -> core count.
        return int(knee) + 1


# ---------------------------------------------------------------------------
# Module-level model global for spaCy/rank-based KE methods.
#
# On Linux, ProcessPoolExecutor workers are created via fork() by default.
# A forked child inherits a copy-on-write view of the PARENT's memory at the
# moment of forking. So if a large model is loaded ONCE, here, at module
# level, BEFORE the pool is created (i.e. before any fork happens), every
# worker process inherits a reference to that same already-loaded model and
# can read its weights from the same physical RAM pages -- no reloading,
# no duplication, no per-task pickling.
#
# This is different from (and cheaper than) loading the model separately
# inside each worker (e.g. via ProcessPoolExecutor's `initializer=`): that
# approach calls spacy.load() once PER WORKER, which means N workers -> N
# full copies of the model in RAM. Pattern B loads it exactly once, total,
# as long as workers only READ the model (pure inference -- which is the
# case here: PositionRank/TextRank/TopicRank only score text, they never
# update model weights).
#
# This also means: NEVER pass this model object as an argument to
# executor.submit() -- doing so forces Python to pickle and retransmit a
# full copy through a pipe to the worker, which defeats the whole point.
# Workers must reference SPACY_MODEL as a global instead.
#
# Caveat: this sharing relies on fork() (Linux default). It would not hold
# on spawn-based platforms (Windows/macOS spawn), and it does not apply to
# GPU-resident models (CUDA contexts don't survive fork).
# ---------------------------------------------------------------------------
SPACY_MODEL = None
 


def load_spacy_model(spacy_model_path, pipe_component):
    """
    Loads the spaCy model ONCE, in the parent process, before the worker
    pool is created. Must be called from the main process prior to
    instantiating ProcessPoolExecutor so that forked workers inherit the
    already-loaded model via copy-on-write.
    """
    global SPACY_MODEL
    model = spacy.load(spacy_model_path)
    model.add_pipe(pipe_component)
    SPACY_MODEL = model
    return SPACY_MODEL

def create_parallel_settings(data, ke_method, dataset_name, no_docs, reserve_cores=1):
 
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = os.cpu_count()
 
    cores = max(1, cores - reserve_cores)
 
    serial_runtime = data[ke_method][dataset_name]["Runtime"]["Per_Dataset"]
    # Fixed: build [T1/1, T1/2, T1/3, ...] from the ORIGINAL serial runtime,
    # not by repeatedly dividing the previous (already-divided) value.
    runtimes = [serial_runtime / n for n in range(1, cores + 1)]
 
    # Fixed: find_knee_with_kneedle now returns a single core count directly.
    no_cores = find_knee_with_kneedle(runtimes)
 
    # Fixed: avoid batch_size == 0 (and the resulting infinite loop) when
    # no_docs < no_cores.
    batch_size = max(1, int(no_docs / no_cores))
 
    batch_ranges = []
    start = 0
    while start < no_docs:
        end = min(start + batch_size, no_docs)  # Ensure the last range includes all remaining abstracts
        batch_ranges.append([start, end])
        start = end
 
    # Fixed: only merge the last two batches if a genuine small leftover
    # batch exists (i.e. flooring in batch_size produced more ranges than
    # no_cores). Previously this merge ran unconditionally, which either
    # silently dropped a worker's-worth of a batch when no_docs divided
    # evenly by no_cores, or crashed with an IndexError when there was
    # only a single batch.
    if len(batch_ranges) > no_cores and len(batch_ranges) > 1:
        batch_ranges[-2][1] = batch_ranges[-1][1]
        del batch_ranges[-1]
 
    # Print ranges for verification
    print(f"Created {len(batch_ranges)} batch ranges for (Dataset = {dataset_name}, KE Method = {ke_method})")
    for batch_range in batch_ranges:  # Fixed: no longer shadows builtin `range`
        print(batch_range)
 
    return no_cores, batch_ranges



def find_stats_files(root_folder, target_max_len, filename_suffix="_stats.json"):
    """Recursively find all files ending with FILENAME_SUFFIX whose immediate
    parent folder name contains `target_max_len` as a substring.
    Returns a de-duplicated, sorted list of absolute paths."""
    found = set() # set() is used so that there are no duplicate stat files
    skipped_count = 0
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        parent_folder_name = os.path.basename(dirpath)
        for fname in filenames:
            if fname.endswith(filename_suffix):
                if target_max_len not in parent_folder_name:
                    skipped_count += 1
                    continue
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                real_path = os.path.realpath(full_path)  # resolve symlinks
                found.add(real_path)

    if skipped_count:
        print(f"Skipped {skipped_count} '*{filename_suffix}' file(s) whose parent "
              f"folder name did not contain '{target_max_len}'.")
        
    return sorted(found)


def load_stats_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def process_batch_rake(batch_documents, batch_index, ke_method=None):
    batch_keyphrases = []
    
    for doc in batch_documents:        
        rake_extractor = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
        rake_extractor.extract_keywords_from_text(doc)
        keyphrases = sorted(set(rake_extractor.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True) 
        batch_keyphrases.append([kw for _,kw in keyphrases])

    return batch_index, batch_keyphrases

def process_batch_yake(batch_documents, batch_index, ke_method):
    batch_keyphrases = []
    
    for doc in batch_documents:        
        yake_extractor = ke_method.extract_keywords(doc)
        batch_keyphrases.append([kw for kw,_ in yake_extractor])

    return batch_index, batch_keyphrases


def process_batch_rank(batch_documents, batch_index, ke_method=None):
    # Pattern B: the model is never passed in through submit(). Each forked
    # worker already has its own reference to SPACY_MODEL, inherited via
    # copy-on-write from the parent process, where it was loaded exactly
    # once (see load_spacy_model(), called in __main__ before the pool is
    # created). `ke_method` is accepted for call-signature parity with the
    # other process_batch_* functions but is intentionally unused.
    batch_keyphrases = []
 
    for doc in batch_documents:
        rank_extractor = SPACY_MODEL(doc)
        batch_keyphrases.append([kw.text for kw in rank_extractor._.phrases[:]])
 
    return batch_index, batch_keyphrases


def parallel_ke_processing(documents, batch_ranges, ke_method, num_processes, ke_method_obj=None):
    """
    ke_method_obj: used for RAKE/YAKE (small, cheap-to-pickle objects --
        passed straight through to each task as before). For spaCy
        rank-based methods (PositionRank/TextRank/TopicRank), ke_method_obj
        is not needed: those tasks read the model from the module-level
        SPACY_MODEL global instead (see process_batch_rank and
        load_spacy_model). SPACY_MODEL must already be loaded, in the
        parent process, before this function is called -- ProcessPoolExecutor
        forks workers from the current process state, so a global set
        beforehand is inherited by every worker via copy-on-write, without
        ever needing to be passed through submit().
    """
    keyphrases = [None] * len(documents)  # Initialize the final list
    ke_function = None
 
    if ke_method == "RAKE":
        ke_function = process_batch_rake
    elif ke_method == "YAKE":
        ke_function = process_batch_yake
    else:
        ke_function = process_batch_rank
 
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(ke_function, documents[start:end], batch_index, ke_method_obj)
            for batch_index, (start, end) in enumerate(batch_ranges)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            batch_index, batch_keywords = future.result()
            start, end = batch_ranges[batch_index]
            keyphrases[start:end] = batch_keywords  # Place results in the correct range
 
    return keyphrases




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ke_method', type=str, default='RAKE', help="The keyword/keyphrase extraction method") 
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets") 
    parser.add_argument('--T', type=str, default='10', help="Number of keywords/keyphrases to extract")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    args = parser.parse_args() # Parse the arguments

    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a directory: {INPUT_FOLDER}")

    ke_method = args.ke_method
    data_path = args.data_path
    T = int(args.T) 
    datasets_max_len = args.datasets_max_len
    spacy_model_path = "../en_core_web_sm-3.8.0-py3-none-any/en_core_web_sm/en_core_web_sm-3.8.0"

    stats_files = find_stats_files(INPUT_FOLDER, datasets_max_len)
    print(f"Found {len(stats_files)} '*_stats.json' file(s) matching "
          f"TARGET_MAX_LEN='{datasets_max_len}'.")

    # Pattern B: load the spaCy model ONCE here, in the parent process,
    # before ProcessPoolExecutor forks any workers (that happens later,
    # inside parallel_ke_processing(), once per dataset). Because it's
    # loaded before the fork, every worker inherits the same already-loaded
    # model via copy-on-write -- no per-worker reload, no per-task pickling.
    pipe_component_map = {
        'PositionRank': 'positionrank',
        'TextRank': 'textrank',
        'TopicRank': 'topicrank',
    }
 
    if ke_method in pipe_component_map:
        load_spacy_model(spacy_model_path, pipe_component_map[ke_method])

    if ke_method == 'YAKE':
        yake_extractor = yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=T, features=None)


    # data[ke_method][dataset] = raw json dict
    data = {}
    ke_categories = {}  # ke_method -> Category
    category_warnings = []
    datasets_max_length = None
    max_length_warnings = []

    for path in stats_files:
        try:
            record = load_stats_file(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP] Could not read/parse {path}: {e}")
            continue

        ke = record.get("KE")
        dataset = record.get("Dataset")
        if ke is None or dataset is None:
            print(f"  [SKIP] Missing 'KE' or 'Dataset' field in {path}")
            continue

        # Track Datasets_Max_Length (should be the same across all files)
        dml = record.get("Datasets_Max_Length")
        if dml is not None:
            if datasets_max_length is None:
                datasets_max_length = dml
            elif dml != datasets_max_length: # If we extract a Datasets_Max_Len value that is different from TARGET_MAX_LEN
                max_length_warnings.append((path, dml))


        # Track Category per KE method (should be the same across all of a
        # given KE method's dataset files)
        category = record.get("Category")
        if category is not None:
            if ke not in ke_categories:
                ke_categories[ke] = category
            elif ke_categories[ke] != category:
                category_warnings.append((path, ke, ke_categories[ke], category))

        data.setdefault(ke, {})[dataset] = record



    # Create a timestamp, e.g 2026-05-03_07-29-30
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join('results', 
                               f"Parallel_{ke_method}/Parallel_{ke_method}" if ke_method != 'YAKE' else f"Parallel_{ke_method}/Parallel_{ke_method}_T{T}", 
                               f'{timestamp}_{datasets_max_len}') 
    
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
            data_list = [json.loads(line.strip()) for line in lines] # data_list contains information about the documents (doc, label, stemmed_label)

        
        no_cores, batch_ranges = create_parallel_settings(data, ke_method, dataset_name, len(data_list))


        print(f"Created {len(batch_ranges)} batch ranges for (Dataset = {dataset_name}, KE Method = {ke_method})")
        for r in batch_ranges:
            print(r)
        print()

        perkeyphrase_no_tokens = [] # Number of tokens (words, numbers, symbols) each keyphrase has
        perdoc_no_keyphrases = [] # Number of keyphrases extracted by a KE method for some dataset

        perdataset_start_time = time.perf_counter()

        if ke_method == "RAKE":
            keyphrases_list = parallel_ke_processing([j_data['doc'] for j_data in data_list],
                                                      batch_ranges, ke_method, no_cores)
        elif ke_method == "YAKE":
            keyphrases_list = parallel_ke_processing([j_data['doc'] for j_data in data_list],
                                                      batch_ranges, ke_method, no_cores, ke_method_obj=yake_extractor)
        elif ke_method in pipe_component_map:
            # SPACY_MODEL was already loaded once, above, before any worker
            # pool existed -- workers inherit it via fork copy-on-write.
            keyphrases_list = parallel_ke_processing([j_data['doc'] for j_data in data_list],
                                                      batch_ranges, ke_method, no_cores)
            
        perdataset_end_time = time.perf_counter()
        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases(keyphrases_list)

        for keyphrases in keyphrases_list:
            perdoc_no_keyphrases.append(len(keyphrases))
            
            for kw in keyphrases:
                perkeyphrase_no_tokens.append(len(kw.split())) # Alternative: len([token for part in kw.split('-') for token in part.split()])


        stats = {
            "KE": f"Parallel_{ke_method}",
            "Dataset": dataset_name,
            "T": T,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len,
            "Category": "B",

            "Runtime": {
                "Per_Document": {
                    "Mean": -1.0,
                    "Median": -1.0,
                    "Min": -1.0,
                    "Max": -1.0
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


        