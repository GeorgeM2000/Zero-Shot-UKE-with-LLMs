import os
import json
import datetime
import argparse
import pytextrank # Required if you want to use PositionRank, TextRank, TopicRank
import pke # Required if you want to use KPMiner, MPRank, and other KE methods 
import spacy
import string
import time
import psutil
import numpy as np
import concurrent.futures

from tqdm import tqdm
from RAKE import Rake, Metric
from Utilities import process_keyphrases, cluster_keywords, cluster_keywords_embeddings, lemmatize_keywords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
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

    kneedle = KneeLocator(x, runtimes, curve="convex", direction="decreasing")
    knee = kneedle.knee

    if knee is None:
        return len(runtimes)
    else:
        return int(knee) + 1


# ---------------------------------------------------------------------------
# Module-level globals, all loaded ONCE in the parent process, BEFORE the
# ProcessPoolExecutor pool is created.
#
# On Linux, pool workers are created via fork(), inheriting a copy-on-write
# view of the parent's memory at the moment of forking. Anything loaded here
# -- the spaCy model, the (CPU) embedding model -- is therefore loaded exactly once,
# total, and shared (read-only) across every worker, rather than being
# reloaded per worker or re-pickled per task.
#
# This script is CPU-only throughout, so unlike a
# GPU-batched script, there's no fork/CUDA sequencing concern here -- a
# single pass is enough: load everything, then create the pool once per
# dataset.
# ---------------------------------------------------------------------------
SPACY_MODEL = None
EMBEDDING_MODEL = None



def load_spacy_model(spacy_model_path):
    """Loads the spaCy model once, in the parent, before the pool exists."""
    global SPACY_MODEL
    SPACY_MODEL = spacy.load(spacy_model_path)
    return SPACY_MODEL


def load_embedding_model():
    """Loads the SentenceTransformer model once, in the parent, on CPU."""
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    return EMBEDDING_MODEL



def create_parallel_settings(data, ke_method, dataset_name, no_docs, reserve_cores=1):

    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = os.cpu_count()

    cores = max(1, cores - reserve_cores)

    serial_runtime = data[ke_method][dataset_name]["Runtime"]["Per_Dataset"]
    runtimes = [serial_runtime / n for n in range(1, cores + 1)]

    no_cores = find_knee_with_kneedle(runtimes)

    batch_size = max(1, int(no_docs / no_cores))

    batch_ranges = []
    start = 0
    while start < no_docs:
        end = min(start + batch_size, no_docs)
        batch_ranges.append([start, end])
        start = end

    if len(batch_ranges) > no_cores and len(batch_ranges) > 1:
        batch_ranges[-2][1] = batch_ranges[-1][1]
        del batch_ranges[-1]

    print(f"Created {len(batch_ranges)} batch ranges for (Dataset = {dataset_name}, KE = {ke_method})")
    for batch_range in batch_ranges:
        print(batch_range)

    return no_cores, batch_ranges



def find_stats_files(root_folder, target_max_len, filename_suffix="_stats.json"):
    """Recursively find all files ending with FILENAME_SUFFIX whose immediate
    parent folder name contains `target_max_len` as a substring.
    Returns a de-duplicated, sorted list of absolute paths."""
    found = set()
    skipped_count = 0
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        parent_folder_name = os.path.basename(dirpath)
        for fname in filenames:
            if fname.endswith(filename_suffix):
                if target_max_len not in parent_folder_name:
                    skipped_count += 1
                    continue
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                real_path = os.path.realpath(full_path)
                found.add(real_path)

    if skipped_count:
        print(f"Skipped {skipped_count} '*{filename_suffix}' file(s) whose parent "
              f"folder name did not contain '{target_max_len}'.")

    return sorted(found)


def load_stats_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_batch_full_pipeline(batch_data_list, batch_index, ke_method, sim_technique, sim_threshold,
                                 kpminer_weights_file=None):
    """
    Runs the FULL per-document pipeline -- KE extraction, then per-document
    .encode() (if sim_technique == 'Eb'), then per-document HAC clustering
    -- for every document in this batch. This mirrors exactly what
    KE_with_Clustering.py's serial per-document loop does; the only thing
    that changes is that this runs inside one of several parallel worker
    processes instead of a single serial loop.

    All models/static resources (SPACY_MODEL, EMBEDDING_MODEL) are read from module-level globals, inherited via fork
    from the parent process -- never passed in as arguments, never
    reloaded here.
    """
    batch_keyphrases = []

    # Hoisted out of the per-document loop for RAKE, same fix as in
    # Parallel_KE.py: one Rake extractor reused across this batch's
    # documents instead of constructing a fresh one per document.
    rake_extractor = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO) if ke_method == 'RAKE' else None

    for j_data in batch_data_list:
        doc = j_data['doc']

        # =============================== KE Process ========================================
        if ke_method == 'RAKE':
            rake_extractor.extract_keywords_from_text(doc)
            keyphrases = sorted(set(rake_extractor.get_ranked_phrases_with_scores()), key=lambda x: x[0], reverse=True)
            keyphrases = [kw for _,kw in keyphrases]

        elif ke_method == 'PositionRank':
            positionrank_keyphrases = SPACY_MODEL(doc)
            keyphrases = [kw.text for kw in positionrank_keyphrases._.phrases[:]]

        elif ke_method == 'TextRank':
            textrank_keyphrases = SPACY_MODEL(doc)
            keyphrases = [kw.text for kw in textrank_keyphrases._.phrases[:]]

        # ===================================================================================

        # =============================== Clustering Process ========================================
        if sim_technique == 'Eb':
            # Per-document .encode() call, exactly as in the original serial
            # script -- not batched across documents.
            keyphrase_embeddings = EMBEDDING_MODEL.encode(
                keyphrases,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            keyphrases = cluster_keywords_embeddings(keyphrases, keyphrase_embeddings, sim_threshold)
        else:
            keyphrases = cluster_keywords(
                keyphrases,
                lemmatize_keywords(keyphrases, SPACY_MODEL),
                sim_threshold,
            )
        # ===========================================================================================

        batch_keyphrases.append([pred.strip() for pred in keyphrases])

    return batch_index, batch_keyphrases


def parallel_full_pipeline_processing(documents, batch_ranges, num_processes, ke_method, sim_technique, sim_threshold):
    keyphrases = [None] * len(documents)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                process_batch_full_pipeline,
                documents[start:end],
                batch_index,
                ke_method,
                sim_technique,
                sim_threshold
            )
            for batch_index, (start, end) in enumerate(batch_ranges)
        ]

        for future in concurrent.futures.as_completed(futures):
            batch_index, batch_keywords = future.result()
            start, end = batch_ranges[batch_index]
            keyphrases[start:end] = batch_keywords

    return keyphrases




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ke_method', type=str, default='RAKE', help="The keyword/keyphrase extraction method")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--similarity_technique', type=str, default='NEb', help="Similarity technique (Embedding-based or Non-Embedding-based)")
    parser.add_argument('--similarity_threshold', type=str, default='0.25', help="Similarity threshold")
    parser.add_argument('--datasets_max_len', type=str, default='FULL', help="Maximum length of test datasets")
    args = parser.parse_args()

    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a directory: {INPUT_FOLDER}")

    sim_threshold = float(args.similarity_threshold)
    sim_technique = args.similarity_technique
    ke_method = args.ke_method
    data_path = args.data_path
    datasets_max_len = args.datasets_max_len

    spacy_model_path = "../en_core_web_sm-3.8.0-py3-none-any/en_core_web_sm/en_core_web_sm-3.8.0"

    # Load everything ONCE here, in the parent, before any
    # ProcessPoolExecutor pool is created below. ---

    # spaCy model is always needed: KE extraction for PositionRank/TextRank/
    # KPMiner/MPRank uses it directly, and lemmatize_keywords() needs it for
    # NEb clustering regardless of ke_method (matches the original script,
    # which loads it unconditionally too).
    load_spacy_model(spacy_model_path)

    if ke_method == 'PositionRank':
        SPACY_MODEL.add_pipe("positionrank")
    elif ke_method == 'TextRank':
        SPACY_MODEL.add_pipe("textrank")

    if sim_technique == 'Eb':  # Embedding-based (Eb) / Non-Embedding-based (NEb)
        load_embedding_model()  # CPU only -- no GPU/fork sequencing concern here

    stats_files = find_stats_files(INPUT_FOLDER, datasets_max_len)
    print(f"Found {len(stats_files)} '*_stats.json' file(s) matching "
          f"TARGET_MAX_LEN='{datasets_max_len}'.")

    # data[KE][dataset] = raw json dict, loaded from prior *_stats.json runs.
    # The serial (non-parallel) KE_with_Clustering.py run for this exact
    # (ke_method, sim_technique, sim_threshold) combination is the baseline
    # runtime used for the Kneedle core-count estimate below.
    data = {}
    ke_categories = {}
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

        dml = record.get("Datasets_Max_Length")
        if dml is not None:
            if datasets_max_length is None:
                datasets_max_length = dml
            elif dml != datasets_max_length:
                max_length_warnings.append((path, dml))

        category = record.get("Category")
        if category is not None:
            if ke not in ke_categories:
                ke_categories[ke] = category
            elif ke_categories[ke] != category:
                category_warnings.append((path, ke, ke_categories[ke], category))

        data.setdefault(ke, {})[dataset] = record

    # This must match the "KE" field the SERIAL script (KE_with_Clustering.py)
    # writes into its own stats.json for this exact combination -- that's
    # the baseline serial runtime the Kneedle core-count estimate is based on.
    serial_ke_key = f"{ke_method}_HAC_{sim_technique}_{int(sim_threshold * 100)}"

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    result_path = os.path.join(
        'results',
        f"Parallel_{ke_method}/HAC_{sim_technique}_{int(sim_threshold * 100)}",
        f'{timestamp}_{datasets_max_len}'
    )  # e.g. results/Parallel_RAKE/HAC_NEb_25/{timestamp}_{datasets_max_len}

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

        with open(os.path.join(data_path, f'{dataset_name}_MAX{datasets_max_len}.jsonl'), "r", encoding='utf-8') as f:
            lines = f.readlines()
            data_list = [json.loads(line.strip()) for line in lines]

        no_cores, batch_ranges = create_parallel_settings(data, serial_ke_key, dataset_name, len(data_list))


        print(f"Created {len(batch_ranges)} batch ranges for (Dataset = {dataset_name}, KE Method = {ke_method})")
        for r in batch_ranges:
            print(r)
        print()


        perdataset_start_time = time.perf_counter()

        output_list, perdoc_times = parallel_full_pipeline_processing(
            data_list, batch_ranges, no_cores, ke_method, sim_technique, sim_threshold
        )

        perdataset_end_time = time.perf_counter()

        total_word_count, total_non_word_count, perdataset_avg_no_words = process_keyphrases(
            [log['final_pred_keyphrase'] for log in output_list]
        )

        perkeyphrase_no_tokens = []
        perdoc_no_keyphrases = []
        for log in output_list:
            keyphrases = log['final_pred_keyphrase']
            perdoc_no_keyphrases.append(len(keyphrases))

            for kw in keyphrases:
                perkeyphrase_no_tokens.append(len(kw.split()))

        stats = {
            "KE": f"Parallel_{ke_method}_HAC_{sim_technique}_{int(sim_threshold * 100)}",
            "Dataset": dataset_name,
            "Timestamp": timestamp,
            "Datasets_Max_Length": datasets_max_len,
            "Category": "C",

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