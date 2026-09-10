import os
import sys
import json
import datetime
import argparse
import torch
import time
import re
import math
import numpy as np

from pathlib import Path
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from Utilities import process_keyphrases


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.set_verbosity_error()


def get_generated_output(str):
    split_prompt_output = str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
    return split_prompt_output[-1].strip().replace("<|eot_id|>", "")


def next_power_of_two(x):
    return 1 << x.bit_length()


def get_data_files(data_path, T):

    patterns = [
        re.compile(r"^(.+)_MAX([A-Z0-9]+)_(.+)\.jsonl$"),
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_YAKE_{T}\.jsonl$"),
        re.compile(rf"^(.+)_MAX([A-Z0-9]+)_TopicRank\.jsonl$")
    ]

    data_files = [
        file.name.split('/')[-1]
        for file in Path(data_path).iterdir()
        if (file.is_file() and (
                                (patterns[0].match(file.name) and file.name not in ["YAKE", "TopicRank"])
                                or patterns[1].match(file.name)
                                or patterns[2].match(file.name)
                               )
        )
    ]

    return data_files


# =========================================================================================
# PHASE 1a: THE VRAM-SAFE EQUATION
# =========================================================================================

def get_bytes_per_token(model):
    """
    KV cache cost per token per sequence, in bytes.
    bytes_per_token = 2 (K and V) x num_layers x num_kv_heads x head_dim x bytes_per_element
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    if model.dtype in (torch.float16, torch.bfloat16):
        bytes_per_element = 2
    else:
        bytes_per_element = 4

    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    return bytes_per_token


def get_vram_budget_mb(device, safety_fraction=0.30, fixed_overhead_mb=2500):
    """
    Computes how much VRAM (in MB) is safely available for the KV cache.

    Practical/safe design choices:
      - Uses MEASURED memory actually allocated by the model after loading
        (torch.cuda.memory_allocated), instead of a theoretical parameter-count
        estimate. Measured usage already reflects real weight storage plus
        any buffers PyTorch/transformers allocated, so it is closer to ground
        truth than a back-of-envelope calculation.
      - fixed_overhead_mb reserves room for CUDA context, cuBLAS/cuDNN
        workspace, tokenizer buffers, and other fixed costs that don't scale
        with batch size.
      - safety_fraction then discounts what's left further, to absorb memory
        fragmentation, PyTorch's caching allocator behavior, and the
        activation-memory spike that occurs during prefill (briefly higher
        than steady-state decoding memory). Default 0.30 (keep 70% of the
        remaining headroom, reserve 30% as margin) is intentionally
        conservative -- the goal is to never touch the OOM edge, even if it
        costs some batch size efficiency.

    Returns v_available_mb: the MB budget usable for KV cache.
    """
    torch.cuda.synchronize(device)

    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    measured_weights_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    remaining_mb = total_vram_mb - measured_weights_mb - fixed_overhead_mb

    if remaining_mb <= 0:
        raise RuntimeError(
            f"No VRAM remaining after accounting for model weights "
            f"({measured_weights_mb:.0f} MB) and fixed overhead "
            f"({fixed_overhead_mb} MB) out of {total_vram_mb:.0f} MB total. "
            f"Cannot safely batch."
        )

    v_available_mb = remaining_mb * (1 - safety_fraction)

    print(f"\n[VRAM BUDGET]")
    print(f"  Total VRAM:              {total_vram_mb:.0f} MB")
    print(f"  Measured model weights:  {measured_weights_mb:.0f} MB")
    print(f"  Fixed overhead reserved: {fixed_overhead_mb} MB")
    print(f"  Remaining before margin: {remaining_mb:.0f} MB")
    print(f"  Safety fraction held back: {safety_fraction * 100:.0f}%")
    print(f"  --> Available for KV cache: {v_available_mb:.0f} MB\n")

    return v_available_mb


def round_down_batch_size(n, multiple=8):
    """
    Rounds a batch size limit down to a multiple of `multiple` (GPU-friendly),
    but never below 1.
    """
    if n < multiple:
        return max(1, int(n))
    return int((n // multiple) * multiple)


def batch_limit_for_seq_len(seq_len, bytes_per_token, v_available_mb):
    """
    Given a sequence length, how many sequences of that length can safely
    fit together in one batch, according to the VRAM budget.
    """
    v_available_bytes = v_available_mb * (1024 ** 2)
    raw_limit = v_available_bytes // (bytes_per_token * seq_len)
    return round_down_batch_size(max(1, int(raw_limit)))


# =========================================================================================
# PHASE 1b: DATASET LOADING WITH REAL TOKENIZED LENGTHS
# =========================================================================================

def load_documents_with_lengths(data_path, data_file, tokenizer, task_instruction, prompt_template, T):
    """
    Loads a dataset file and computes the REAL tokenized prompt length for
    each document (instruction + special tokens + TITLE/KEYWORDS content),
    instead of a word-count approximation.

    Returns a list of dicts: {"index": i, "j_data": <original json>, "prompt": <str>, "length": <int>}
    """
    with open(os.path.join(data_path, data_file), "r", encoding='utf-8') as f:
        lines = f.readlines()
        data_list = [json.loads(line.strip()) for line in lines]

    documents = []
    for i, j_data in enumerate(data_list):
        doc = f"TITLE: {j_data['title']}. KEYWORDS: {'; '.join(j_data['keyphrases'][:T])}"
        prompt = prompt_template.format(task_instruction, doc)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        documents.append({
            "index": i,
            "j_data": j_data,
            "doc": doc,
            "prompt": prompt,
            "length": prompt_len
        })

    return documents


# =========================================================================================
# PHASE 1c: BIN-PACKING BATCH CONSTRUCTION (implementation 2)
# =========================================================================================

def create_batches(documents, bytes_per_token, v_available_mb, max_new_tokens):
    """
    Sorts documents by prompt token length ascending, then greedily packs
    them into batches such that adding the next document never exceeds the
    VRAM-safe batch size limit for the resulting max sequence length in that
    batch. No arbitrary "short/long" threshold is used -- the cutoff between
    batches falls directly out of the VRAM equation.

    Note: seq_len used for the KV-cache budget must include the tokens that
    will be GENERATED too (the KV cache grows as generation proceeds), so we
    add max_new_tokens to each document's prompt length before applying the
    equation.

    Returns a list of batch dicts:
      {"documents": [...], "max_len": int, "batch_size": int}
    """
    sorted_docs = sorted(documents, key=lambda d: d["length"])

    batches = []
    current_batch = []
    current_max_len = 0

    for doc in sorted_docs:
        effective_len = doc["length"] + max_new_tokens
        candidate_max_len = max(current_max_len, effective_len)
        candidate_count = len(current_batch) + 1

        limit = batch_limit_for_seq_len(candidate_max_len, bytes_per_token, v_available_mb)

        if candidate_count <= limit:
            current_batch.append(doc)
            current_max_len = candidate_max_len
        else:
            if current_batch:
                batches.append({
                    "documents": current_batch,
                    "max_len": current_max_len,
                    "batch_size": len(current_batch)
                })
            current_batch = [doc]
            current_max_len = effective_len

    if current_batch:
        batches.append({
            "documents": current_batch,
            "max_len": current_max_len,
            "batch_size": len(current_batch)
        })

    return batches


def print_batch_plan(batches, data_file):
    print(f"\n[BATCH PLAN] {data_file}")
    print(f"  Total documents: {sum(b['batch_size'] for b in batches)}")
    print(f"  Total batches:   {len(batches)}")
    for i, b in enumerate(batches):
        print(f"    Batch {i:>3}: size={b['batch_size']:>4}  max_len={b['max_len']:>5}")
    print()


if __name__ == '__main__':

    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nText: {}<|eot_id|>"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Llama3 path")
    parser.add_argument('--data_path', type=str, default='data/processed', help="Directory path of test datasets")
    parser.add_argument('--max_new_tokens', type=str, default='64', help="Maximum number of tokens to generate")
    parser.add_argument('--cuda', type=str, default='0', help="GPU")
    parser.add_argument('--auth_token', type=str, default='', help="Authentication token for Llama")
    parser.add_argument('--T', type=str, default='10', help="Number of keywords to extract")
    parser.add_argument('--safety_fraction', type=str, default='0.30', help="Fraction of remaining VRAM to hold back as margin")
    parser.add_argument('--fixed_overhead_mb', type=str, default='2500', help="Fixed VRAM overhead reserved (MB)")
    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    max_new_tokens = int(args.max_new_tokens)
    auth_token = args.auth_token
    T = int(args.T)
    safety_fraction = float(args.safety_fraction)
    fixed_overhead_mb = int(args.fixed_overhead_mb)

    data_files = get_data_files(data_path, T)

    print("The test datasets are:\n")
    for data_file in data_files:
        print(data_file)

    choice = input("\nContinue? (y/n): ").strip().lower()

    if choice in ("y", "yes"):
        pass
    elif choice in ("n", "no"):
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid input. Exiting...")
        sys.exit(1)

    task_instruction = (
        f"You are a keyphrase synthesis assistant. Given a document where the title appears after TITLE: "
        f"and the initial semicolon-separated list of keywords appears after KEYWORDS:, produce exactly {T} "
        f"concise, relevant, and informative keyphrases by refining, normalizing, combining, removing "
        f"irrelevant entries, and adding important concepts implied by the title or existing keywords, using "
        f"both as relevance anchors. The answer should be listed after 'Keyphrases: ' and separated by "
        f"semicolons (;). 'Keyphrases: keyphrase 1 ; keyphrase 2 ; ... ; keyphrase {T}'"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        output_attentions=False,
        token=auth_token,
        attn_implementation="sdpa"
    )

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # =====================================================================================
    # PHASE 1: BATCH PLANNING (equation + dataset loading + bin-packing)
    # This happens ONCE, after the model is loaded (so measured VRAM usage is accurate),
    # and BEFORE any generation takes place.
    # =====================================================================================

    bytes_per_token = get_bytes_per_token(model)
    print(f"[EQUATION] bytes_per_token (KV cache cost per token per sequence): {bytes_per_token} bytes")

    v_available_mb = get_vram_budget_mb(
        device,
        safety_fraction=safety_fraction,
        fixed_overhead_mb=fixed_overhead_mb
    )

    all_batch_plans = {}  # data_file -> list of batches

    for data_file in data_files:
        print(f"Data file: {data_file}")

        documents = load_documents_with_lengths(
            data_path, data_file, tokenizer, task_instruction, prompt_template, T
        )

        batches = create_batches(documents, bytes_per_token, v_available_mb, max_new_tokens)
        print_batch_plan(batches, data_file)

        all_batch_plans[data_file] = batches

    print("=" * 70)
    print("PHASE 1 complete: batch plans constructed for all data files.")
    print("No generation has been run yet. Review the batch plans above before")
    print("proceeding to Phase 2 (padded tokenization + model.generate() per batch).")
    print("=" * 70)