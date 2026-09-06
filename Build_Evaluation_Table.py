"""
Build_Evaluation_Table.py

Scans a folder (recursively) for "evaluation.json" files -- one per KE
method, each holding F1 scores for every dataset -- and aggregates them
into TWO CSV tables:

    Table 1 (f1@5):
        Rows    -> KE methods
        Columns -> one block per Dataset (+ a final "Average" block),
                   each block with 3 sub-columns:
                       F1@5, Partial 0.25 F1@5, Partial 0.50 F1@5

    Table 2 (f1@10): identical structure, using the f1@10 variants.

Only "evaluation.json" files whose IMMEDIATE parent folder name contains
TARGET_MAX_LEN (as a substring) are included -- same rule as the stats
table script.

A single "Datasets Max Length" value (from "Datasets_Max_Len" in the JSON
files) is captured once and written above each table; any inconsistency
across files is printed as a warning.

------------------------------------------------------------------------
HOW TO CONFIGURE
------------------------------------------------------------------------
1. Set INPUT_FOLDER, OUTPUT_CSV_F1_AT_5, OUTPUT_CSV_F1_AT_10, and
   TARGET_MAX_LEN below.
2. Set DATASET_NAMES to the known list of dataset names (these become the
   dataset column blocks, in this order). An "Average" block is always
   appended automatically after them.
------------------------------------------------------------------------
"""

import os
import csv
import json


# =========================================================================
# CONFIGURATION
# =========================================================================

INPUT_FOLDER = "/path/to/your/results/folder"
OUTPUT_CSV_F1_AT_5 = "/path/to/your/output/f1_at_5_table.csv"
OUTPUT_CSV_F1_AT_10 = "/path/to/your/output/f1_at_10_table.csv"

# Only files whose immediate parent folder name contains this value (as a
# substring) will be picked up.
TARGET_MAX_LEN = "FULL"


# Controls row ordering (same in both tables). KE methods are grouped by
# their "Category" field (present in every evaluation.json) and groups are
# listed in this order. Order of KE methods *within* a group does not
# matter (alphabetical). Must contain every category value in your files.
GROUP_ORDER = ["A", "B", "C", "D", "E", "F"]


# Known dataset names, in the order you want them to appear as columns.
DATASET_NAMES = [
    "MDPI",
    "krapivin",
    "nus", 
    "SemEval2010", 
    "DUC2001", 
]

# Label used for the extra "average across datasets" column block.
AVERAGE_LABEL = "Average"

# The exact filename to search for (case-sensitive, exact match).
FILENAME = "evaluation.json"

# Sub-columns for each table: (display_name, path_as_list_of_keys)
# NOTE: paths are lists of literal keys (not dotted strings), because keys
# like "partial_0.25" already contain a dot.
METRICS_F1_AT_5 = [
    ("F1@5",                 ["f1@5"]),
    ("Partial 0.25 F1@5",    ["partial_0.25", "f1@5"]),
    ("Partial 0.50 F1@5",    ["partial_0.50", "f1@5"]),
]

METRICS_F1_AT_10 = [
    ("F1@10",                ["f1@10"]),
    ("Partial 0.25 F1@10",   ["partial_0.25", "f1@10"]),
    ("Partial 0.50 F1@10",   ["partial_0.50", "f1@10"]),
]


# =========================================================================
# IMPLEMENTATION
# =========================================================================

def find_evaluation_files(root_folder, target_max_len, filename):
    """Recursively find all files with the exact name `filename` whose
    immediate parent folder name contains `target_max_len` as a substring.
    Returns a de-duplicated, sorted list of absolute paths."""
    found = set()
    skipped_count = 0
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        parent_folder_name = os.path.basename(dirpath)
        for fname in filenames:
            if fname == filename: # If fname == "evaluations.json"
                if target_max_len not in parent_folder_name:
                    skipped_count += 1
                    continue
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                real_path = os.path.realpath(full_path)  # resolve symlinks
                found.add(real_path)
    if skipped_count:
        print(f"Skipped {skipped_count} '{filename}' file(s) whose parent "
              f"folder name did not contain '{target_max_len}'.")
              
    return sorted(found)


def get_nested(d, path_keys): # {d} is the record (dict) and {path_keys} is the metric
    """Fetch a value from a nested dict using a list of literal keys.
    Returns None if any part of the path is missing."""
    current = d
    for key in path_keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table(data, ke_methods, metrics, output_csv, datasets_max_len):
    """Build and write one CSV table (either the f1@5 or f1@10 version)."""
    columns = DATASET_NAMES + [AVERAGE_LABEL]
    k = len(metrics)

    header_row1 = ["KE Method"]
    header_row2 = [""]
    for column in columns:
        header_row1.append(column)
        header_row1.extend([""] * (k - 1))
        header_row2.extend([display_name for display_name, _ in metrics])

    data_rows = []
    for ke in ke_methods:
        record = data.get(ke, {})
        row = [ke]
        for column in columns:
            if column == AVERAGE_LABEL:
                source = record.get("average")
            else:
                source = record.get("datasets", {}).get(column)

            if source is None:
                row.extend([""] * k)
                continue

            for _display_name, path_keys in metrics:
                value = get_nested(source, path_keys)
                row.append("" if value is None else value)

        data_rows.append(row)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Datasets Max Length: {datasets_max_len}"])
        writer.writerow([])  # blank spacer row
        writer.writerow(header_row1)
        writer.writerow(header_row2)
        writer.writerows(data_rows)

    print(f"Wrote table with {len(ke_methods)} KE method(s) x "
          f"{len(columns)} column(s) x {k} sub-column(s) to:\n  {output_csv}")






def main():
    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a directory: {INPUT_FOLDER}")

    eval_files = find_evaluation_files(INPUT_FOLDER, TARGET_MAX_LEN, FILENAME)
    print(f"Found {len(eval_files)} '{FILENAME}' file(s) matching "
          f"TARGET_MAX_LEN='{TARGET_MAX_LEN}'.")


    print("The paths of the evaluation files are: \n")
    for path in eval_files:
        print(path)

    print()


    # data[ke_method] = full evaluation dict for that method
    data = {}
    datasets_max_len = None
    max_len_warnings = []

    for path in eval_files:
        try:
            record = load_json_file(path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP] Could not read/parse {path}: {e}")
            continue

        ke = record.get("KE")
        if ke is None:
            print(f"  [SKIP] Missing 'KE' field in {path}")
            continue

        dml = record.get("Datasets_Max_Len")
        if dml is not None:
            if datasets_max_len is None:
                datasets_max_len = dml
            elif dml != datasets_max_len:
                max_len_warnings.append((path, dml))

        if ke in data:
            print(f"  [WARNING] Duplicate KE method '{ke}' found again in {path}; "
                  f"overwriting previous entry for this KE method.")
        data[ke] = record




    if max_len_warnings:
        print("\n[WARNING] Inconsistent 'Datasets_Max_Len' values found "
              f"(keeping the first one seen: {datasets_max_len}):")
        for path, dml in max_len_warnings:
            print(f"    {path} -> {dml}")


    missing_category = sorted(ke for ke, record in data.items() if record.get("Category") is None)
    if missing_category:
        raise SystemExit(
            "The following KE method(s) are missing a 'Category' field in their "
            f"evaluation.json file(s): {missing_category}. Every evaluation file must "
            "include a 'Category' key that appears in GROUP_ORDER."
        )
 
    unknown_categories = sorted({record["Category"] for record in data.values()
                                  if record["Category"] not in GROUP_ORDER})
    if unknown_categories:
        raise SystemExit(
            f"The following Category value(s) are not listed in GROUP_ORDER: "
            f"{unknown_categories}. Add them to GROUP_ORDER (in your desired group order)."
        )
 



    ke_methods = sorted(
        data.keys(),
        key=lambda ke: (GROUP_ORDER.index(data[ke]["Category"]), ke)
    )

    
    #ke_methods = sorted(data.keys())

    print(f"The Keyword Extraction (KE) methods are: ")
    for ke in ke_methods:
        print()
    print()

    if not ke_methods:
        print("No valid records found. Nothing to write.")
        return

    print()
    build_table(data, ke_methods, METRICS_F1_AT_5, OUTPUT_CSV_F1_AT_5, datasets_max_len)
    build_table(data, ke_methods, METRICS_F1_AT_10, OUTPUT_CSV_F1_AT_10, datasets_max_len)
    print(f"\nDatasets Max Length: {datasets_max_len}")


if __name__ == "__main__":
    main()