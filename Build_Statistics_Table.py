"""
Build_Statistics_Table.py

Scans a folder (recursively) for all "*_stats.json" files produced by the
Keyword Extraction (KE) methods, and aggregates them into a
single CSV table:

    - Rows    -> KE methods
    - Columns -> one block per Dataset, each block containing K sub-columns
                 (the statistical metrics / metadata you configure below)

The table has two physical header rows:
    Row 1: Dataset name (written once, blank for the rest of its sub-columns)
    Row 2: Metric / metadata name

A single "Datasets Max Length" value (taken from Datasets_Max_Length in the
JSON files) is printed and written as a note above the table.

------------------------------------------------------------------------
HOW TO CONFIGURE
------------------------------------------------------------------------
1. Set INPUT_FOLDER and OUTPUT_CSV below.
2. Edit STAT_METRICS to choose which statistical metrics (K of them) show
   up as sub-columns under each dataset. Each entry is:
       (display_name, dotted_path_into_json)
   Dotted paths are resolved against the JSON object, e.g.
       "Runtime.Per_Document.Mean" -> json["Runtime"]["Per_Document"]["Mean"]
3. Edit METADATA_FIELDS the same way if you want extra columns (e.g.
   Timestamp, T) appended after the statistical metrics for each dataset
   block. Leave it as an empty list if you don't want any metadata columns.
------------------------------------------------------------------------
"""

import os
import csv
import json


# =========================================================================
# CONFIGURATION
# =========================================================================

INPUT_FOLDER = "/path/to/your/results/folder"
OUTPUT_CSV = "/path/to/your/output/ke_stats_table.csv"

# The statistical metrics you want as sub-columns under every dataset.
# (display_name, dotted_path_in_json)
STAT_METRICS = [
    ("Runtime Mean",        "Runtime.Per_Document.Mean"),
    #("Runtime Median",      "Runtime.Per_Document.Median"),
    #("Runtime Min",         "Runtime.Per_Document.Min"),
    #("Runtime Max",         "Runtime.Per_Document.Max"),
    ("Runtime Per Dataset", "Runtime.Per_Dataset"),

    ("Keywords Count Mean",   "Keywords.Count.Mean"),
    #("Keywords Count Median", "Keywords.Count.Median"),
    #("Keywords Count Min",    "Keywords.Count.Min"),
    #("Keywords Count Max",    "Keywords.Count.Max"),

    ("Keywords Length Mean",   "Keywords.Length.Mean"),
    #("Keywords Length Median", "Keywords.Length.Median"),

    #("Avg Doc Words",  "Vocabulary.Avg_Doc_Words"),
    ("Non Word Ratio", "Vocabulary.Non_Word_Ratio"),
]

# Optional metadata columns, appended after STAT_METRICS in each dataset
# block. Leave empty ([]) if you don't want any.
METADATA_FIELDS = [
    # ("Timestamp", "Timestamp"),
    # ("T", "T"),
]

# The filename suffix used to identify stats files.
FILENAME_SUFFIX = "_stats.json"

# Only files whose immediate parent folder name contains this value (as a
# substring) will be picked up. E.g. if TARGET_MAX_LEN = "512", a file at
# ".../max_len_512/methodA_dataset1_stats.json" will be included, while one
# at ".../max_len_256/methodA_dataset1_stats.json" will be skipped.
TARGET_MAX_LEN = "FULL"

# Controls row ordering. KE methods are grouped by their "Category" field
# (present in every stats JSON) and groups are listed in this order. The
# order of KE methods *within* a group does not matter (alphabetical).
# This list must contain every category value that appears in your files.
GROUP_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]

# ---- Config for the second ("merged runtime") table -----------------------
# The primary category whose KE method names become the rows of the second
# table. Each primary-category KE method's name must contain, as a
# substring, the name of exactly one KE method from one of the secondary
# categories below. That secondary method's "Runtime.Per_Dataset" value (per
# dataset) is added to the primary method's own "Runtime.Per_Dataset" value.
PRIMARY_CATEGORY = "A"
SECONDARY_CATEGORIES = ["B", "C"]


# =========================================================================
# IMPLEMENTATION
# =========================================================================

def find_stats_files(root_folder, target_max_len):
    """Recursively find all files ending with FILENAME_SUFFIX whose immediate
    parent folder name contains `target_max_len` as a substring.
    Returns a de-duplicated, sorted list of absolute paths."""
    found = set() # set() is used so that there are no duplicate stat files
    skipped_count = 0
    for dirpath, _dirnames, filenames in os.walk(root_folder):
        parent_folder_name = os.path.basename(dirpath)
        for fname in filenames:
            if fname.endswith(FILENAME_SUFFIX):
                if target_max_len not in parent_folder_name:
                    skipped_count += 1
                    continue
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                real_path = os.path.realpath(full_path)  # resolve symlinks
                found.add(real_path)

    if skipped_count:
        print(f"Skipped {skipped_count} '*{FILENAME_SUFFIX}' file(s) whose parent "
              f"folder name did not contain '{target_max_len}'.")
        
    return sorted(found)


def get_nested(d, dotted_path):
    """Fetch a value from a nested dict using a dotted path string.
    Returns None if any part of the path is missing."""
    current = d
    for key in dotted_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
        
    return current


def load_stats_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def match_secondary_to_primary(primary_names, secondary_names):
    """For each secondary-category KE method name, find the single
    primary-category KE method name that contains it as a substring.
    Returns a dict: primary_name -> matched_secondary_name.
    Raises SystemExit if a secondary name has zero or multiple matches,
    or if a primary name ends up with more than one match."""

    secondary_names = ["_".join(sn.split("_")[1:]) for sn in secondary_names]
    primary_names = ["_".join(pn.split("_")[2:]) for pn in primary_names]


    primary_to_secondary = {}
    for sec_name in secondary_names:
        matches = [p for p in primary_names if sec_name == p]


        if len(matches) == 0:
            raise SystemExit(
                f"No primary-category ('{PRIMARY_CATEGORY}') KE method name contains "
                f"'{sec_name}' as a substring. Every secondary-category method must "
                "match exactly one primary-category method."
            )
        if len(matches) > 1:
            raise SystemExit(
                f"Secondary-category KE method '{sec_name}' matches multiple "
                f"primary-category ('{PRIMARY_CATEGORY}') method names as a substring: "
                f"{matches}. Matching must be unique (1-to-1)."
            )
        primary_name = matches[0]
        if primary_name in primary_to_secondary:
            raise SystemExit(
                f"Primary-category method '{primary_name}' matches more than one "
                f"secondary-category method: '{primary_to_secondary[primary_name]}' "
                f"and '{sec_name}'. Matching must be unique (1-to-1)."
            )
        primary_to_secondary[f"Llama3_T10_{primary_name}"] = f"Parallel_{sec_name}"
 
    missing = sorted(set(primary_names) - set(primary_to_secondary.keys()))
    if missing:
        raise SystemExit(
            f"The following primary-category ('{PRIMARY_CATEGORY}') KE method(s) have "
            f"no matching secondary-category method: {missing}."
        )
 
    return primary_to_secondary
 
 
def build_merged_runtime_table(data, ke_categories, datasets):
    """Build the rows for the second table: primary-category KE methods,
    one 'Runtime.Per_Dataset' sub-column per dataset, where each value is
    the primary method's own Runtime.Per_Dataset plus its matched
    secondary-category method's Runtime.Per_Dataset (same dataset).
    Returns (primary_names_sorted, data_rows)."""

    primary_names =   [ke for ke, cat in ke_categories.items() if cat == PRIMARY_CATEGORY]
    secondary_names = [ke for ke, cat in ke_categories.items() if cat in SECONDARY_CATEGORIES]
 
    primary_to_secondary = match_secondary_to_primary(primary_names, secondary_names)
 
    primary_names_sorted = sorted(primary_names)
 
    data_rows = []
    for primary_name in primary_names_sorted:
        secondary_name = primary_to_secondary[primary_name]
        row = [primary_name]
        for dataset in datasets:
            primary_record = data.get(primary_name, {}).get(dataset)
            secondary_record = data.get(secondary_name, {}).get(dataset)
 
            primary_value = get_nested(primary_record, "Runtime.Per_Dataset") if primary_record else None
            secondary_value = get_nested(secondary_record, "Runtime.Per_Dataset") if secondary_record else None
 
            if primary_value is None or secondary_value is None:
                row.append("")
            else:
                row.append(primary_value + secondary_value)
        data_rows.append(row)
 
    return primary_names_sorted, data_rows


def main():
    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"INPUT_FOLDER does not exist or is not a directory: {INPUT_FOLDER}")

    stats_files = find_stats_files(INPUT_FOLDER, TARGET_MAX_LEN)
    print(f"Found {len(stats_files)} '*{FILENAME_SUFFIX}' file(s) matching "
          f"TARGET_MAX_LEN='{TARGET_MAX_LEN}'.")

    print("The paths of the statistics files are: \n")
    for path in stats_files:
        print(path)

    print()

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


        # Something like this:
        """
        data = {
            RAKE: {
                "MDPI": record <--> {...}
                "Krapivin": record <--> {...}
                "NUS": record <--> {...}
                ...        
            },
            YAKE: {
                "MDPI": record <--> {...}
                "Krapivin": record <--> {...}
                "NUS": record <--> {...}
                ...
            },
            ...
        }
        """

    if max_length_warnings:
        print("\n[WARNING] Inconsistent 'Datasets_Max_Length' values found "
              f"(keeping the first one seen: {datasets_max_length}):")
        for path, dml in max_length_warnings:
            print(f"    {path} -> {dml}")

    if category_warnings:
        print("\n[WARNING] Inconsistent 'Category' values found for the same KE method "
              "(keeping the first one seen):")
        for path, ke, kept, found in category_warnings:
            print(f"    {path} -> KE '{ke}' has Category '{found}', "
                  f"but '{kept}' was used earlier")
 
    missing_category = sorted(ke for ke in data.keys() if ke not in ke_categories)
    if missing_category:
        raise SystemExit(
            "The following KE method(s) are missing a 'Category' field in their "
            f"stats JSON file(s): {missing_category}. Every stats file must include "
            "a 'Category' key that appears in GROUP_ORDER."
        )
 
    unknown_categories = sorted({cat for cat in ke_categories.values() if cat not in GROUP_ORDER})
    if unknown_categories:
        raise SystemExit(
            f"The following Category value(s) are not listed in GROUP_ORDER: "
            f"{unknown_categories}. Add them to GROUP_ORDER (in your desired group order)."
        )

    #ke_methods = sorted(data.keys())
    ke_methods = sorted(
        data.keys(),
        key=lambda ke: (GROUP_ORDER.index(ke_categories[ke]), ke)
    )


    print(f"The Keyword Extraction (KE) methods are: ")
    for ke in ke_methods:
        print()
    print()


    datasets = sorted({ds for ke_data in data.values() for ds in ke_data.keys()}) # de-duplicated sorted dataset names

    if not ke_methods or not datasets:
        print("No valid records found. Nothing to write.")
        return

    

    # Combine stat metrics + metadata fields into the ordered list of
    # sub-columns for each dataset block.
    all_fields = STAT_METRICS + METADATA_FIELDS  # [(display_name, dotted_path), ...]
    k = len(all_fields)

    # ---------------------------------------------------------------
    # Build the two header rows
    # ---------------------------------------------------------------
    header_row1 = ["KE Method"]
    header_row2 = [""]
    for dataset in datasets:
        header_row1.append(dataset)
        header_row1.extend([""] * (k - 1))
        header_row2.extend([display_name for display_name, _ in all_fields])

    # ---------------------------------------------------------------
    # Build data rows
    # ---------------------------------------------------------------
    data_rows = []
    for ke in ke_methods:
        row = [ke]
        for dataset in datasets:
            record = data.get(ke, {}).get(dataset)
            if record is None:
                row.extend([""] * k)
                continue
            for _display_name, dotted_path in all_fields:
                value = get_nested(record, dotted_path)
                row.append("" if value is None else value)
        data_rows.append(row)

    # ---------------------------------------------------------------
    # Write CSV: first table, then the merged-runtime table below it
    # ---------------------------------------------------------------
    primary_names_sorted, merged_data_rows = build_merged_runtime_table(data, ke_categories, datasets)
 
    merged_header_row1 = ["KE Method"]
    merged_header_row2 = [""]
    for dataset in datasets:
        merged_header_row1.append(dataset)
        merged_header_row2.append("Runtime.Per_Dataset")



    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Datasets Max Length: {datasets_max_length}"])
        writer.writerow([])  # blank spacer row
        writer.writerow(header_row1)
        writer.writerow(header_row2)
        writer.writerows(data_rows)

        writer.writerow([])  # blank spacer row between tables
        writer.writerow([])

        writer.writerow([f"Merged Runtime Table (Category '{PRIMARY_CATEGORY}' "
                          f"+ matched {SECONDARY_CATEGORIES} method)"])
                          
        writer.writerow(merged_header_row1)
        writer.writerow(merged_header_row2)
        writer.writerows(merged_data_rows)


    print(f"\nWrote table 1 with {len(ke_methods)} KE method(s) x "
          f"{len(datasets)} dataset(s) x {k} sub-column(s).")
    
    print(f"Wrote table 2 (merged runtime) with {len(primary_names_sorted)} KE method(s) x "
          f"{len(datasets)} dataset(s) x 1 sub-column.")

    print(f"Both tables written to:\n  {OUTPUT_CSV}")
    print(f"Datasets Max Length: {datasets_max_length}")


if __name__ == "__main__":
    main()