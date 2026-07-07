# Auto-extracted from notebooks/01_zuco2_scientific_ml_pipeline.ipynb
# The notebook remains the authoritative executable workflow.

# %% [cell 6]
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped. This is expected outside Google Colab.")
    print("Message:", e)

# %% [cell 8]
# !pip install h5py pandas numpy matplotlib scikit-learn scipy

from pathlib import Path
import os
import re
import warnings
import itertools
import hashlib
import math
import requests

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.base import clone

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 140)

print("Packages imported successfully.")

# %% [cell 10]
# === User configuration ===
# Annotation:
# Keep ZuCo files in a dedicated folder rather than searching the whole Google Drive.
# Recommended folder:
# /content/drive/MyDrive/ZuCo/

DATA_DIR = Path("/content/drive/MyDrive/ZuCo")
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional folder where OSF-downloaded files will be stored.
OSF_DOWNLOAD_DIR = DATA_DIR / "osf_downloads"
OSF_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Quick-start default: one participant.
# Add more subjects after the pipeline works.
# Examples: ["YAC"], ["YAC", "YAK", "YAG"]
SUBJECTS = ["YAC","YAK","YAG","YDG","YDR","YFR","YFS","YHS","YLS","YIS","YMD","YRK","YRP","YSD","YSL","YTL"]

# Use ["NR"] if you only have normal-reading files.
# Use ["NR", "TSR"] if both tasks are available.
TASKS = ["NR", "TSR"]

# If True, the notebook will automatically include all local files matching results*_*.mat.
# This is useful after downloading multiple ZuCo files from OSF.
AUTO_DISCOVER_ALL_LOCAL_MAT_FILES = False

# Optional OSF indexing/downloading.
# Leave False for ordinary local-file analysis.
USE_OSF_DISCOVERY = False
DOWNLOAD_FROM_OSF = False
OSF_ROOT_NODE_ID = "d7frw"
OSF_MAX_DOWNLOADS = 4

# Validation grouping:
# "sentence_hash" is stricter because it keeps identical sentence text out of both train and test.
# "sentence_uid" is less strict and only separates subject-task-sentence IDs.
GROUP_COL_FOR_VALIDATION = "sentence_hash"

# Should task be included as a feature in Models A/B?
# For pure within-task load modelling, set this to False.
# For mixed NR+TSR modelling where task context is meaningful, True is acceptable.
USE_TASK_AS_FEATURE = True

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Accuracy-oriented modelling settings
# ---------------------------------------------------------------------
# These options improve model accuracy in a statistically defensible way.
# They use only training/validation data for tuning. The test set remains held out.

TUNE_MODEL_HYPERPARAMETERS = True
TUNE_CLASSIFICATION_THRESHOLD = True

# Optimise thresholds for balanced accuracy by default because high-load labels
# are usually imbalanced. Change to "f1" if your priority is positive-class detection.
THRESHOLD_OPTIMIZATION_METRIC = "balanced_accuracy"

# Fraction of the training portion held out for validation tuning.
VALIDATION_SIZE = 0.20

# Keep grids small so Colab remains responsive.
TUNING_MAX_CANDIDATES = 12


print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("OSF_DOWNLOAD_DIR:", OSF_DOWNLOAD_DIR)
print("SUBJECTS:", SUBJECTS)
print("TASKS:", TASKS)
print("AUTO_DISCOVER_ALL_LOCAL_MAT_FILES:", AUTO_DISCOVER_ALL_LOCAL_MAT_FILES)
print("USE_OSF_DISCOVERY:", USE_OSF_DISCOVERY)
print("DOWNLOAD_FROM_OSF:", DOWNLOAD_FROM_OSF)
print("GROUP_COL_FOR_VALIDATION:", GROUP_COL_FOR_VALIDATION)
print("USE_TASK_AS_FEATURE:", USE_TASK_AS_FEATURE)

# %% [cell 12]
def osf_api_get_json(url, timeout=30):
    """
    Get JSON from the OSF API.

    This function is used only when USE_OSF_DISCOVERY=True.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def osf_paginated_items(url, max_items=None):
    """
    Collect paginated OSF API results.
    """
    items = []
    next_url = url

    while next_url is not None:
        payload = osf_api_get_json(next_url)
        items.extend(payload.get("data", []))

        if max_items is not None and len(items) >= max_items:
            return items[:max_items]

        next_url = payload.get("links", {}).get("next")

    return items


def osf_node_url(node_id):
    return f"https://api.osf.io/v2/nodes/{node_id}/"


def osf_children_url(node_id):
    return f"https://api.osf.io/v2/nodes/{node_id}/children/"


def osf_files_url(node_id):
    return f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/"


def osf_list_child_nodes(root_node_id, max_nodes=100):
    """
    List first-level child components for an OSF node.
    """
    children = osf_paginated_items(osf_children_url(root_node_id), max_items=max_nodes)
    rows = []
    for item in children:
        rows.append({
            "node_id": item.get("id"),
            "title": item.get("attributes", {}).get("title"),
            "category": item.get("attributes", {}).get("category"),
            "api_url": item.get("links", {}).get("self"),
        })
    return pd.DataFrame(rows)


def osf_item_children_url(item):
    """
    Return the API URL for a folder's children, if available.

    OSF file/folder objects usually store folder children under:
    relationships -> files -> links -> related -> href
    """
    return (
        item.get("relationships", {})
        .get("files", {})
        .get("links", {})
        .get("related", {})
        .get("href")
    )


def osf_list_storage_files_recursive(node_id, component_title=None, max_items=2000, max_depth=8):
    """
    Recursively list files/folders in OSF Storage for a node.

    Why recursive?
    Many OSF projects store data inside nested folders. A top-level-only index can miss
    the actual .mat files.
    """
    rows = []
    queue = [(osf_files_url(node_id), "", 0)]
    visited_urls = set()

    while queue and len(rows) < max_items:
        url, parent_path, depth = queue.pop(0)
        if url in visited_urls or depth > max_depth:
            continue
        visited_urls.add(url)

        try:
            items = osf_paginated_items(url, max_items=max_items)
        except Exception as e:
            rows.append({
                "node_id": node_id,
                "component_title": component_title,
                "kind": "error",
                "name": None,
                "path": parent_path,
                "size": None,
                "download_url": None,
                "api_url": url,
                "depth": depth,
                "error": str(e)
            })
            continue

        for item in items:
            attr = item.get("attributes", {})
            links = item.get("links", {})
            kind = attr.get("kind")
            name = attr.get("name")
            item_path = f"{parent_path}/{name}" if parent_path and name else (name or parent_path)

            rows.append({
                "node_id": node_id,
                "component_title": component_title,
                "kind": kind,
                "name": name,
                "path": item_path,
                "size": attr.get("size"),
                "download_url": links.get("download"),
                "api_url": links.get("self"),
                "depth": depth,
                "error": None
            })

            if kind == "folder":
                child_url = osf_item_children_url(item)
                if child_url is not None:
                    queue.append((child_url, item_path, depth + 1))

    return pd.DataFrame(rows)


def download_file(url, output_path, chunk_size=1024 * 1024):
    """
    Stream-download a file from OSF to Google Drive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    return output_path


osf_components_df = pd.DataFrame()
osf_file_index_df = pd.DataFrame()

if USE_OSF_DISCOVERY:
    print(f"Indexing OSF root node: {OSF_ROOT_NODE_ID}")

    root_payload = osf_api_get_json(osf_node_url(OSF_ROOT_NODE_ID))
    root_title = root_payload.get("data", {}).get("attributes", {}).get("title")
    print("OSF root title:", root_title)

    osf_components_df = osf_list_child_nodes(OSF_ROOT_NODE_ID)
    display(osf_components_df)

    file_tables = []

    # Include root storage recursively.
    try:
        file_tables.append(osf_list_storage_files_recursive(OSF_ROOT_NODE_ID, component_title=root_title))
    except Exception as e:
        print("Root recursive file listing failed:", e)

    # Include one-level child component storage recursively.
    for _, row in osf_components_df.iterrows():
        try:
            file_tables.append(osf_list_storage_files_recursive(row["node_id"], component_title=row["title"]))
        except Exception as e:
            print(f"Recursive file listing failed for {row['node_id']} {row['title']}: {e}")

    if file_tables:
        osf_file_index_df = pd.concat(file_tables, ignore_index=True)
        osf_file_index_path = OUTPUT_DIR / "osf_file_index.csv"
        osf_file_index_df.to_csv(osf_file_index_path, index=False)
        print("Saved OSF file index:", osf_file_index_path)
        display(osf_file_index_df.head(50))

        mat_candidates = osf_file_index_df[
            (osf_file_index_df["kind"] == "file")
            & osf_file_index_df["name"].astype(str).str.endswith(".mat", na=False)
            & osf_file_index_df["download_url"].notna()
        ].copy()
        print("Number of recursive .mat candidates found:", len(mat_candidates))
        display(mat_candidates.head(20))

        if DOWNLOAD_FROM_OSF and len(mat_candidates) > 0:
            selected = mat_candidates.head(OSF_MAX_DOWNLOADS).copy()
            downloaded = []
            for _, row in selected.iterrows():
                safe_name = str(row["name"]).replace("/", "_")
                out = OSF_DOWNLOAD_DIR / safe_name
                if out.exists():
                    print("Already exists:", out)
                else:
                    print("Downloading:", row["path"])
                    download_file(row["download_url"], out)
                downloaded.append(out)
            print("Downloaded files:")
            for p in downloaded:
                print(" ", p)
    else:
        print("No OSF file tables were collected.")
else:
    print("OSF discovery is off. Set USE_OSF_DISCOVERY=True to index the OSF node tree.")

# %% [cell 14]
def parse_subject_task_from_filename(path):
    """
    Parse filenames such as resultsYAC_NR.mat or resultsYAC_TSR.mat.

    Returns (subject_id, task) or (None, None) if parsing fails.
    """
    name = Path(path).name
    match = re.match(r"results(?P<subject>[A-Za-z0-9]+)_(?P<task>[A-Za-z0-9]+)\.mat$", name)
    if match:
        return match.group("subject"), match.group("task")
    return None, None


def find_zuco_mat_file(data_dirs, subject, task):
    """
    Find one ZuCo .mat file for a subject-task pair across one or more directories.
    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]
    else:
        data_dirs = [Path(d) for d in data_dirs]

    exact_name = f"results{subject}_{task}.mat"

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        exact_matches = list(data_dir.rglob(exact_name))
        if exact_matches:
            return exact_matches[0]

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        fallback_matches = list(data_dir.rglob(f"*{subject}*{task}*.mat"))
        if fallback_matches:
            return fallback_matches[0]

    return None


def discover_all_local_zuco_mat_files(data_dirs):
    """
    Auto-discover all local files matching results*_*.mat.
    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]
    else:
        data_dirs = [Path(d) for d in data_dirs]

    rows = []
    seen_paths = set()

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in data_dir.rglob("results*_*.mat"):
            if path in seen_paths:
                continue
            subject, task = parse_subject_task_from_filename(path)
            if subject is None:
                continue
            rows.append({
                "subject_id": subject,
                "task": task,
                "path": path,
                "file_size_mb": path.stat().st_size / 1024 / 1024,
                "discovery_mode": "auto"
            })
            seen_paths.add(path)

    return rows


LOCAL_SEARCH_DIRS = [DATA_DIR, OSF_DOWNLOAD_DIR]
available_files = []

if AUTO_DISCOVER_ALL_LOCAL_MAT_FILES:
    available_files = discover_all_local_zuco_mat_files(LOCAL_SEARCH_DIRS)
else:
    for subject in SUBJECTS:
        for task in TASKS:
            path = find_zuco_mat_file(LOCAL_SEARCH_DIRS, subject, task)
            if path is None:
                print(f"Missing file: subject={subject}, task={task}")
            else:
                available_files.append({
                    "subject_id": subject,
                    "task": task,
                    "path": path,
                    "file_size_mb": path.stat().st_size / 1024 / 1024,
                    "discovery_mode": "manual"
                })
                print(f"Found: subject={subject}, task={task}")
                print("  ", path)
                print("  size MB:", round(path.stat().st_size / 1024 / 1024, 2))

# De-duplicate by path.
if available_files:
    available_files_df = pd.DataFrame(available_files)
    available_files_df["path_str"] = available_files_df["path"].astype(str)
    available_files_df = available_files_df.drop_duplicates("path_str").drop(columns=["path_str"])
    available_files = available_files_df.to_dict("records")
else:
    available_files_df = pd.DataFrame(columns=["subject_id", "task", "path", "file_size_mb", "discovery_mode"])

display(available_files_df)

if len(available_files) == 0:
    raise FileNotFoundError(
        "No .mat files found. Check DATA_DIR, OSF_DOWNLOAD_DIR, SUBJECTS, TASKS, and file names."
    )

print("Total local .mat files selected for extraction:", len(available_files))

# %% [cell 16]
example_path = available_files[0]["path"]

with h5py.File(example_path, "r") as f:
    print("Example file:", example_path)
    print("\nTop-level keys:")
    print(list(f.keys()))

    if "sentenceData" in f:
        print("\nsentenceData fields:")
        print(list(f["sentenceData"].keys()))
    else:
        print("Warning: sentenceData not found.")

# %% [cell 18]
def load_matlab_string(matlab_obj):
    """
    Convert a MATLAB/HDF5 character array into a Python string.
    """
    try:
        return "".join(chr(int(c[0])) for c in matlab_obj)
    except Exception:
        try:
            arr = np.asarray(matlab_obj).squeeze()
            return "".join(chr(int(c)) for c in arr)
        except Exception:
            return None


def read_scalar_or_none(f, ref):
    """
    Read a scalar numeric value from an HDF5 reference.
    Return None if the value is missing or not scalar-like.
    """
    try:
        arr = np.asarray(f[ref][()])
        if arr.size == 0:
            return None
        value = np.squeeze(arr)
        if np.asarray(value).size != 1:
            return None
        return float(value)
    except Exception:
        return None


def is_real_word(word):
    """
    Keep tokens that contain at least one letter or digit.
    This removes punctuation-only tokens.
    """
    if word is None:
        return False
    return re.search(r"[A-Za-z0-9]", str(word)) is not None


def has_required_fields(h5_group, fields):
    """
    Check whether an HDF5 group contains all required fields.
    """
    available = set(list(h5_group.keys()))
    return all(field in available for field in fields)

# %% [cell 20]
def read_word_field_scalar(f, word_obj, field, word_idx):
    """
    Safely read a scalar field for one word.
    Returns None if the field is absent or unreadable.
    """
    if field not in word_obj.keys():
        return None
    try:
        return read_scalar_or_none(f, word_obj[field][word_idx][0])
    except Exception:
        return None


def extract_zuco_eye_tracking(mat_path, subject_id, task, verbose=True):
    """
    Extract word-level eye-tracking variables from a ZuCo .mat file.

    v2 change:
    SFD is treated as optional. Missing SFD should not cause the whole sentence to be skipped.
    """
    mat_path = Path(mat_path)
    rows = []
    logs = []

    with h5py.File(mat_path, "r") as f:
        if "sentenceData" not in f:
            raise KeyError(f"sentenceData not found in {mat_path}")

        sentence_data = f["sentenceData"]
        n_sentences = len(sentence_data["content"])

        if verbose:
            print(f"Extracting {mat_path.name}: {n_sentences} sentences")

        for sent_idx in range(n_sentences):
            try:
                sentence_ref = sentence_data["content"][sent_idx][0]
                sentence_text = load_matlab_string(f[sentence_ref])

                word_ref = sentence_data["word"][sent_idx][0]
                word_obj = f[word_ref]

                # SFD is intentionally not required.
                required = ["content", "FFD", "GD", "GPT", "TRT", "nFixations"]
                if not has_required_fields(word_obj, required):
                    logs.append({
                        "subject_id": subject_id,
                        "task": task,
                        "source_file": mat_path.name,
                        "sentence_id": sent_idx,
                        "reason": "missing required word-level fields",
                        "available_fields": ",".join(list(word_obj.keys()))
                    })
                    continue

                n_words = len(word_obj["content"])

                for word_idx in range(n_words):
                    word_text_ref = word_obj["content"][word_idx][0]
                    word_text = load_matlab_string(f[word_text_ref])

                    if not is_real_word(word_text):
                        continue

                    rows.append({
                        "subject_id": subject_id,
                        "task": task,
                        "source_file": mat_path.name,
                        "sentence_id": sent_idx,
                        "word_id": word_idx,
                        "sentence": sentence_text,
                        "word": word_text,
                        "word_length": len(str(word_text)),
                        "sentence_length_words": n_words,
                        "word_position_norm": word_idx / max(n_words - 1, 1),
                        "FFD": read_word_field_scalar(f, word_obj, "FFD", word_idx),
                        "GD": read_word_field_scalar(f, word_obj, "GD", word_idx),
                        "GPT": read_word_field_scalar(f, word_obj, "GPT", word_idx),
                        "TRT": read_word_field_scalar(f, word_obj, "TRT", word_idx),
                        "SFD": read_word_field_scalar(f, word_obj, "SFD", word_idx),
                        "nFixations": read_word_field_scalar(f, word_obj, "nFixations", word_idx),
                        "SFD_available_in_file": int("SFD" in word_obj.keys())
                    })

            except Exception as e:
                logs.append({
                    "subject_id": subject_id,
                    "task": task,
                    "source_file": mat_path.name,
                    "sentence_id": sent_idx,
                    "reason": str(e),
                    "available_fields": None
                })

    df_out = pd.DataFrame(rows)

    if verbose:
        print("  extracted rows:", len(df_out))
        print("  skipped/problematic sentences:", len(logs))

    return df_out, logs


all_tables = []
all_logs = []

for item in available_files:
    table, log = extract_zuco_eye_tracking(
        mat_path=item["path"],
        subject_id=item["subject_id"],
        task=item["task"],
        verbose=True
    )
    all_tables.append(table)
    all_logs.extend(log)

df_raw = pd.concat(all_tables, ignore_index=True)
log_df = pd.DataFrame(all_logs)

print("\nCombined raw table:", df_raw.shape)
display(df_raw.head())

print("\nExtraction log:", log_df.shape)
display(log_df.head())

if "SFD_available_in_file" in df_raw.columns:
    print("\nSFD availability by source file:")
    display(df_raw.groupby("source_file")["SFD_available_in_file"].mean().reset_index())

# %% [cell 22]
def normalize_sentence_text(text):
    """
    Normalize sentence text for grouping identical sentence content across subjects/tasks.
    This reduces train/test leakage when the same sentence appears in multiple files.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_eye_tracking_table(df):
    """
    Clean extracted eye-tracking data and add derived variables.

    Critical design choices:
    - skipped words are preserved as behaviour
    - duration analyses are restricted to fixated words
    - sentence_hash groups identical sentence text across subjects/tasks
    """
    df = df.copy()

    duration_cols = ["FFD", "GD", "GPT", "TRT", "SFD"]
    numeric_cols = duration_cols + [
        "nFixations", "word_length", "sentence_length_words", "word_position_norm"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Negative durations are impossible; set them to missing and flag them.
    for col in duration_cols:
        df[f"{col}_negative_flag"] = (df[col] < 0).astype(int)
        df.loc[df[col] < 0, col] = np.nan

    # A word is fixated if nFixations is observed and greater than zero.
    df["is_fixated"] = (df["nFixations"].fillna(0) > 0).astype(int)
    df["skipped"] = (df["is_fixated"] == 0).astype(int)

    # Log transforms for right-skewed reading-time variables.
    for col in duration_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    # Late processing proxy:
    # TRT includes all time on a word; GD captures first-pass gaze.
    # TRT - GD approximates extra late/re-reading burden.
    df["late_time"] = df["TRT"] - df["GD"]
    df["late_time_negative_flag"] = (df["late_time"] < 0).astype(int)
    df.loc[df["late_time"] < 0, "late_time"] = np.nan
    df["log_late_time"] = np.log1p(df["late_time"])

    # Unique sentence ID.
    df["sentence_uid"] = (
        df["subject_id"].astype(str)
        + "_"
        + df["task"].astype(str)
        + "_sent"
        + df["sentence_id"].astype(str)
    )

    # Stricter group: same sentence text gets the same hash across subjects/tasks.
    df["sentence_norm"] = df["sentence"].apply(normalize_sentence_text)
    df["sentence_hash"] = pd.util.hash_pandas_object(df["sentence_norm"], index=False).astype(str)

    # Dataset/file identifiers for multi-file analysis.
    df["subject_task"] = df["subject_id"].astype(str) + "_" + df["task"].astype(str)
    df["source_dataset"] = df["source_file"].astype(str).str.replace(".mat", "", regex=False)

    # Basic quality flags.
    df["any_negative_duration_flag"] = df[[f"{c}_negative_flag" for c in duration_cols]].sum(axis=1).gt(0).astype(int)
    df["duration_missing_but_fixated_flag"] = (
        (df["is_fixated"] == 1)
        & df[["FFD", "GD", "TRT"]].isna().any(axis=1)
    ).astype(int)

    return df


df = clean_eye_tracking_table(df_raw)

print("Cleaned table:", df.shape)
display(df.head())

print("\nRows by subject and task:")
display(df.groupby(["subject_id", "task"]).size().rename("n_words").reset_index())

print("\nQuality flags:")
quality_summary = df[[
    "any_negative_duration_flag",
    "late_time_negative_flag",
    "duration_missing_but_fixated_flag"
]].mean().rename("proportion_flagged").reset_index()
display(quality_summary)

# %% [cell 24]
BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "in", "on", "at", "to", "for",
    "from", "by", "with", "about", "as", "into", "like", "through", "after", "over", "between",
    "out", "against", "during", "without", "before", "under", "around", "among",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "has", "have", "had",
    "he", "she", "it", "they", "we", "you", "i", "his", "her", "their", "our", "your", "its",
    "this", "that", "these", "those", "there", "here", "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "not", "no", "so", "than", "then", "too", "very", "can", "could", "may", "might", "must", "should", "would", "will"
}


def add_basic_lexical_features(df):
    """
    Add transparent lexical/contextual baseline features.
    """
    df = df.copy()
    df["word_str"] = df["word"].astype(str)
    df["word_lower"] = df["word_str"].str.lower()

    df["log_word_length"] = np.log1p(df["word_length"])
    df["is_stopword"] = df["word_lower"].isin(BASIC_STOPWORDS).astype(int)
    df["is_numeric"] = df["word_str"].str.fullmatch(r"[0-9]+(?:[\.,][0-9]+)?").fillna(False).astype(int)
    df["has_digit"] = df["word_str"].str.contains(r"\d", regex=True).fillna(False).astype(int)
    df["is_capitalized"] = df["word_str"].str.match(r"^[A-Z][a-z]+", na=False).astype(int)
    df["is_all_caps"] = df["word_str"].str.match(r"^[A-Z]+$", na=False).astype(int)
    df["is_short_word"] = (df["word_length"] <= 3).astype(int)
    df["is_long_word"] = (df["word_length"] >= 8).astype(int)
    df["is_sentence_initial"] = (df["word_id"] == 0).astype(int)
    df["is_sentence_final"] = (df["word_id"] == df["sentence_length_words"] - 1).astype(int)

    return df


df = add_basic_lexical_features(df)

print("Added lexical features. Example:")
display(df[[
    "word", "word_length", "log_word_length", "is_stopword", "is_numeric",
    "is_capitalized", "is_short_word", "is_sentence_initial", "is_sentence_final"
]].head(20))

# %% [cell 26]
# === Data-quality and leakage audit ===

audit = {
    "n_rows": len(df),
    "n_subjects": df["subject_id"].nunique(),
    "n_tasks": df["task"].nunique(),
    "n_source_files": df["source_file"].nunique(),
    "n_sentence_uid": df["sentence_uid"].nunique(),
    "n_sentence_hash": df["sentence_hash"].nunique(),
    "skipping_rate": df["skipped"].mean(),
    "fixated_rate": df["is_fixated"].mean(),
}

audit_df = pd.DataFrame([audit]).T.rename(columns={0: "value"})
display(audit_df)

print("Repeated sentence text across files/tasks/subjects:")
sentence_repeat = (
    df.drop_duplicates(["sentence_hash", "source_file"])
    .groupby("sentence_hash")
    .agg(
        n_files=("source_file", "nunique"),
        example_sentence=("sentence", "first")
    )
    .reset_index()
    .sort_values("n_files", ascending=False)
)

display(sentence_repeat.head(10))

if GROUP_COL_FOR_VALIDATION not in df.columns:
    raise KeyError(f"GROUP_COL_FOR_VALIDATION={GROUP_COL_FOR_VALIDATION!r} is not a column in df.")

print(f"Validation will group by: {GROUP_COL_FOR_VALIDATION}")
print("Number of validation groups:", df[GROUP_COL_FOR_VALIDATION].nunique())

# %% [cell 28]
eye_cols = ["FFD", "GD", "GPT", "TRT", "SFD", "nFixations", "late_time"]

missing_summary = df[eye_cols].isna().sum().rename("n_missing").to_frame()
missing_summary["n_observed"] = len(df) - missing_summary["n_missing"]
missing_summary["pct_missing"] = missing_summary["n_missing"] / len(df)

display(missing_summary)

print("Overall skipping rate:", df["skipped"].mean())

print("\nSkipping rate by task:")
display(df.groupby("task")["skipped"].mean().rename("skipping_rate").reset_index())

# %% [cell 29]
duration_cols = ["FFD", "GD", "GPT", "TRT", "SFD", "late_time"]

missing_by_fixation = []

for col in duration_cols:
    tmp = (
        df.groupby("is_fixated")[col]
        .apply(lambda x: x.isna().mean())
        .rename("pct_missing")
        .reset_index()
    )
    tmp["variable"] = col
    missing_by_fixation.append(tmp)

missing_by_fixation = pd.concat(missing_by_fixation, ignore_index=True)
missing_by_fixation = missing_by_fixation[["variable", "is_fixated", "pct_missing"]]

display(missing_by_fixation)

# %% [cell 31]
df_fixated = df[df["is_fixated"] == 1].copy()

desc_cols = ["FFD", "GD", "GPT", "TRT", "SFD", "nFixations", "late_time"]

print("Fixated-word descriptive statistics:")
display(df_fixated[desc_cols].describe().T)

print("\nTask-level means among fixated words:")
display(df_fixated.groupby("task")[desc_cols].mean().reset_index())

# %% [cell 32]
plt.figure(figsize=(7, 4))
df_fixated["TRT"].dropna().hist(bins=50)
plt.xlabel("TRT: Total Reading Time")
plt.ylabel("Number of words")
plt.title("Distribution of total reading time")
plt.show()

plt.figure(figsize=(7, 4))
df_fixated["log_TRT"].dropna().hist(bins=50)
plt.xlabel("log(TRT + 1)")
plt.ylabel("Number of words")
plt.title("Distribution of log-transformed total reading time")
plt.show()

plt.figure(figsize=(7, 4))
df_fixated["late_time"].dropna().hist(bins=50)
plt.xlabel("late_time = TRT - GD")
plt.ylabel("Number of words")
plt.title("Distribution of late processing time")
plt.show()

# %% [cell 33]
if df["task"].nunique() >= 2:
    plt.figure(figsize=(7, 4))
    df_fixated.boxplot(column="TRT", by="task")
    plt.suptitle("")
    plt.title("TRT by task")
    plt.xlabel("Task")
    plt.ylabel("TRT")
    plt.show()

    plt.figure(figsize=(7, 4))
    df_fixated.boxplot(column="late_time", by="task")
    plt.suptitle("")
    plt.title("Late processing time by task")
    plt.xlabel("Task")
    plt.ylabel("late_time = TRT - GD")
    plt.show()
else:
    print("Only one task is available, so task comparison plots are skipped.")

# %% [cell 35]
def add_rank_based_binary_target(
    df,
    value_col,
    target_col,
    valid_mask,
    group_cols=("subject_id", "task"),
    top_quantile=0.75,
    min_group_n=20
):
    """
    Create a rank-based high-load target within groups.

    Why rank-based?
    If many late_time values are zero or tied, direct quantile thresholds can label
    too many or too few observations. Percentile ranks handle ties more gracefully.

    target = 1 means the observation is in the top tail of the group's distribution.
    """
    df[target_col] = np.nan
    df[f"{target_col}_percentile"] = np.nan

    grouped = df.loc[valid_mask].groupby(list(group_cols), dropna=False)

    for group_key, group in grouped:
        idx = group.index
        values = group[value_col].dropna()

        if len(values) < min_group_n or values.nunique() < 2:
            # Not enough variation for a meaningful binary target.
            continue

        ranks = group[value_col].rank(method="average", pct=True)
        df.loc[idx, f"{target_col}_percentile"] = ranks
        df.loc[idx, target_col] = (ranks >= top_quantile).astype(int)

    return df


def add_modelling_targets(df, top_quantile=0.75, min_group_n=20):
    """
    Add ML targets for skipping, high late load, high total load, and task.

    The main target is target_high_late_load:
        late_time = TRT - GD
        high = top 25% within subject-task group by percentile rank
    """
    df = df.copy()

    # Target A: skipping.
    df["target_skipped"] = df["skipped"].astype(int)

    # Target B: high late processing load.
    valid_late = (
        (df["is_fixated"] == 1)
        & df["late_time"].notna()
        & np.isfinite(df["late_time"])
    )

    df = add_rank_based_binary_target(
        df=df,
        value_col="late_time",
        target_col="target_high_late_load",
        valid_mask=valid_late,
        group_cols=("subject_id", "task"),
        top_quantile=top_quantile,
        min_group_n=min_group_n
    )

    # Comparison target: high total reading time.
    valid_trt = (
        (df["is_fixated"] == 1)
        & df["TRT"].notna()
        & np.isfinite(df["TRT"])
    )

    df = add_rank_based_binary_target(
        df=df,
        value_col="TRT",
        target_col="target_high_total_load",
        valid_mask=valid_trt,
        group_cols=("subject_id", "task"),
        top_quantile=top_quantile,
        min_group_n=min_group_n
    )

    # Optional task target.
    if set(df["task"].dropna().unique()).issuperset({"NR", "TSR"}):
        df["target_task_TSR"] = (df["task"] == "TSR").astype(int)
    else:
        df["target_task_TSR"] = np.nan

    return df


df_model = add_modelling_targets(df, top_quantile=0.75, min_group_n=20)

for target in ["target_skipped", "target_high_late_load", "target_high_total_load", "target_task_TSR"]:
    print("\n", target)
    display(df_model[target].value_counts(dropna=False, normalize=True).rename("proportion"))

print("\nTarget counts by subject-task:")
target_count_cols = ["target_high_late_load", "target_high_total_load"]
display(
    df_model
    .groupby(["subject_id", "task"])[target_count_cols]
    .agg(lambda s: s.value_counts(dropna=False).to_dict())
    .reset_index()
)

# %% [cell 37]
def existing_features(df, features):
    return [f for f in features if f in df.columns]


# ---------------------------------------------------------------------
# B-series alignment principle
# ---------------------------------------------------------------------
# B1-B4 should answer the same question:
# "Does early fixation information, especially FFD, improve prediction of late processing load?"
#
# Therefore, the central B-series feature sets deliberately avoid:
# GD, SFD, TRT, GPT, nFixations, late_time
# for target_high_late_load.
# ---------------------------------------------------------------------


# Safe FFD transformation.
# Statistical reason:
# FFD is often right-skewed. log_FFD improves numerical stability while preserving
# the same underlying early-fixation signal.
if "FFD" in df_model.columns:
    df_model["log_FFD"] = np.log1p(pd.to_numeric(df_model["FFD"], errors="coerce"))
else:
    df_model["log_FFD"] = np.nan


# A richer lexical baseline makes the scientific comparison more conservative.
LEXICAL_NUMERIC = existing_features(df_model, [
    "word_length",
    "log_word_length",
    "sentence_length_words",
    "word_position_norm",
    "is_stopword",
    "is_numeric",
    "has_digit",
    "is_capitalized",
    "is_all_caps",
    "is_short_word",
    "is_long_word",
    "is_sentence_initial",
    "is_sentence_final"
])

LEXICAL_CATEGORICAL = existing_features(df_model, ["task"]) if USE_TASK_AS_FEATURE else []


# Create a numeric task flag only for possible interaction construction.
if "task" in df_model.columns:
    df_model["task_is_TSR"] = (df_model["task"].astype(str).str.upper() == "TSR").astype(int)
else:
    df_model["task_is_TSR"] = 0


# B2: additive early-fixation model.
# FFD and log_FFD are both FFD-based. The research question remains the same.
EARLY_EYE_NUMERIC = existing_features(df_model, LEXICAL_NUMERIC + [
    "FFD",
    "log_FFD"
])


# B3: moderated FFD model.
# These interaction terms keep B3 on the same question: FFD signal under context moderation.
if "log_FFD" in df_model.columns:
    if "log_word_length" in df_model.columns:
        df_model["log_FFD_x_log_word_length"] = df_model["log_FFD"] * df_model["log_word_length"]
    if "word_position_norm" in df_model.columns:
        df_model["log_FFD_x_word_position_norm"] = df_model["log_FFD"] * df_model["word_position_norm"]
    if "is_long_word" in df_model.columns:
        df_model["log_FFD_x_is_long_word"] = df_model["log_FFD"] * df_model["is_long_word"]
    if "is_sentence_final" in df_model.columns:
        df_model["log_FFD_x_is_sentence_final"] = df_model["log_FFD"] * df_model["is_sentence_final"]
    if USE_TASK_AS_FEATURE and "task_is_TSR" in df_model.columns:
        df_model["log_FFD_x_task_is_TSR"] = df_model["log_FFD"] * df_model["task_is_TSR"]

FFD_CONTEXT_INTERACTION_FEATURES = existing_features(df_model, [
    "log_FFD_x_log_word_length",
    "log_FFD_x_word_position_norm",
    "log_FFD_x_is_long_word",
    "log_FFD_x_is_sentence_final",
    "log_FFD_x_task_is_TSR"
])

CONFIRMATORY_FFD_INTERACTION_NUMERIC = existing_features(
    df_model,
    EARLY_EYE_NUMERIC + FFD_CONTEXT_INTERACTION_FEATURES
)

# B4: nonlinear robustness model using the same allowed early-signal family.
CONFIRMATORY_FFD_NONLINEAR_NUMERIC = EARLY_EYE_NUMERIC.copy()

# Backward-compatible alias, but no longer "first-pass exploratory".
FIRST_PASS_EYE_NUMERIC = CONFIRMATORY_FFD_INTERACTION_NUMERIC.copy()


TASK_CLASSIFICATION_NUMERIC = existing_features(df_model, [
    "word_length",
    "log_word_length",
    "sentence_length_words",
    "word_position_norm",
    "is_stopword",
    "is_numeric",
    "is_capitalized",
    "is_short_word",
    "skipped",
    "FFD",
    "log_FFD",
    "GD",
    "TRT",
    "GPT",
    "SFD",
    "nFixations",
    "late_time"
])

TASK_CLASSIFICATION_CATEGORICAL = []


# Target-leakage guardrails.
FORBIDDEN_FEATURES_BY_TARGET = {
    "target_skipped": set(),
    "target_high_late_load": {
        "late_time", "log_late_time",
        "TRT", "log_TRT",
        "GPT", "log_GPT",
        "GD", "log_GD",
        "SFD", "log_SFD",
        "nFixations"
    },
    "target_high_total_load": {
        "TRT", "log_TRT",
        "GPT", "log_GPT",
        "GD", "log_GD",
        "SFD", "log_SFD",
        "nFixations",
        "late_time", "log_late_time"
    },
    "target_task_TSR": set()
}


def assert_no_target_leakage(target_col, features):
    """
    Block known forbidden predictors for a modelling target.

    Statistical reason:
    If a predictor is mathematically or behaviourally too close to the target,
    high accuracy can be inflated and scientifically misleading.
    """
    forbidden = FORBIDDEN_FEATURES_BY_TARGET.get(target_col, set())

    leaked = []
    for feature in features:
        for bad in forbidden:
            if (
                feature == bad
                or feature.startswith(f"{bad}_x_")
                or feature.endswith(f"_x_{bad}")
                or feature.startswith(f"log_{bad}_x_")
            ):
                leaked.append(feature)

    leaked = sorted(set(leaked))
    if leaked:
        raise ValueError(
            f"Target leakage risk for {target_col}. Remove forbidden predictors: {leaked}"
        )


print("LEXICAL_NUMERIC:", LEXICAL_NUMERIC)
print("LEXICAL_CATEGORICAL:", LEXICAL_CATEGORICAL)
print("EARLY_EYE_NUMERIC / B2 features:", EARLY_EYE_NUMERIC)
print("FFD_CONTEXT_INTERACTION_FEATURES:", FFD_CONTEXT_INTERACTION_FEATURES)
print("CONFIRMATORY_FFD_INTERACTION_NUMERIC / B3 features:", CONFIRMATORY_FFD_INTERACTION_NUMERIC)
print("CONFIRMATORY_FFD_NONLINEAR_NUMERIC / B4 features:", CONFIRMATORY_FFD_NONLINEAR_NUMERIC)
print("TASK_CLASSIFICATION_NUMERIC:", TASK_CLASSIFICATION_NUMERIC)

# Sanity checks: these should pass.
assert_no_target_leakage("target_high_late_load", LEXICAL_NUMERIC + LEXICAL_CATEGORICAL)
assert_no_target_leakage("target_high_late_load", EARLY_EYE_NUMERIC + LEXICAL_CATEGORICAL)
assert_no_target_leakage("target_high_late_load", CONFIRMATORY_FFD_INTERACTION_NUMERIC + LEXICAL_CATEGORICAL)
assert_no_target_leakage("target_high_late_load", CONFIRMATORY_FFD_NONLINEAR_NUMERIC + LEXICAL_CATEGORICAL)

# %% [cell 39]
# Defaults for older notebooks or partial execution.
TUNE_MODEL_HYPERPARAMETERS = globals().get("TUNE_MODEL_HYPERPARAMETERS", True)
TUNE_CLASSIFICATION_THRESHOLD = globals().get("TUNE_CLASSIFICATION_THRESHOLD", True)
THRESHOLD_OPTIMIZATION_METRIC = globals().get("THRESHOLD_OPTIMIZATION_METRIC", "balanced_accuracy")
VALIDATION_SIZE = globals().get("VALIDATION_SIZE", 0.20)
TUNING_MAX_CANDIDATES = globals().get("TUNING_MAX_CANDIDATES", 12)


def make_onehot_encoder():
    """
    Compatible OneHotEncoder for different sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features, categorical_features):
    """
    Build preprocessing pipelines.

    Numeric predictors:
    - median imputation
    - standardisation

    Categorical predictors:
    - most-frequent imputation
    - one-hot encoding

    Statistical annotation:
    Imputation is applied inside the training pipeline, so the test set does not influence
    the imputation values.
    """
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder())
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier_pipeline(numeric_features, categorical_features, model_type="logreg"):
    """
    Build preprocessing + classifier pipeline.
    """
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    if model_type == "dummy":
        classifier = DummyClassifier(strategy="most_frequent")

    elif model_type == "logreg":
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs"
        )

    elif model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1
        )

    else:
        raise ValueError("model_type must be 'dummy', 'logreg', or 'rf'.")

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", classifier)
    ])


def candidate_parameter_grid(model_type):
    """
    Small validation-tuning grids.

    Statistical annotation:
    The grid is deliberately small to reduce overfitting to the validation set and keep
    Colab runtime manageable.
    """
    if (not TUNE_MODEL_HYPERPARAMETERS) or model_type == "dummy":
        return [{}]

    if model_type == "logreg":
        candidates = [
            {"classifier__C": C, "classifier__class_weight": cw}
            for C in [0.05, 0.10, 0.30, 1.0, 3.0, 10.0]
            for cw in ["balanced", None]
        ]

    elif model_type == "rf":
        candidates = [
            {
                "classifier__max_depth": max_depth,
                "classifier__min_samples_leaf": min_leaf,
                "classifier__max_features": max_features,
                "classifier__class_weight": cw
            }
            for max_depth in [4, 8, None]
            for min_leaf in [2, 5, 10]
            for max_features in ["sqrt", 0.60]
            for cw in ["balanced", "balanced_subsample"]
        ]

    else:
        candidates = [{}]

    # Reproducible candidate subsampling if the grid is large.
    if len(candidates) > TUNING_MAX_CANDIDATES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(candidates), size=TUNING_MAX_CANDIDATES, replace=False)
        candidates = [candidates[i] for i in idx]

    return candidates


def is_trainable_binary_target(df, target_col, min_n=30):
    """
    Check whether a binary target has enough rows and both classes.
    """
    data = df.dropna(subset=[target_col]).copy()
    if len(data) < min_n:
        return False, f"not enough labelled rows: {len(data)} < {min_n}"
    if data[target_col].nunique() < 2:
        return False, f"fewer than two classes: {data[target_col].nunique()}"
    return True, "ok"


def choose_group_train_test_split(
    df,
    target_col,
    group_col=None,
    test_size=0.25,
    random_state=None,
    n_tries=100,
    max_rate_diff=0.20,
    verbose=True
):
    """
    Search for a grouped train/test split with both classes and acceptable class balance.

    Statistical annotation:
    GroupShuffleSplit is not stratified. With imbalanced targets, a single random split
    can produce a misleading test set. This function tries multiple seeds and chooses a
    split with closer train/test positive rates.
    """
    if group_col is None:
        group_col = GROUP_COL_FOR_VALIDATION
    if random_state is None:
        random_state = RANDOM_STATE

    data = df.dropna(subset=[target_col]).copy()
    data = data[data[group_col].notna()].copy()

    if data[target_col].nunique() < 2:
        raise ValueError(f"Target {target_col} has fewer than two classes.")
    if data[group_col].nunique() < 2:
        raise ValueError(f"Not enough groups in {group_col} for train/test splitting.")

    best = None
    best_score = np.inf

    for offset in range(n_tries):
        seed = random_state + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        groups = data[group_col]
        train_idx, test_idx = next(splitter.split(data, data[target_col], groups))

        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            continue

        train_rate = train_df[target_col].astype(int).mean()
        test_rate = test_df[target_col].astype(int).mean()
        rate_diff = abs(train_rate - test_rate)

        score = rate_diff + 1 / max(test_df[group_col].nunique(), 1)

        if score < best_score:
            best_score = score
            best = (train_df, test_df, seed, train_rate, test_rate, rate_diff)

        if rate_diff <= max_rate_diff:
            break

    if best is None:
        raise ValueError(
            f"Could not find a valid grouped split for {target_col}. "
            "Try more data, a different group column, or a different RANDOM_STATE."
        )

    train_df, test_df, chosen_seed, train_rate, test_rate, rate_diff = best

    if verbose:
        print(
            f"Grouped split for {target_col}: seed={chosen_seed}, "
            f"train positive rate={train_rate:.3f}, test positive rate={test_rate:.3f}, "
            f"absolute difference={rate_diff:.3f}"
        )

    return train_df, test_df


def metric_value(y_true, y_pred, y_score=None, metric_name="balanced_accuracy"):
    """
    Compute a single validation-selection metric.
    """
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric_name == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric_name == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric_name == "roc_auc" and y_score is not None and len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_score)
    raise ValueError(f"Unsupported metric for threshold optimisation: {metric_name}")


def tune_threshold(y_true, y_score, metric_name="balanced_accuracy", thresholds=None):
    """
    Choose a probability threshold on validation data only.

    Statistical annotation:
    This improves classification accuracy/F1 without contaminating the independent test set.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "score": metric_value(y_true, y_pred, y_score, metric_name),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0)
        })

    out = pd.DataFrame(rows)
    best = out.sort_values(["score", "balanced_accuracy", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), float(best["score"]), out


def safe_binary_metrics(y_true, y_pred, y_score=None):
    """
    Compute binary classification metrics robustly.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = np.nan

        try:
            metrics["average_precision"] = average_precision_score(y_true, y_score)
        except Exception:
            metrics["average_precision"] = np.nan

        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_score)
        except Exception:
            metrics["brier_score"] = np.nan

    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
        metrics["brier_score"] = np.nan

    return metrics


def predict_with_threshold(model, X, threshold=0.50):
    """
    Predict class labels using a tuned probability threshold when possible.
    """
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
            y_pred = (y_score >= threshold).astype(int)
            return y_pred, y_score
        except Exception:
            pass

    y_pred = model.predict(X)
    return y_pred, None


def fit_model_with_validation_tuning(
    train_valid_df,
    target_col,
    numeric_features,
    categorical_features,
    model_type,
    group_col=None,
    validation_size=None,
    random_state=None
):
    """
    Fit model using inner validation for hyperparameter and threshold selection.

    Workflow:
    1. Split train_valid into train and validation groups.
    2. Fit candidate models on train.
    3. Select hyperparameters and threshold on validation.
    4. Refit selected model on all train_valid rows.
    """
    if group_col is None:
        group_col = GROUP_COL_FOR_VALIDATION
    if validation_size is None:
        validation_size = VALIDATION_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    features = numeric_features + categorical_features
    assert_no_target_leakage(target_col, features)

    # Try to create a validation split. If not possible, fall back to no tuning.
    try:
        train_df, valid_df = choose_group_train_test_split(
            train_valid_df,
            target_col=target_col,
            group_col=group_col,
            test_size=validation_size,
            random_state=random_state + 1000,
            n_tries=50,
            max_rate_diff=0.25,
            verbose=False
        )
        use_validation = True
    except Exception as e:
        print("Validation split unavailable; using default parameters/threshold.")
        print("Reason:", e)
        train_df = train_valid_df.copy()
        valid_df = None
        use_validation = False

    X_train = train_df[features]
    y_train = train_df[target_col].astype(int)

    if use_validation:
        X_valid = valid_df[features]
        y_valid = valid_df[target_col].astype(int)
    else:
        X_valid = None
        y_valid = None

    candidate_params = candidate_parameter_grid(model_type)

    best = {
        "score": -np.inf,
        "params": {},
        "threshold": 0.50,
        "validation_score": np.nan,
        "validation_threshold_table": pd.DataFrame()
    }

    for params in candidate_params:
        candidate = make_classifier_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_type
        )
        if params:
            candidate.set_params(**params)

        candidate.fit(X_train, y_train)

        if use_validation:
            y_valid_pred_default, y_valid_score = predict_with_threshold(candidate, X_valid, threshold=0.50)

            if (
                TUNE_CLASSIFICATION_THRESHOLD
                and y_valid_score is not None
                and model_type != "dummy"
            ):
                threshold, score, threshold_table = tune_threshold(
                    y_true=y_valid,
                    y_score=y_valid_score,
                    metric_name=THRESHOLD_OPTIMIZATION_METRIC
                )
                y_valid_pred = (y_valid_score >= threshold).astype(int)
            else:
                threshold = 0.50
                threshold_table = pd.DataFrame()
                score = metric_value(
                    y_true=y_valid,
                    y_pred=y_valid_pred_default,
                    y_score=y_valid_score,
                    metric_name=THRESHOLD_OPTIMIZATION_METRIC
                )
                y_valid_pred = y_valid_pred_default

            # Tie-breakers favour simpler default threshold when performance is identical.
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "params": params,
                    "threshold": float(threshold),
                    "validation_score": float(score),
                    "validation_threshold_table": threshold_table
                }
        else:
            # No validation split: use defaults.
            best = {
                "score": np.nan,
                "params": params,
                "threshold": 0.50,
                "validation_score": np.nan,
                "validation_threshold_table": pd.DataFrame()
            }
            break

    # Refit final model on all train_valid rows with selected hyperparameters.
    final_model = make_classifier_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type
    )

    if best["params"]:
        final_model.set_params(**best["params"])

    X_train_valid = train_valid_df[features]
    y_train_valid = train_valid_df[target_col].astype(int)

    final_model.fit(X_train_valid, y_train_valid)

    tuning_info = {
        "selected_params": best["params"],
        "decision_threshold": best["threshold"],
        "validation_score": best["validation_score"],
        "n_validation": int(len(valid_df)) if use_validation else 0,
        "hyperparameter_tuned": bool(TUNE_MODEL_HYPERPARAMETERS and model_type != "dummy"),
        "threshold_tuned": bool(TUNE_CLASSIFICATION_THRESHOLD and model_type != "dummy" and use_validation),
        "threshold_optimization_metric": THRESHOLD_OPTIMIZATION_METRIC,
        "validation_threshold_table": best["validation_threshold_table"]
    }

    return final_model, tuning_info


def fit_and_evaluate_model(
    df,
    target_col,
    numeric_features,
    categorical_features,
    model_type="logreg",
    model_name="model",
    group_col=None,
    test_size=0.25,
    random_state=None,
    show_plots=True
):
    """
    Fit and evaluate a binary classifier with grouped train/test split and validation-only tuning.

    Statistical annotation:
    The reported test metrics are computed on held-out groups that are not used for either
    fitting or tuning. This keeps the final evaluation more honest.
    """
    if group_col is None:
        group_col = GROUP_COL_FOR_VALIDATION
    if random_state is None:
        random_state = RANDOM_STATE

    trainable, reason = is_trainable_binary_target(df, target_col)
    if not trainable:
        print(f"Model skipped for {target_col}: {reason}")
        return None, None, None

    features = numeric_features + categorical_features
    assert_no_target_leakage(target_col, features)

    train_valid_df, test_df = choose_group_train_test_split(
        df=df,
        target_col=target_col,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state
    )

    model, tuning_info = fit_model_with_validation_tuning(
        train_valid_df=train_valid_df,
        target_col=target_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        group_col=group_col,
        validation_size=VALIDATION_SIZE,
        random_state=random_state
    )

    X_test = test_df[features]
    y_test = test_df[target_col].astype(int)

    y_pred, y_score = predict_with_threshold(
        model,
        X_test,
        threshold=tuning_info["decision_threshold"]
    )

    metrics = safe_binary_metrics(y_test, y_pred, y_score)
    metrics["model"] = model_name
    metrics["target"] = target_col
    metrics["model_type"] = model_type
    metrics["group_col"] = group_col
    metrics["n_train"] = len(train_valid_df)
    metrics["n_test"] = len(test_df)
    metrics["n_train_groups"] = train_valid_df[group_col].nunique()
    metrics["n_test_groups"] = test_df[group_col].nunique()
    metrics["positive_rate_train"] = train_valid_df[target_col].astype(int).mean()
    metrics["positive_rate_test"] = y_test.mean()
    metrics["decision_threshold"] = tuning_info["decision_threshold"]
    metrics["validation_score"] = tuning_info["validation_score"]
    metrics["n_validation"] = tuning_info["n_validation"]
    metrics["selected_params"] = str(tuning_info["selected_params"])
    metrics["hyperparameter_tuned"] = tuning_info["hyperparameter_tuned"]
    metrics["threshold_tuned"] = tuning_info["threshold_tuned"]
    metrics["threshold_optimization_metric"] = tuning_info["threshold_optimization_metric"]

    print(f"\n=== {model_name} ===")
    print("Statistical interpretation:")
    print("- Test metrics below are evaluated on held-out groups only.")
    print("- decision_threshold was selected on validation data, not test data.")
    print("- balanced_accuracy is prioritised because high-load labels are usually imbalanced.")
    display(pd.DataFrame([metrics]))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if show_plots:
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title(f"Confusion matrix: {model_name}")
        plt.show()

        if y_score is not None and len(np.unique(y_test)) == 2:
            RocCurveDisplay.from_predictions(y_test, y_score)
            plt.title(f"ROC curve: {model_name}")
            plt.show()

            PrecisionRecallDisplay.from_predictions(y_test, y_score)
            plt.title(f"Precision-recall curve: {model_name}")
            plt.show()

    pred_df = test_df.copy()
    pred_df[f"pred_{target_col}"] = y_pred
    pred_df[f"threshold_{target_col}"] = tuning_info["decision_threshold"]

    if y_score is not None:
        pred_df[f"prob_{target_col}"] = y_score

    # Attach tuning diagnostics for optional later inspection.
    pred_df.attrs["tuning_info"] = tuning_info

    return model, metrics, pred_df

# %% [cell 41]
skip_df = df_model.copy()

print("Target distribution:")
display(skip_df["target_skipped"].value_counts(normalize=True).rename("proportion"))

skip_dummy_model, skip_dummy_metrics, skip_dummy_pred = fit_and_evaluate_model(
    df=skip_df,
    target_col="target_skipped",
    numeric_features=LEXICAL_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="dummy",
    model_name="Model A0: skipped-word dummy baseline",
    show_plots=False
)

skip_logreg_model, skip_logreg_metrics, skip_logreg_pred = fit_and_evaluate_model(
    df=skip_df,
    target_col="target_skipped",
    numeric_features=LEXICAL_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="logreg",
    model_name="Model A1: skipped-word lexical logistic regression"
)

skip_rf_model, skip_rf_metrics, skip_rf_pred = fit_and_evaluate_model(
    df=skip_df,
    target_col="target_skipped",
    numeric_features=LEXICAL_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="rf",
    model_name="Model A2: skipped-word lexical random forest"
)

# %% [cell 44]
late_df = df_model[
    (df_model["is_fixated"] == 1)
    & df_model["target_high_late_load"].notna()
].copy()

late_df["target_high_late_load"] = late_df["target_high_late_load"].astype(int)

print("Late-load modelling rows:", late_df.shape)
display(late_df["target_high_late_load"].value_counts(normalize=True).rename("proportion"))

display(late_df[[
    "subject_id", "task", "sentence_id", "word",
    "word_length", "FFD", "GD", "TRT", "late_time",
    "target_high_late_load"
]].head(20))

# %% [cell 46]
late_dummy_model, late_dummy_metrics, late_dummy_pred = fit_and_evaluate_model(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=LEXICAL_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="dummy",
    model_name="Model B0: high-late-load dummy baseline",
    show_plots=False
)

# %% [cell 48]
late_lex_model, late_lex_metrics, late_lex_pred = fit_and_evaluate_model(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=LEXICAL_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="logreg",
    model_name="Model B1: high-late-load lexical baseline"
)

# %% [cell 50]
late_early_model, late_early_metrics, late_early_pred = fit_and_evaluate_model(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=EARLY_EYE_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="logreg",
    model_name="Model B2: high-late-load lexical + FFD model"
)

# %% [cell 52]
def threshold_sensitivity_table(pred_df, target_col, prob_col, thresholds=None):
    """
    Evaluate classification metrics across probability thresholds.

    Statistical annotation:
    This is a diagnostic table for the held-out test set. It should not be used to choose
    the final threshold, because that would leak test information into the model decision rule.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    if pred_df is None or prob_col not in pred_df.columns:
        return pd.DataFrame()

    y_true = pred_df[target_col].astype(int).values
    y_score = pred_df[prob_col].values

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        })

    return pd.DataFrame(rows)


if late_early_pred is None:
    print("Threshold analysis skipped because Model B2 was not fitted.")
else:
    prob_col = "prob_target_high_late_load"
    threshold_df = threshold_sensitivity_table(
        late_early_pred,
        target_col="target_high_late_load",
        prob_col=prob_col
    )

    chosen_threshold = None
    if isinstance(late_early_metrics, dict):
        chosen_threshold = late_early_metrics.get("decision_threshold", None)

    print("Validation-selected threshold used by Model B2:", chosen_threshold)

    if threshold_df.empty:
        print("Threshold analysis skipped because predicted probabilities are unavailable.")
    else:
        display(threshold_df.sort_values("balanced_accuracy", ascending=False).head(10))

        plt.figure(figsize=(7, 4))
        plt.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], marker="o", label="balanced accuracy")
        plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")

        if chosen_threshold is not None:
            plt.axvline(chosen_threshold, linestyle="--", label="validation-selected threshold")

        plt.xlabel("Classification threshold")
        plt.ylabel("Metric value")
        plt.title("Diagnostic threshold sensitivity: Model B2 test set")
        plt.legend()
        plt.show()

# %% [cell 54]
assert_no_target_leakage(
    "target_high_late_load",
    CONFIRMATORY_FFD_INTERACTION_NUMERIC + LEXICAL_CATEGORICAL
)

late_firstpass_model, late_firstpass_metrics, late_firstpass_pred = fit_and_evaluate_model(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=CONFIRMATORY_FFD_INTERACTION_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="logreg",
    model_name="Model B3: high-late-load moderated FFD logistic model"
)

assert_no_target_leakage(
    "target_high_late_load",
    CONFIRMATORY_FFD_NONLINEAR_NUMERIC + LEXICAL_CATEGORICAL
)

late_rf_model, late_rf_metrics, late_rf_pred = fit_and_evaluate_model(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=CONFIRMATORY_FFD_NONLINEAR_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="rf",
    model_name="Model B4: high-late-load nonlinear FFD robustness model"
)

# %% [cell 56]
metric_objects = [
    skip_dummy_metrics,
    skip_logreg_metrics,
    skip_rf_metrics,
    late_dummy_metrics,
    late_lex_metrics,
    late_early_metrics,
    late_firstpass_metrics,
    late_rf_metrics
]

metric_objects = [m for m in metric_objects if isinstance(m, dict)]

B_MODEL_METADATA = pd.DataFrame([
    {
        "model": "Model B0: high-late-load dummy baseline",
        "model_code": "B0",
        "evidence_order": 0,
        "evidence_role": "no-signal reference",
        "research_question_role": "Can any real model beat trivial prediction?",
        "feature_family": "none / majority class"
    },
    {
        "model": "Model B1: high-late-load lexical baseline",
        "model_code": "B1",
        "evidence_order": 1,
        "evidence_role": "lexical/contextual baseline",
        "research_question_role": "How much late load is predictable without eye-tracking?",
        "feature_family": "lexical/contextual"
    },
    {
        "model": "Model B2: high-late-load lexical + FFD model",
        "model_code": "B2",
        "evidence_order": 2,
        "evidence_role": "additive FFD test",
        "research_question_role": "Does FFD add predictive value beyond B1?",
        "feature_family": "lexical/contextual + FFD"
    },
    {
        "model": "Model B3: high-late-load moderated FFD logistic model",
        "model_code": "B3",
        "evidence_order": 3,
        "evidence_role": "context-moderated FFD robustness",
        "research_question_role": "Does the FFD signal remain useful when context interactions are allowed?",
        "feature_family": "lexical/contextual + FFD + FFD×context"
    },
    {
        "model": "Model B4: high-late-load nonlinear FFD robustness model",
        "model_code": "B4",
        "evidence_order": 4,
        "evidence_role": "nonlinear FFD robustness",
        "research_question_role": "Does the FFD signal remain useful under a nonlinear classifier?",
        "feature_family": "lexical/contextual + FFD, nonlinear"
    },
])

if len(metric_objects) == 0:
    print("No model metrics were produced. Check target distributions and file availability.")
    all_metrics = pd.DataFrame()
    b_metrics = pd.DataFrame()
else:
    all_metrics = pd.DataFrame(metric_objects)

    metric_order = [
        "model", "target", "model_type", "group_col",
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1", "roc_auc", "average_precision", "brier_score",
        "decision_threshold", "validation_score", "n_validation",
        "selected_params", "hyperparameter_tuned", "threshold_tuned",
        "n_train", "n_test", "n_train_groups", "n_test_groups",
        "positive_rate_train", "positive_rate_test"
    ]

    all_metrics = all_metrics[[c for c in metric_order if c in all_metrics.columns]]
    all_metrics = all_metrics.merge(B_MODEL_METADATA, on="model", how="left")

    b_metrics = (
        all_metrics[all_metrics["target"].eq("target_high_late_load")]
        .sort_values("evidence_order", na_position="last")
        .reset_index(drop=True)
    )

    print("Unified B-series evidence table:")
    display_cols = [
        "model_code", "evidence_role", "feature_family",
        "balanced_accuracy", "f1", "roc_auc", "average_precision", "brier_score",
        "decision_threshold", "validation_score", "precision", "recall",
        "n_validation", "n_test", "positive_rate_test"
    ]
    display(b_metrics[[c for c in display_cols if c in b_metrics.columns]])

    print("All model metrics, including skipping models:")
    display(all_metrics.sort_values(["target", "balanced_accuracy"], ascending=[True, False]))

    model_comparison_path = OUTPUT_DIR / "model_comparison_metrics_all.csv"
    all_metrics.to_csv(model_comparison_path, index=False)
    print("Saved:", model_comparison_path)

    b_metrics_path = OUTPUT_DIR / "model_B_series_aligned_metrics.csv"
    b_metrics.to_csv(b_metrics_path, index=False)
    print("Saved:", b_metrics_path)

# %% [cell 58]
if "b_metrics" not in globals() or b_metrics.empty:
    print("No B-series metrics available yet.")
else:
    diagnostic_cols = [
        "model_code", "evidence_role", "decision_threshold", "validation_score",
        "selected_params", "hyperparameter_tuned", "threshold_tuned",
        "balanced_accuracy", "f1", "roc_auc", "average_precision", "brier_score"
    ]
    diagnostic_cols = [c for c in diagnostic_cols if c in b_metrics.columns]

    print("Accuracy optimisation diagnostics for B-series models:")
    display(b_metrics[diagnostic_cols])

    print("\nStraight statistical reading:")
    print("- Higher balanced_accuracy means better class-balanced test performance.")
    print("- Higher F1 means better positive-class detection under the chosen threshold.")
    print("- Higher ROC-AUC means better probability ranking, independent of a single threshold.")
    print("- Lower Brier score means better calibrated probabilities.")
    print("- Compare B2/B3/B4 against B1, not only against B0.")

# %% [cell 60]
def mcnemar_test_from_predictions(y_true, pred_a, pred_b):
    """
    Approximate McNemar test with continuity correction.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    b = np.sum(a_correct & ~b_correct)
    c = np.sum(~a_correct & b_correct)

    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "A_correct_B_wrong": int(b),
        "A_wrong_B_correct": int(c),
        "mcnemar_chi2": chi2,
        "p_value": p_value
    }


B_PREDICTION_REGISTRY = {
    "B1": {
        "label": "lexical baseline",
        "pred_df": late_lex_pred
    },
    "B2": {
        "label": "additive FFD",
        "pred_df": late_early_pred
    },
    "B3": {
        "label": "moderated FFD",
        "pred_df": late_firstpass_pred
    },
    "B4": {
        "label": "nonlinear FFD",
        "pred_df": late_rf_pred
    },
}

B_PAIRWISE_COMPARISONS = [
    ("B1", "B2"),
    ("B1", "B3"),
    ("B1", "B4"),
    ("B2", "B3"),
    ("B2", "B4"),
]


def align_two_prediction_tables(model_a_key, model_b_key):
    """
    Align two held-out prediction tables on the same word rows.
    """
    pred_a = B_PREDICTION_REGISTRY[model_a_key]["pred_df"]
    pred_b = B_PREDICTION_REGISTRY[model_b_key]["pred_df"]

    if pred_a is None or pred_b is None:
        return pd.DataFrame()

    target_col = "target_high_late_load"
    pred_col = "pred_target_high_late_load"

    id_cols = ["subject_id", "task", "sentence_id", "word_id", GROUP_COL_FOR_VALIDATION]
    id_cols = [c for c in dict.fromkeys(id_cols) if c in pred_a.columns and c in pred_b.columns]

    a = pred_a[id_cols + [target_col, pred_col]].copy()
    a = a.rename(columns={
        target_col: "target",
        pred_col: f"pred_{model_a_key}"
    })

    b = pred_b[id_cols + [pred_col]].copy()
    b = b.rename(columns={
        pred_col: f"pred_{model_b_key}"
    })

    aligned = a.merge(b, on=id_cols, how="inner")
    aligned["comparison"] = f"{model_b_key} minus {model_a_key}"
    aligned["model_A"] = model_a_key
    aligned["model_B"] = model_b_key

    return aligned


mcnemar_rows = []

for model_a, model_b in B_PAIRWISE_COMPARISONS:
    aligned = align_two_prediction_tables(model_a, model_b)

    if aligned.empty:
        print(f"McNemar skipped for {model_a} vs {model_b}: predictions unavailable or not alignable.")
        continue

    result = mcnemar_test_from_predictions(
        y_true=aligned["target"],
        pred_a=aligned[f"pred_{model_a}"],
        pred_b=aligned[f"pred_{model_b}"]
    )
    result.update({
        "comparison": f"{model_b} minus {model_a}",
        "model_A": model_a,
        "model_B": model_b,
        "n_aligned_rows": len(aligned)
    })
    mcnemar_rows.append(result)

mcnemar_df = pd.DataFrame(mcnemar_rows)

if mcnemar_df.empty:
    print("No McNemar comparisons were produced.")
else:
    display(mcnemar_df)
    mcnemar_path = OUTPUT_DIR / "model_B_series_mcnemar_comparisons.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print("Saved:", mcnemar_path)

# %% [cell 62]
def grouped_bootstrap_metric_difference(
    aligned_pred_df,
    target_col,
    pred_a_col,
    pred_b_col,
    group_col,
    metric_func=balanced_accuracy_score,
    n_boot=1000,
    random_state=RANDOM_STATE
):
    """
    Paired grouped bootstrap for metric difference B - A.
    """
    if aligned_pred_df is None or aligned_pred_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    groups = aligned_pred_df[group_col].dropna().unique()

    if len(groups) < 2:
        return pd.DataFrame()

    deltas = []

    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([
            aligned_pred_df[aligned_pred_df[group_col] == g]
            for g in sampled_groups
        ], ignore_index=True)

        y_true = sample[target_col].astype(int).values
        pred_a = sample[pred_a_col].astype(int).values
        pred_b = sample[pred_b_col].astype(int).values

        if len(np.unique(y_true)) < 2:
            continue

        delta = metric_func(y_true, pred_b) - metric_func(y_true, pred_a)
        deltas.append(delta)

    if len(deltas) == 0:
        return pd.DataFrame()

    deltas = np.asarray(deltas)

    return pd.DataFrame([{
        "metric": getattr(metric_func, "__name__", "metric"),
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_delta_le_0": float(np.mean(deltas <= 0)),
        "n_boot_valid": len(deltas),
        "group_col": group_col
    }])


bootstrap_comparison_rows = []

for model_a, model_b in B_PAIRWISE_COMPARISONS:
    aligned = align_two_prediction_tables(model_a, model_b)

    if aligned.empty or GROUP_COL_FOR_VALIDATION not in aligned.columns:
        print(f"Bootstrap skipped for {model_a} vs {model_b}: predictions unavailable or group column missing.")
        continue

    boot = grouped_bootstrap_metric_difference(
        aligned_pred_df=aligned,
        target_col="target",
        pred_a_col=f"pred_{model_a}",
        pred_b_col=f"pred_{model_b}",
        group_col=GROUP_COL_FOR_VALIDATION,
        metric_func=balanced_accuracy_score,
        n_boot=1000
    )

    if boot.empty:
        continue

    boot["comparison"] = f"{model_b} minus {model_a}"
    boot["model_A"] = model_a
    boot["model_B"] = model_b
    boot["n_aligned_rows"] = len(aligned)
    bootstrap_comparison_rows.append(boot)

paired_bootstrap_delta_df = (
    pd.concat(bootstrap_comparison_rows, ignore_index=True)
    if bootstrap_comparison_rows else pd.DataFrame()
)

if paired_bootstrap_delta_df.empty:
    print("No paired grouped bootstrap comparisons were produced.")
else:
    display(paired_bootstrap_delta_df[[
        "comparison", "metric", "mean_delta", "ci_low", "ci_high",
        "p_delta_le_0", "n_boot_valid", "group_col", "n_aligned_rows"
    ]])

    paired_bootstrap_path = OUTPUT_DIR / "model_B_series_paired_grouped_bootstrap.csv"
    paired_bootstrap_delta_df.to_csv(paired_bootstrap_path, index=False)
    print("Saved:", paired_bootstrap_path)

# %% [cell 64]
def get_feature_names_from_pipeline(pipeline):
    """
    Extract final feature names after preprocessing.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            try:
                onehot = transformer.named_steps["onehot"]
                feature_names.extend(onehot.get_feature_names_out(columns))
            except Exception:
                feature_names.extend(columns)

    return np.array(feature_names)


def logistic_coefficient_table(pipeline, model_code=None, model_label=None):
    """
    Return coefficients for a fitted logistic-regression pipeline.
    """
    if pipeline is None:
        return pd.DataFrame()

    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        return pd.DataFrame()

    feature_names = get_feature_names_from_pipeline(pipeline)
    coefs = classifier.coef_[0]

    out = (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs)
        })
        .sort_values("abs_coefficient", ascending=False)
    )

    if model_code is not None:
        out.insert(0, "model_code", model_code)
    if model_label is not None:
        out.insert(1, "model_label", model_label)

    return out


coefficient_tables = []

for model_code, model_label, model_obj in [
    ("B1", "lexical baseline", late_lex_model),
    ("B2", "additive FFD", late_early_model),
    ("B3", "moderated FFD", late_firstpass_model),
]:
    coef_table = logistic_coefficient_table(model_obj, model_code=model_code, model_label=model_label)
    if not coef_table.empty:
        coefficient_tables.append(coef_table)

if not coefficient_tables:
    print("Coefficient tables skipped because no fitted logistic B models are available.")
    b_series_coef_df = pd.DataFrame()
else:
    b_series_coef_df = pd.concat(coefficient_tables, ignore_index=True)

    print("Top coefficients by B-series logistic model:")
    display(
        b_series_coef_df
        .sort_values(["model_code", "abs_coefficient"], ascending=[True, False])
        .groupby("model_code")
        .head(15)
        .reset_index(drop=True)
    )

    coef_path = OUTPUT_DIR / "model_B_series_logistic_coefficients.csv"
    b_series_coef_df.to_csv(coef_path, index=False)
    print("Saved:", coef_path)

# %% [cell 65]
PERMUTATION_MODEL_SPECS = [
    ("B1", "lexical baseline", late_lex_model, LEXICAL_NUMERIC, LEXICAL_CATEGORICAL),
    ("B2", "additive FFD", late_early_model, EARLY_EYE_NUMERIC, LEXICAL_CATEGORICAL),
    ("B3", "moderated FFD", late_firstpass_model, CONFIRMATORY_FFD_INTERACTION_NUMERIC, LEXICAL_CATEGORICAL),
    ("B4", "nonlinear FFD", late_rf_model, CONFIRMATORY_FFD_NONLINEAR_NUMERIC, LEXICAL_CATEGORICAL),
]

perm_tables = []

for model_code, model_label, model_obj, numeric_features, categorical_features in PERMUTATION_MODEL_SPECS:
    if model_obj is None:
        print(f"Permutation importance skipped for {model_code}: model not fitted.")
        continue

    train_tmp, test_tmp = choose_group_train_test_split(
        df=late_df,
        target_col="target_high_late_load",
        group_col=GROUP_COL_FOR_VALIDATION,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    features = numeric_features + categorical_features
    X_test_tmp = test_tmp[features]
    y_test_tmp = test_tmp["target_high_late_load"].astype(int)

    perm = permutation_importance(
        model_obj,
        X_test_tmp,
        y_test_tmp,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy"
    )

    perm_df_tmp = pd.DataFrame({
        "model_code": model_code,
        "model_label": model_label,
        "feature": X_test_tmp.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values(["model_code", "importance_mean"], ascending=[True, False])

    perm_tables.append(perm_df_tmp)

if not perm_tables:
    print("No permutation-importance tables were produced.")
    perm_df = pd.DataFrame()
else:
    perm_df = pd.concat(perm_tables, ignore_index=True)
    display(perm_df.sort_values(["model_code", "importance_mean"], ascending=[True, False]))

    # Plot top features from B2-B4 because those are the FFD-bearing models.
    top_perm = (
        perm_df[perm_df["model_code"].isin(["B2", "B3", "B4"])]
        .sort_values(["model_code", "importance_mean"], ascending=[True, False])
        .groupby("model_code")
        .head(10)
    )

    for model_code, group in top_perm.groupby("model_code"):
        plt.figure(figsize=(8, 4))
        group_sorted = group.sort_values("importance_mean", ascending=True)
        plt.barh(group_sorted["feature"], group_sorted["importance_mean"], xerr=group_sorted["importance_std"])
        plt.xlabel("Mean decrease in balanced accuracy")
        plt.ylabel("Feature")
        plt.title(f"Permutation importance: {model_code}")
        plt.show()

    perm_path = OUTPUT_DIR / "model_B_series_permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    print("Saved:", perm_path)

# %% [cell 67]
def add_error_labels(pred_df, model_code):
    """
    Add error labels to a prediction table for target_high_late_load.
    """
    if pred_df is None:
        return pd.DataFrame()

    pred_col = "pred_target_high_late_load"
    prob_col = "prob_target_high_late_load"

    out = pred_df.copy()
    out["model_code"] = model_code
    out["error_type"] = "correct"
    out.loc[(out["target_high_late_load"] == 0) & (out[pred_col] == 1), "error_type"] = "false_positive"
    out.loc[(out["target_high_late_load"] == 1) & (out[pred_col] == 0), "error_type"] = "false_negative"

    if prob_col not in out.columns:
        out[prob_col] = np.nan

    return out


error_tables = []

for model_code, registry in B_PREDICTION_REGISTRY.items():
    tmp = add_error_labels(registry["pred_df"], model_code)
    if not tmp.empty:
        error_tables.append(tmp)

if not error_tables:
    print("Error analysis skipped because no B-series predictions are available.")
    error_df = pd.DataFrame()
else:
    error_df = pd.concat(error_tables, ignore_index=True)

    print("Error counts by model:")
    display(
        error_df
        .groupby(["model_code", "error_type"])
        .size()
        .rename("n")
        .reset_index()
    )

    print("Error rates by model and task:")
    display(
        error_df
        .assign(is_error=lambda d: (d["error_type"] != "correct").astype(int))
        .groupby(["model_code", "task"])
        .agg(
            n=("word", "count"),
            error_rate=("is_error", "mean"),
            false_negative_rate=("error_type", lambda x: np.mean(x == "false_negative")),
            false_positive_rate=("error_type", lambda x: np.mean(x == "false_positive")),
            mean_word_length=("word_length", "mean"),
            mean_FFD=("FFD", "mean"),
            mean_late_time=("late_time", "mean")
        )
        .reset_index()
    )

    inspect_cols = [
        "model_code", "subject_id", "task", "sentence_id", "word_id", "word",
        "word_length", "FFD", "GD", "TRT", "late_time",
        "target_high_late_load", "pred_target_high_late_load",
        "prob_target_high_late_load", "error_type"
    ]
    inspect_cols = [c for c in inspect_cols if c in error_df.columns]

    print("Theory-primary B2 false negatives:")
    display(
        error_df[(error_df["model_code"] == "B2") & (error_df["error_type"] == "false_negative")]
        [inspect_cols]
        .sort_values("prob_target_high_late_load", ascending=True)
        .head(20)
    )

    print("Theory-primary B2 false positives:")
    display(
        error_df[(error_df["model_code"] == "B2") & (error_df["error_type"] == "false_positive")]
        [inspect_cols]
        .sort_values("prob_target_high_late_load", ascending=False)
        .head(20)
    )

    error_path = OUTPUT_DIR / "model_B_series_error_analysis.csv"
    error_df.to_csv(error_path, index=False)
    print("Saved:", error_path)

# %% [cell 69]
def grouped_cross_validation(
    df,
    target_col,
    numeric_features,
    categorical_features,
    model_type="logreg",
    group_col=None,
    max_splits=5
):
    """
    Grouped cross-validation with inner validation tuning.

    For each outer fold:
    - train/validation tuning happens only within the fold's training data
    - the fold test data remains unseen until final fold evaluation
    """
    if group_col is None:
        group_col = GROUP_COL_FOR_VALIDATION

    trainable, reason = is_trainable_binary_target(df, target_col)
    if not trainable:
        print(f"Cross-validation skipped for {target_col}: {reason}")
        return pd.DataFrame()

    data = df.dropna(subset=[target_col]).copy()
    data = data[data[group_col].notna()].copy()

    features = numeric_features + categorical_features
    assert_no_target_leakage(target_col, features)

    y = data[target_col].astype(int)
    groups = data[group_col]

    n_groups = groups.nunique()
    n_splits = min(max_splits, n_groups)

    if n_splits < 2:
        print("Cross-validation skipped: not enough groups.")
        return pd.DataFrame()

    cv = GroupKFold(n_splits=n_splits)

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data, y, groups), start=1):
        train_valid_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        if train_valid_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            continue

        model, tuning_info = fit_model_with_validation_tuning(
            train_valid_df=train_valid_df,
            target_col=target_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_type,
            group_col=group_col,
            validation_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE + fold_idx
        )

        X_test = test_df[features]
        y_test = test_df[target_col].astype(int)

        y_pred, y_score = predict_with_threshold(
            model,
            X_test,
            threshold=tuning_info["decision_threshold"]
        )

        metrics = safe_binary_metrics(y_test, y_pred, y_score)
        metrics.update({
            "fold": fold_idx,
            "n_test": len(test_df),
            "n_test_groups": test_df[group_col].nunique(),
            "positive_rate_test": y_test.mean(),
            "decision_threshold": tuning_info["decision_threshold"],
            "validation_score": tuning_info["validation_score"],
            "selected_params": str(tuning_info["selected_params"]),
            "model_type": model_type,
            "target": target_col,
            "group_col": group_col
        })
        fold_rows.append(metrics)

    if not fold_rows:
        return pd.DataFrame()

    fold_df = pd.DataFrame(fold_rows)

    summary_rows = []
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score"]

    for metric in metric_names:
        if metric in fold_df.columns:
            summary_rows.append({
                "metric": metric,
                "mean": fold_df[metric].mean(),
                "std": fold_df[metric].std(ddof=1),
                "n_splits": len(fold_df),
                "group_col": group_col,
                "model_type": model_type,
                "target": target_col
            })

    summary = pd.DataFrame(summary_rows)
    summary.attrs["fold_metrics"] = fold_df
    return summary


CV_B_MODEL_SPECS = [
    ("B1", "lexical baseline", LEXICAL_NUMERIC, LEXICAL_CATEGORICAL, "logreg"),
    ("B2", "additive FFD", EARLY_EYE_NUMERIC, LEXICAL_CATEGORICAL, "logreg"),
    ("B3", "moderated FFD", CONFIRMATORY_FFD_INTERACTION_NUMERIC, LEXICAL_CATEGORICAL, "logreg"),
    ("B4", "nonlinear FFD", CONFIRMATORY_FFD_NONLINEAR_NUMERIC, LEXICAL_CATEGORICAL, "rf"),
]

cv_tables = []
cv_fold_tables = []

for model_code, model_label, numeric_features, categorical_features, model_type in CV_B_MODEL_SPECS:
    cv_tmp = grouped_cross_validation(
        df=late_df,
        target_col="target_high_late_load",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type
    )
    if not cv_tmp.empty:
        cv_tmp["model_code"] = model_code
        cv_tmp["model_label"] = model_label

        # Create a copy and clear attributes before appending to cv_tables
        # to avoid the ValueError during pd.concat due to DataFrame in attrs.
        summary_copy = cv_tmp.copy()
        summary_copy.attrs = {} # Clear attributes
        cv_tables.append(summary_copy)

        fold_df = cv_tmp.attrs.get("fold_metrics", pd.DataFrame())
        if not fold_df.empty:
            fold_df["model_code"] = model_code
            fold_df["model_label"] = model_label
            cv_fold_tables.append(fold_df)

cv_comparison = pd.concat(cv_tables, ignore_index=True) if cv_tables else pd.DataFrame()
cv_fold_metrics = pd.concat(cv_fold_tables, ignore_index=True) if cv_fold_tables else pd.DataFrame()

if cv_comparison.empty:
    print("No B-series cross-validation results produced.")
else:
    display(
        cv_comparison
        .sort_values(["metric", "model_code"])
        .reset_index(drop=True)
    )

    cv_path = OUTPUT_DIR / "cross_validation_B_series_inner_tuned_summary.csv"
    cv_comparison.to_csv(cv_path, index=False)
    print("Saved:", cv_path)

    if not cv_fold_metrics.empty:
        fold_path = OUTPUT_DIR / "cross_validation_B_series_inner_tuned_fold_metrics.csv"
        cv_fold_metrics.to_csv(fold_path, index=False)
        print("Saved fold-level CV metrics:", fold_path)

# %% [cell 71]
def bootstrap_metric_ci(y_true, y_pred, metric_func=balanced_accuracy_score, n_boot=1000, random_state=RANDOM_STATE):
    """
    Bootstrap confidence interval for a classification metric.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_func(y_true[idx], y_pred[idx]))

    if len(values) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot_valid": 0}

    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_boot_valid": len(values)
    }


bootstrap_rows = []

for model_code, registry in B_PREDICTION_REGISTRY.items():
    pred_df = registry["pred_df"]
    if pred_df is None or "pred_target_high_late_load" not in pred_df.columns:
        continue

    bootstrap_rows.append({
        "model_code": model_code,
        "model_label": registry["label"],
        **bootstrap_metric_ci(
            pred_df["target_high_late_load"],
            pred_df["pred_target_high_late_load"]
        )
    })

bootstrap_ci_df = pd.DataFrame(bootstrap_rows)

if bootstrap_ci_df.empty:
    print("Bootstrap CI skipped because model predictions are unavailable.")
else:
    display(bootstrap_ci_df.sort_values("model_code"))

    bootstrap_path = OUTPUT_DIR / "bootstrap_balanced_accuracy_ci_B_series.csv"
    bootstrap_ci_df.to_csv(bootstrap_path, index=False)
    print("Saved:", bootstrap_path)

# %% [cell 73]
if df_model["target_task_TSR"].notna().sum() == 0:
    print("Model C skipped: both NR and TSR are required.")
else:
    task_df = df_model[df_model["target_task_TSR"].notna()].copy()
    task_df["target_task_TSR"] = task_df["target_task_TSR"].astype(int)

    print("Task target distribution:")
    display(task_df["target_task_TSR"].value_counts(normalize=True).rename("proportion"))

    task_dummy_model, task_dummy_metrics, task_dummy_pred = fit_and_evaluate_model(
        df=task_df,
        target_col="target_task_TSR",
        numeric_features=TASK_CLASSIFICATION_NUMERIC,
        categorical_features=TASK_CLASSIFICATION_CATEGORICAL,
        model_type="dummy",
        model_name="Model C0: NR-vs-TSR dummy baseline",
        show_plots=False
    )

    task_logreg_model, task_logreg_metrics, task_logreg_pred = fit_and_evaluate_model(
        df=task_df,
        target_col="target_task_TSR",
        numeric_features=TASK_CLASSIFICATION_NUMERIC,
        categorical_features=TASK_CLASSIFICATION_CATEGORICAL,
        model_type="logreg",
        model_name="Model C1: NR-vs-TSR eye-tracking logistic regression"
    )

    task_rf_model, task_rf_metrics, task_rf_pred = fit_and_evaluate_model(
        df=task_df,
        target_col="target_task_TSR",
        numeric_features=TASK_CLASSIFICATION_NUMERIC,
        categorical_features=TASK_CLASSIFICATION_CATEGORICAL,
        model_type="rf",
        model_name="Model C2: NR-vs-TSR eye-tracking random forest"
    )

# %% [cell 75]
if df_model["target_task_TSR"].notna().sum() == 0:
    print("Sentence-level task model skipped: both NR and TSR are required.")
else:
    sentence_task_df = (
        df_model
        .groupby(["subject_id", "task", "sentence_uid", "sentence_hash"], dropna=False)
        .agg(
            target_task_TSR=("target_task_TSR", "first"),
            n_words=("word", "count"),
            mean_word_length=("word_length", "mean"),
            skipping_rate=("skipped", "mean"),
            mean_FFD=("FFD", "mean"),
            mean_GD=("GD", "mean"),
            mean_TRT=("TRT", "mean"),
            mean_GPT=("GPT", "mean"),
            mean_late_time=("late_time", "mean"),
            total_fixations=("nFixations", "sum")
        )
        .reset_index()
    )

    sentence_features = [
        "n_words", "mean_word_length", "skipping_rate", "mean_FFD", "mean_GD",
        "mean_TRT", "mean_GPT", "mean_late_time", "total_fixations"
    ]

    print("Sentence-level task dataset:", sentence_task_df.shape)
    display(sentence_task_df.head())

    sent_task_model, sent_task_metrics, sent_task_pred = fit_and_evaluate_model(
        df=sentence_task_df,
        target_col="target_task_TSR",
        numeric_features=sentence_features,
        categorical_features=[],
        model_type="logreg",
        model_name="Model C3: sentence-level NR-vs-TSR classifier",
        group_col="sentence_hash",
        show_plots=True
    )

# %% [cell 77]
def subject_held_out_cv(
    df,
    target_col,
    numeric_features,
    categorical_features,
    model_type="logreg",
    subject_col="subject_id"
):
    data = df.dropna(subset=[target_col]).copy()
    subjects = sorted(data[subject_col].dropna().unique())

    if len(subjects) < 2:
        print("Subject-held-out CV skipped: need at least two subjects.")
        return None

    rows = []

    for held_out in subjects:
        train_df = data[data[subject_col] != held_out].copy()
        test_df = data[data[subject_col] == held_out].copy()

        if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            print(f"Skipping {held_out}: insufficient class variation.")
            continue

        features = numeric_features + categorical_features

        X_train = train_df[features]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[features]
        y_test = test_df[target_col].astype(int)

        model = make_classifier_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_type
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_score = None
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]

        metrics = safe_binary_metrics(y_test, y_pred, y_score)
        metrics["held_out_subject"] = held_out
        metrics["n_train"] = len(train_df)
        metrics["n_test"] = len(test_df)

        rows.append(metrics)

    if not rows:
        return None

    return pd.DataFrame(rows)


subject_cv_results = subject_held_out_cv(
    df=late_df,
    target_col="target_high_late_load",
    numeric_features=EARLY_EYE_NUMERIC,
    categorical_features=LEXICAL_CATEGORICAL,
    model_type="logreg"
)

if subject_cv_results is not None:
    display(subject_cv_results)
    subject_cv_path = OUTPUT_DIR / "subject_held_out_cv_results.csv"
    subject_cv_results.to_csv(subject_cv_path, index=False)
    print("Saved:", subject_cv_path)

# %% [cell 79]
import json

cleaned_path = OUTPUT_DIR / "zuco_cleaned_eye_tracking_word_level.csv"
model_path = OUTPUT_DIR / "zuco_ml_ready_eye_tracking_word_level.csv"
manifest_path = OUTPUT_DIR / "reproducibility_manifest.json"

df.to_csv(cleaned_path, index=False)
df_model.to_csv(model_path, index=False)

print("Saved cleaned data:", cleaned_path)
print("Saved ML-ready data:", model_path)

B_PREDICTION_OUTPUTS = {
    "B1_lexical_baseline": late_lex_pred,
    "B2_additive_FFD": late_early_pred,
    "B3_moderated_FFD": late_firstpass_pred,
    "B4_nonlinear_FFD": late_rf_pred,
}

for name, pred_df in B_PREDICTION_OUTPUTS.items():
    if pred_df is not None:
        out_path = OUTPUT_DIR / f"model_{name}_high_late_load_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        print("Saved:", out_path)

if "error_df" in globals() and isinstance(error_df, pd.DataFrame) and not error_df.empty:
    error_path = OUTPUT_DIR / "model_B_series_error_analysis.csv"
    error_df.to_csv(error_path, index=False)
    print("Saved B-series error analysis:", error_path)

if "sent_task_pred" in globals() and sent_task_pred is not None:
    sent_task_path = OUTPUT_DIR / "sentence_level_task_predictions.csv"
    sent_task_pred.to_csv(sent_task_path, index=False)
    print("Saved sentence-level task predictions:", sent_task_path)

# Save extraction log when available.
if "log_df" in globals() and isinstance(log_df, pd.DataFrame) and not log_df.empty:
    log_path = OUTPUT_DIR / "zuco_extraction_log.csv"
    log_df.to_csv(log_path, index=False)
    print("Saved extraction log:", log_path)

manifest = {
    "DATA_DIR": str(DATA_DIR),
    "OUTPUT_DIR": str(OUTPUT_DIR),
    "OSF_ROOT_NODE_ID": OSF_ROOT_NODE_ID,
    "SUBJECTS": SUBJECTS,
    "TASKS": TASKS,
    "AUTO_DISCOVER_ALL_LOCAL_MAT_FILES": AUTO_DISCOVER_ALL_LOCAL_MAT_FILES,
    "USE_OSF_DISCOVERY": USE_OSF_DISCOVERY,
    "DOWNLOAD_FROM_OSF": DOWNLOAD_FROM_OSF,
    "GROUP_COL_FOR_VALIDATION": GROUP_COL_FOR_VALIDATION,
    "USE_TASK_AS_FEATURE": USE_TASK_AS_FEATURE,
    "RANDOM_STATE": RANDOM_STATE,
    "TUNE_MODEL_HYPERPARAMETERS": TUNE_MODEL_HYPERPARAMETERS,
    "TUNE_CLASSIFICATION_THRESHOLD": TUNE_CLASSIFICATION_THRESHOLD,
    "THRESHOLD_OPTIMIZATION_METRIC": THRESHOLD_OPTIMIZATION_METRIC,
    "VALIDATION_SIZE": VALIDATION_SIZE,
    "TUNING_MAX_CANDIDATES": TUNING_MAX_CANDIDATES,
    "n_rows_cleaned": int(len(df)),
    "n_rows_model": int(len(df_model)),
    "n_subjects": int(df_model["subject_id"].nunique()),
    "n_tasks": int(df_model["task"].nunique()),
    "n_source_files": int(df_model["source_file"].nunique()),
    "central_research_question": "Do FFD-based early fixation signals robustly improve prediction of high late processing load beyond lexical/contextual features?",
    "target": "target_high_late_load based on late_time = TRT - GD",
    "lexical_numeric_features_B1": LEXICAL_NUMERIC,
    "lexical_categorical_features": LEXICAL_CATEGORICAL,
    "additive_FFD_features_B2": EARLY_EYE_NUMERIC,
    "moderated_FFD_features_B3": CONFIRMATORY_FFD_INTERACTION_NUMERIC,
    "nonlinear_FFD_features_B4": CONFIRMATORY_FFD_NONLINEAR_NUMERIC,
    "forbidden_features_for_high_late_load": sorted(FORBIDDEN_FEATURES_BY_TARGET["target_high_late_load"]),
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Saved reproducibility manifest:", manifest_path)
