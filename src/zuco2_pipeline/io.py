from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def parse_subject_task_from_filename(path: str | Path) -> tuple[str | None, str | None]:
    match = re.match(r"results(?P<subject>[A-Za-z0-9]+)_(?P<task>[A-Za-z0-9]+)\.mat$", Path(path).name)
    if not match:
        return None, None
    return match.group("subject"), match.group("task")


def find_zuco_mat_file(data_dirs: Iterable[str | Path], subject: str, task: str) -> Path | None:
    exact_name = f"results{subject}_{task}.mat"
    dirs = [Path(item) for item in data_dirs]

    for data_dir in dirs:
        if data_dir.exists():
            matches = sorted(data_dir.rglob(exact_name))
            if matches:
                return matches[0]

    for data_dir in dirs:
        if data_dir.exists():
            matches = sorted(data_dir.rglob(f"*{subject}*{task}*.mat"))
            if matches:
                return matches[0]
    return None


def discover_local_zuco_files(data_dirs: Iterable[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()

    for data_dir in [Path(item) for item in data_dirs]:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("results*_*.mat")):
            if path in seen:
                continue
            subject, task = parse_subject_task_from_filename(path)
            if subject is None or task is None:
                continue
            rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "path": path,
                    "file_size_mb": path.stat().st_size / 1024 / 1024,
                    "discovery_mode": "auto",
                }
            )
            seen.add(path)

    return pd.DataFrame(rows)


def select_configured_files(data_dirs: Iterable[str | Path], subjects: list[str], tasks: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for task in tasks:
            path = find_zuco_mat_file(data_dirs, subject, task)
            if path is None:
                continue
            rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "path": path,
                    "file_size_mb": path.stat().st_size / 1024 / 1024,
                    "discovery_mode": "configured",
                }
            )
    return pd.DataFrame(rows).drop_duplicates("path") if rows else pd.DataFrame()


def load_matlab_string(matlab_obj) -> str | None:
    try:
        return "".join(chr(int(c[0])) for c in matlab_obj)
    except Exception:
        try:
            arr = np.asarray(matlab_obj).squeeze()
            return "".join(chr(int(c)) for c in arr)
        except Exception:
            return None


def read_scalar_or_none(file_handle: h5py.File, ref) -> float | None:
    try:
        arr = np.asarray(file_handle[ref][()])
        if arr.size == 0:
            return None
        value = np.squeeze(arr)
        if np.asarray(value).size != 1:
            return None
        return float(value)
    except Exception:
        return None


def is_real_word(word: object) -> bool:
    return word is not None and re.search(r"[A-Za-z0-9]", str(word)) is not None


def has_required_fields(h5_group: Any, fields: list[str]) -> bool:
    return all(field in set(h5_group.keys()) for field in fields)


def read_word_field_scalar(file_handle: Any, word_obj: Any, field: str, word_idx: int) -> float | None:
    if field not in word_obj.keys():
        return None
    try:
        return read_scalar_or_none(file_handle, word_obj[field][word_idx][0])
    except Exception:
        return None


def extract_zuco_eye_tracking(mat_path: str | Path, subject_id: str, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to extract ZuCo MATLAB/HDF5 files.") from exc

    mat_path = Path(mat_path)
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []

    with h5py.File(mat_path, "r") as file_handle:
        if "sentenceData" not in file_handle:
            raise KeyError(f"sentenceData not found in {mat_path}")

        sentence_data = file_handle["sentenceData"]
        n_sentences = len(sentence_data["content"])

        for sent_idx in range(n_sentences):
            try:
                sentence_ref = sentence_data["content"][sent_idx][0]
                sentence_text = load_matlab_string(file_handle[sentence_ref])

                word_ref = sentence_data["word"][sent_idx][0]
                word_obj = file_handle[word_ref]
                required = ["content", "FFD", "GD", "GPT", "TRT", "nFixations"]

                if not has_required_fields(word_obj, required):
                    logs.append(
                        {
                            "subject_id": subject_id,
                            "task": task,
                            "source_file": mat_path.name,
                            "sentence_id": sent_idx,
                            "reason": "missing required word-level fields",
                            "available_fields": ",".join(list(word_obj.keys())),
                        }
                    )
                    continue

                n_words = len(word_obj["content"])
                for word_idx in range(n_words):
                    word_text_ref = word_obj["content"][word_idx][0]
                    word_text = load_matlab_string(file_handle[word_text_ref])
                    if not is_real_word(word_text):
                        continue

                    rows.append(
                        {
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
                            "FFD": read_word_field_scalar(file_handle, word_obj, "FFD", word_idx),
                            "GD": read_word_field_scalar(file_handle, word_obj, "GD", word_idx),
                            "GPT": read_word_field_scalar(file_handle, word_obj, "GPT", word_idx),
                            "TRT": read_word_field_scalar(file_handle, word_obj, "TRT", word_idx),
                            "SFD": read_word_field_scalar(file_handle, word_obj, "SFD", word_idx),
                            "nFixations": read_word_field_scalar(file_handle, word_obj, "nFixations", word_idx),
                            "SFD_available_in_file": int("SFD" in word_obj.keys()),
                        }
                    )
            except Exception as exc:
                logs.append(
                    {
                        "subject_id": subject_id,
                        "task": task,
                        "source_file": mat_path.name,
                        "sentence_id": sent_idx,
                        "reason": str(exc),
                        "available_fields": None,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(logs)


def extract_files(file_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    logs: list[pd.DataFrame] = []

    for row in file_index.to_dict("records"):
        table, log = extract_zuco_eye_tracking(row["path"], row["subject_id"], row["task"])
        if not table.empty:
            tables.append(table)
        if not log.empty:
            logs.append(log)

    data = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    log_data = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    return data, log_data
