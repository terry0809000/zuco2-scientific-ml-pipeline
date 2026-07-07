from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


LEXICAL_NUMERIC = ["word_length", "log_word_length", "sentence_length_words", "word_position_norm"]
BASE_EYE_NUMERIC = ["FFD", "log_FFD"]
INTERACTION_NUMERIC = ["FFD", "log_FFD", "FFD_x_word_length", "FFD_x_position_norm"]
NONLINEAR_NUMERIC = ["FFD", "log_FFD", "FFD_squared", "FFD_x_word_length", "FFD_x_position_norm"]
TIME_COLUMNS = ["FFD", "GD", "GPT", "TRT", "SFD", "nFixations"]


def sentence_hash(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_feature_table(raw: pd.DataFrame, late_load_quantile: float = 0.75) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()

    df = raw.copy()
    for col in TIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sentence_uid"] = (
        df["subject_id"].astype(str)
        + "_"
        + df["task"].astype(str)
        + "_"
        + df["source_file"].astype(str)
        + "_"
        + df["sentence_id"].astype(str)
    )
    df["sentence_hash"] = df["sentence"].map(sentence_hash)
    df["word"] = df["word"].astype(str)
    df["word_lower"] = df["word"].str.lower()
    df["log_word_length"] = np.log1p(df["word_length"].clip(lower=0))

    fixation_signal = df[["FFD", "GD", "TRT", "nFixations"]].fillna(0).sum(axis=1)
    df["is_fixated"] = (fixation_signal > 0).astype(int)
    df["skipped"] = (df["is_fixated"] == 0).astype(int)

    df["late_time"] = df["TRT"] - df["GD"]
    df.loc[df["late_time"] < 0, "late_time"] = np.nan
    valid_late = df["late_time"].dropna()
    threshold = valid_late.quantile(late_load_quantile) if len(valid_late) else np.nan
    df["late_load_threshold"] = threshold
    df["target_high_late_load"] = np.where(df["late_time"].notna(), (df["late_time"] >= threshold).astype(int), np.nan)
    df["target_skipped"] = df["skipped"].astype(int)
    df["target_task_TSR"] = np.where(df["task"].isin(["NR", "TSR"]), (df["task"] == "TSR").astype(int), np.nan)

    for col in ["FFD", "GD", "GPT", "TRT", "SFD"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    df["FFD_squared"] = df["FFD"] ** 2
    df["FFD_x_word_length"] = df["FFD"] * df["word_length"]
    df["FFD_x_position_norm"] = df["FFD"] * df["word_position_norm"]
    return df


def categorical_features(use_task_as_feature: bool = True) -> list[str]:
    return ["task"] if use_task_as_feature else []


def model_feature_sets(use_task_as_feature: bool = True) -> list[dict[str, object]]:
    cats = categorical_features(use_task_as_feature)
    return [
        {"model_code": "B0", "model_label": "dummy baseline", "numeric": [], "categorical": [], "model_type": "dummy"},
        {"model_code": "B1", "model_label": "lexical baseline", "numeric": LEXICAL_NUMERIC, "categorical": cats, "model_type": "logreg"},
        {"model_code": "B2", "model_label": "additive FFD", "numeric": LEXICAL_NUMERIC + BASE_EYE_NUMERIC, "categorical": cats, "model_type": "logreg"},
        {"model_code": "B3", "model_label": "moderated FFD", "numeric": LEXICAL_NUMERIC + INTERACTION_NUMERIC, "categorical": cats, "model_type": "logreg"},
        {"model_code": "B4", "model_label": "nonlinear FFD", "numeric": LEXICAL_NUMERIC + NONLINEAR_NUMERIC, "categorical": cats, "model_type": "rf"},
    ]
