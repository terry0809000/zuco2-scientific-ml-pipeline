from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .features import build_feature_table, model_feature_sets
from .io import discover_local_zuco_files, extract_files, select_configured_files
from .modeling import TuningOptions, fit_model_series


def select_input_files(config: PipelineConfig) -> pd.DataFrame:
    if config.auto_discover_all_local_mat_files:
        file_index = discover_local_zuco_files(config.search_dirs)
    else:
        file_index = select_configured_files(config.search_dirs, config.subjects or [], config.tasks or [])

    if file_index.empty:
        raise FileNotFoundError(
            "No ZuCo .mat files were found. Check data_dir, subjects, tasks, "
            "or enable auto_discover_all_local_mat_files."
        )
    return file_index


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    config.ensure_output_dir()
    file_index = select_input_files(config)
    file_index_for_write = file_index.copy()
    file_index_for_write["path"] = file_index_for_write["path"].astype(str)
    file_index_path = config.output_dir / "input_file_index.csv"
    file_index_for_write.to_csv(file_index_path, index=False)

    raw, extraction_log = extract_files(file_index)
    if raw.empty:
        raise RuntimeError("No word-level rows were extracted from the selected files.")

    features = build_feature_table(raw, late_load_quantile=config.late_load_quantile)
    raw_path = config.output_dir / "zuco_extracted_word_level.csv"
    features_path = config.output_dir / "zuco_ml_ready_word_level.csv"
    raw.to_csv(raw_path, index=False)
    features.to_csv(features_path, index=False)

    if not extraction_log.empty:
        extraction_log.to_csv(config.output_dir / "zuco_extraction_log.csv", index=False)

    options = TuningOptions(
        random_state=config.random_state,
        test_size=config.test_size,
        validation_size=config.validation_size,
        tune_model_hyperparameters=config.tune_model_hyperparameters,
        tune_classification_threshold=config.tune_classification_threshold,
        threshold_optimization_metric=config.threshold_optimization_metric,
        tuning_max_candidates=config.tuning_max_candidates,
        min_labelled_rows=config.min_labelled_rows,
    )
    metrics, predictions = fit_model_series(
        data=features,
        target_col="target_high_late_load",
        group_col=config.group_col_for_validation,
        model_specs=model_feature_sets(config.use_task_as_feature),
        options=options,
    )
    metrics_path = config.output_dir / "model_B_series_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    prediction_paths = {}
    for model_code, table in predictions.items():
        path = config.output_dir / f"model_{model_code}_high_late_load_predictions.csv"
        table.to_csv(path, index=False)
        prediction_paths[model_code] = str(path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "subjects": config.subjects,
        "tasks": config.tasks,
        "auto_discover_all_local_mat_files": config.auto_discover_all_local_mat_files,
        "group_col_for_validation": config.group_col_for_validation,
        "use_task_as_feature": config.use_task_as_feature,
        "random_state": config.random_state,
        "late_load_quantile": config.late_load_quantile,
        "n_input_files": int(len(file_index)),
        "n_rows_extracted": int(len(raw)),
        "n_rows_model_ready": int(len(features)),
        "n_subjects": int(features["subject_id"].nunique()) if "subject_id" in features else 0,
        "n_tasks": int(features["task"].nunique()) if "task" in features else 0,
        "central_question": "Do first-fixation duration features improve prediction of high late word-level processing load beyond lexical and contextual predictors?",
        "outputs": {
            "input_file_index": str(file_index_path),
            "extracted_word_level": str(raw_path),
            "ml_ready_word_level": str(features_path),
            "metrics": str(metrics_path),
            "predictions": prediction_paths,
        },
    }
    manifest_path = config.output_dir / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "file_index": file_index_path,
        "raw": raw_path,
        "features": features_path,
        "metrics": metrics_path,
        "manifest": manifest_path,
    }
