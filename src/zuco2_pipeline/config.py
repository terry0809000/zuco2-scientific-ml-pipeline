from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    subjects: list[str] | None = None
    tasks: list[str] | None = None
    auto_discover_all_local_mat_files: bool = False
    group_col_for_validation: str = "sentence_hash"
    use_task_as_feature: bool = True
    random_state: int = 42
    test_size: float = 0.25
    validation_size: float = 0.20
    tune_model_hyperparameters: bool = True
    tune_classification_threshold: bool = True
    threshold_optimization_metric: str = "balanced_accuracy"
    tuning_max_candidates: int = 12
    late_load_quantile: float = 0.75
    min_labelled_rows: int = 30

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.subjects is None:
            self.subjects = ["YAC"]
        if self.tasks is None:
            self.tasks = ["NR"]

    @property
    def search_dirs(self) -> list[Path]:
        return [self.data_dir]

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _coerce_data(data: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(PipelineConfig)}
    coerced = {key: value for key, value in (data or {}).items() if key in allowed}
    for key in ["data_dir", "output_dir"]:
        if key in coerced:
            coerced[key] = Path(coerced[key])
    return coerced


def load_config(path: str | Path = "configs/default.yaml") -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return PipelineConfig(**_coerce_data(data))


def write_config_template(path: str | Path) -> None:
    cfg = PipelineConfig()
    data = {
        "data_dir": str(cfg.data_dir),
        "output_dir": str(cfg.output_dir),
        "subjects": cfg.subjects,
        "tasks": cfg.tasks,
        "auto_discover_all_local_mat_files": cfg.auto_discover_all_local_mat_files,
        "group_col_for_validation": cfg.group_col_for_validation,
        "use_task_as_feature": cfg.use_task_as_feature,
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "validation_size": cfg.validation_size,
        "tune_model_hyperparameters": cfg.tune_model_hyperparameters,
        "tune_classification_threshold": cfg.tune_classification_threshold,
        "threshold_optimization_metric": cfg.threshold_optimization_metric,
        "tuning_max_candidates": cfg.tuning_max_candidates,
        "late_load_quantile": cfg.late_load_quantile,
        "min_labelled_rows": cfg.min_labelled_rows,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
