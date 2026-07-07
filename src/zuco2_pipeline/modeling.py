from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TuningOptions:
    random_state: int = 42
    test_size: float = 0.25
    validation_size: float = 0.20
    tune_model_hyperparameters: bool = True
    tune_classification_threshold: bool = True
    threshold_optimization_metric: str = "balanced_accuracy"
    tuning_max_candidates: int = 12
    min_labelled_rows: int = 30


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot_encoder())]),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: str,
    random_state: int,
) -> Pipeline:
    if model_type == "dummy":
        classifier = DummyClassifier(strategy="most_frequent")
    elif model_type == "logreg":
        classifier = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    elif model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            ("classifier", classifier),
        ]
    )


def candidate_parameter_grid(model_type: str, options: TuningOptions) -> list[dict[str, Any]]:
    if (not options.tune_model_hyperparameters) or model_type == "dummy":
        return [{}]
    if model_type == "logreg":
        candidates = [
            {"classifier__C": c_value, "classifier__class_weight": class_weight}
            for c_value in [0.05, 0.10, 0.30, 1.0, 3.0, 10.0]
            for class_weight in ["balanced", None]
        ]
    elif model_type == "rf":
        candidates = [
            {
                "classifier__max_depth": max_depth,
                "classifier__min_samples_leaf": min_leaf,
                "classifier__max_features": max_features,
                "classifier__class_weight": class_weight,
            }
            for max_depth in [4, 8, None]
            for min_leaf in [2, 5, 10]
            for max_features in ["sqrt", 0.60]
            for class_weight in ["balanced", "balanced_subsample"]
        ]
    else:
        candidates = [{}]

    if len(candidates) > options.tuning_max_candidates:
        rng = np.random.default_rng(options.random_state)
        idx = rng.choice(len(candidates), size=options.tuning_max_candidates, replace=False)
        candidates = [candidates[i] for i in idx]
    return candidates


def binary_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
        metrics["brier_score"] = brier_score_loss(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
        metrics["brier_score"] = np.nan
    return {key: float(value) if pd.notna(value) else np.nan for key, value in metrics.items()}


def choose_threshold(y_true, y_score, metric: str) -> tuple[float, float]:
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.05, 0.95, 91)))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def grouped_split(data: pd.DataFrame, target_col: str, group_col: str, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    y = data[target_col].astype(int)
    groups = data[group_col]
    best = None
    best_diff = np.inf
    for train_idx, test_idx in splitter.split(data, y, groups):
        train_y = y.iloc[train_idx]
        test_y = y.iloc[test_idx]
        if train_y.nunique() < 2 or test_y.nunique() < 2:
            continue
        diff = abs(train_y.mean() - test_y.mean())
        if diff < best_diff:
            best = (train_idx, test_idx)
            best_diff = diff
    if best is None:
        raise ValueError(f"Could not create a grouped split with both classes for {target_col}")
    return best


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def fit_single_model(
    data: pd.DataFrame,
    target_col: str,
    group_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: str,
    options: TuningOptions,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    features = numeric_features + categorical_features
    if model_type == "dummy" and not features:
        features = ["__constant__"]
        numeric_features = ["__constant__"]
    if model_type != "dummy" and not features:
        raise ValueError("Non-dummy models require at least one feature")

    model_data = data.dropna(subset=[target_col, group_col]).copy()
    if "__constant__" in features:
        model_data["__constant__"] = 1.0
    cols = [target_col, group_col] + [feature for feature in features if feature != "__constant__"]
    if len(model_data) < options.min_labelled_rows:
        raise ValueError(f"Not enough labelled rows for {target_col}: {len(model_data)}")
    if model_data[target_col].nunique() < 2:
        raise ValueError(f"Target {target_col} has fewer than two classes")

    train_valid_idx, test_idx = grouped_split(
        model_data,
        target_col=target_col,
        group_col=group_col,
        test_size=options.test_size,
        random_state=options.random_state,
    )
    train_valid = model_data.iloc[train_valid_idx].copy()
    test = model_data.iloc[test_idx].copy()
    train_idx, valid_idx = grouped_split(
        train_valid,
        target_col=target_col,
        group_col=group_col,
        test_size=options.validation_size,
        random_state=options.random_state + 1,
    )
    train = train_valid.iloc[train_idx].copy()
    valid = train_valid.iloc[valid_idx].copy()

    base_model = make_classifier_pipeline(numeric_features, categorical_features, model_type, options.random_state)
    best_model = None
    best_params: dict[str, Any] = {}
    best_valid_score = -np.inf

    for params in candidate_parameter_grid(model_type, options):
        model = clone(base_model)
        model.set_params(**params)
        model.fit(train[features], train[target_col].astype(int))
        valid_score = predict_scores(model, valid[features])
        valid_pred = model.predict(valid[features]) if valid_score is None else (valid_score >= 0.5).astype(int)
        score = balanced_accuracy_score(valid[target_col].astype(int), valid_pred)
        if score > best_valid_score:
            best_valid_score = score
            best_model = model
            best_params = params

    assert best_model is not None
    valid_score = predict_scores(best_model, valid[features])
    threshold = 0.5
    threshold_score = best_valid_score
    if options.tune_classification_threshold and valid_score is not None:
        threshold, threshold_score = choose_threshold(
            valid[target_col].astype(int).to_numpy(),
            valid_score,
            metric=options.threshold_optimization_metric,
        )

    final_model = clone(best_model)
    final_model.fit(train_valid[features], train_valid[target_col].astype(int))

    test_score = predict_scores(final_model, test[features])
    if test_score is None:
        test_pred = final_model.predict(test[features])
    else:
        test_pred = (test_score >= threshold).astype(int)

    metrics = binary_metrics(test[target_col].astype(int), test_pred, test_score)
    metrics.update(
        {
            "target": target_col,
            "model_type": model_type,
            "n_train_valid": len(train_valid),
            "n_test": len(test),
            "n_train_valid_groups": train_valid[group_col].nunique(),
            "n_test_groups": test[group_col].nunique(),
            "positive_rate_train_valid": float(train_valid[target_col].mean()),
            "positive_rate_test": float(test[target_col].mean()),
            "decision_threshold": threshold,
            "validation_score": threshold_score,
            "selected_params": str(best_params),
        }
    )

    predictions = test[cols].copy()
    predictions[f"pred_{target_col}"] = test_pred
    predictions[f"prob_{target_col}"] = test_score if test_score is not None else np.nan
    return final_model, metrics, predictions


def fit_model_series(
    data: pd.DataFrame,
    target_col: str,
    group_col: str,
    model_specs: list[dict[str, object]],
    options: TuningOptions,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_tables: dict[str, pd.DataFrame] = {}

    for spec in model_specs:
        try:
            _, metrics, predictions = fit_single_model(
                data=data,
                target_col=target_col,
                group_col=group_col,
                numeric_features=list(spec["numeric"]),
                categorical_features=list(spec["categorical"]),
                model_type=str(spec["model_type"]),
                options=options,
            )
            metrics["model_code"] = spec["model_code"]
            metrics["model_label"] = spec["model_label"]
            metric_rows.append(metrics)
            prediction_tables[str(spec["model_code"])] = predictions
        except Exception as exc:
            metric_rows.append(
                {
                    "model_code": spec["model_code"],
                    "model_label": spec["model_label"],
                    "target": target_col,
                    "model_type": spec["model_type"],
                    "status": "skipped",
                    "reason": str(exc),
                }
            )

    return pd.DataFrame(metric_rows), prediction_tables
