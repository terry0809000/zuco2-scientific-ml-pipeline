import numpy as np
import pandas as pd

from zuco2_pipeline.modeling import TuningOptions, fit_single_model


def test_dummy_model_runs_without_predictor_columns():
    data = pd.DataFrame(
        {
            "target": [0, 1] * 20,
            "group": [f"sentence_{idx}" for idx in range(40)],
            "value": np.linspace(0, 1, 40),
        }
    )

    _, metrics, predictions = fit_single_model(
        data=data,
        target_col="target",
        group_col="group",
        numeric_features=[],
        categorical_features=[],
        model_type="dummy",
        options=TuningOptions(min_labelled_rows=10, tuning_max_candidates=1),
    )

    assert metrics["model_type"] == "dummy"
    assert len(predictions) == metrics["n_test"]
