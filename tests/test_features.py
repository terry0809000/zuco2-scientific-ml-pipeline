import numpy as np
import pandas as pd

from zuco2_pipeline.features import build_feature_table


def test_build_feature_table_creates_targets_and_hashes():
    raw = pd.DataFrame(
        {
            "subject_id": ["YAC", "YAC", "YAC"],
            "task": ["NR", "NR", "TSR"],
            "source_file": ["resultsYAC_NR.mat", "resultsYAC_NR.mat", "resultsYAC_TSR.mat"],
            "sentence_id": [1, 1, 2],
            "word_id": [0, 1, 0],
            "sentence": ["A short sentence", "A short sentence", "Another sentence"],
            "word": ["A", "short", "Another"],
            "word_length": [1, 5, 7],
            "sentence_length_words": [3, 3, 2],
            "word_position_norm": [0.0, 0.5, 0.0],
            "FFD": [100.0, 150.0, 200.0],
            "GD": [120.0, 180.0, 210.0],
            "GPT": [130.0, 190.0, 230.0],
            "TRT": [200.0, 220.0, 300.0],
            "SFD": [100.0, np.nan, 200.0],
            "nFixations": [1, 1, 2],
            "SFD_available_in_file": [1, 1, 1],
        }
    )

    df = build_feature_table(raw, late_load_quantile=0.5)

    assert "sentence_hash" in df.columns
    assert "target_high_late_load" in df.columns
    assert "FFD_x_word_length" in df.columns
    assert df["target_skipped"].tolist() == [0, 0, 0]
