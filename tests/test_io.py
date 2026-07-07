from pathlib import Path

from zuco2_pipeline.io import parse_subject_task_from_filename


def test_parse_subject_task_from_filename():
    assert parse_subject_task_from_filename(Path("resultsYAC_NR.mat")) == ("YAC", "NR")
    assert parse_subject_task_from_filename(Path("resultsYAC_TSR.mat")) == ("YAC", "TSR")
    assert parse_subject_task_from_filename(Path("other.mat")) == (None, None)
