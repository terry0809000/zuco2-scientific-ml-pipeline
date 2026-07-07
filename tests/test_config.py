from pathlib import Path

from zuco2_pipeline.config import load_config
from zuco2_pipeline.config import write_config_template


def test_load_config_coerces_paths(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
data_dir: data
output_dir: outputs
subjects: [YAC, YAK]
tasks: [NR, TSR]
random_state: 123
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.data_dir == Path("data")
    assert cfg.output_dir == Path("outputs")
    assert cfg.subjects == ["YAC", "YAK"]
    assert cfg.tasks == ["NR", "TSR"]
    assert cfg.random_state == 123


def test_write_config_template_creates_parent_directory(tmp_path):
    cfg_path = tmp_path / "nested" / "config.yaml"

    write_config_template(cfg_path)

    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    assert cfg.subjects == ["YAC"]
