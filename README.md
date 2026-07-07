# ZuCo 2.0 Scientific ML Pipeline

Runnable Python package for word-level ZuCo 2.0 eye-tracking analysis.

The project extracts ZuCo MATLAB/HDF5 `.mat` files, builds lexical and early-fixation features, and evaluates grouped machine-learning baselines for predicting high late word-level processing load. The repository is organized as reusable research software rather than as a single embedded notebook.

## Repository Contents

```text
configs/     YAML configuration files
src/         installable zuco2_pipeline Python package
scripts/     small command-line compatibility wrappers
tests/       package and configuration tests
docs/        data access and method notes
data/        local data mount point, ignored except documentation
outputs/     generated outputs, ignored except placeholder
```

## Quick Start

Create an environment and install the package:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

For macOS or Linux, activate with `source .venv/bin/activate`.

Inspect available commands:

```bash
zuco2-pipeline --help
```

Discover local ZuCo files:

```bash
zuco2-pipeline discover --data-dir data
```

Run the configured pipeline:

```bash
zuco2-pipeline run --config configs/default.yaml
```

The compatibility wrapper below calls the same package entry point:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

## Input Data

Expected input files follow the ZuCo naming pattern:

```text
resultsYAC_NR.mat
resultsYAC_TSR.mat
```

Place local files under `data/` or update `data_dir` in `configs/default.yaml`. The default configuration targets multiple subjects and both normal reading (`NR`) and task-specific reading (`TSR`) when matching files are available.

ZuCo `.mat` files are not redistributed in this repository. Raw data, extracted tables, model outputs, figures, and other generated artifacts are ignored by git. See [docs/DATA.md](docs/DATA.md) for data-access notes.

## Scientific Scope

Primary question:

> Do first-fixation duration features improve prediction of high late word-level processing load beyond lexical and contextual predictors?

The runnable pipeline provides:

- MATLAB/HDF5 extraction for word-level eye-tracking variables;
- lexical, positional, task, and first-fixation feature engineering;
- grouped train/validation/test splitting to reduce sentence-level leakage;
- validation-only model and threshold selection;
- B-series model comparisons from dummy and lexical baselines through additive, moderated, and nonlinear first-fixation models;
- reproducibility outputs including file indices, model-ready tables, prediction files, metrics, and a JSON manifest.

## Validation

The GitHub Actions workflow installs the package, validates YAML configuration, syntax-checks source files, runs the test suite, exercises the CLI, and confirms that restricted data or generated artifacts are not tracked.

Run the same checks locally with:

```bash
pip install -e .[dev]
pytest -q
python -m py_compile src/zuco2_pipeline/*.py scripts/run_pipeline.py
```

## License

Code and documentation in this repository are released under the MIT License. Dataset terms remain governed by the ZuCo data providers.
