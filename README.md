# ZuCo 2.0 Scientific ML Pipeline

Notebook-first scientific machine-learning pipeline for ZuCo 2.0 word-level eye-tracking experiments.

The pipeline extracts word-level eye-tracking variables from ZuCo MATLAB/HDF5 `.mat` files, engineers lexical and early-fixation features, evaluates leakage-aware grouped machine-learning models, and reports statistical robustness diagnostics for predicting late word-level processing load.

## Repository Contents

```text
notebooks/   Authoritative executable notebook
scripts/     Auto-extracted Python audit script and extraction helper
docs/        Data access and pipeline notes
data/        Local data mount point, ignored except documentation
outputs/     Generated outputs, ignored except placeholder
```

## Main Workflow

Open the notebook:

```bash
jupyter notebook notebooks/01_zuco2_scientific_ml_pipeline.ipynb
```

The notebook is designed for Google Colab first, with ZuCo files stored under:

```text
/content/drive/MyDrive/ZuCo/
```

Expected input files follow the ZuCo naming pattern:

```text
resultsYAC_NR.mat
resultsYAC_TSR.mat
```

The default configuration loads multiple subjects and both normal reading (`NR`) and task-specific reading (`TSR`) when files are available.

## Scientific Scope

The central B-series question is:

> Do FFD-based early fixation signals robustly improve prediction of high late processing load beyond lexical/contextual features?

The pipeline includes:

- ZuCo MATLAB/HDF5 extraction helpers;
- missingness, skipped-word, leakage, and data-quality audits;
- grouped train/validation/test splitting;
- validation-only hyperparameter and threshold tuning;
- lexical baseline, additive FFD, moderated FFD, and nonlinear FFD model comparisons;
- McNemar tests, paired grouped bootstrap, cross-validation, calibration/error analysis, and feature interpretation;
- optional NR-versus-TSR and subject-held-out analyses.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Policy

ZuCo `.mat` files are not redistributed in this repository. Place local data under `data/` or a private Google Drive folder. Generated CSV outputs, figures, extracted data, and raw `.mat` files are ignored by git.

See [docs/DATA.md](docs/DATA.md) for expected layout and data-access notes.

## Validation

This repository includes a lightweight GitHub Actions workflow that validates notebook JSON, extracts code cells, syntax-checks the generated pipeline script, and checks that raw data/model output artifacts are not tracked.

## License

Code and documentation in this repository are released under the MIT License. Dataset terms remain governed by the ZuCo data providers.
