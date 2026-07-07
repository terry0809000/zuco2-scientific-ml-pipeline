# Pipeline Notes

The authoritative implementation is `notebooks/01_zuco2_scientific_ml_pipeline.ipynb`.

## Stages

1. Mount or locate data storage.
2. Import scientific Python dependencies.
3. Configure subjects, tasks, OSF discovery, validation grouping, and tuning options.
4. Discover `results{SUBJECT}_{TASK}.mat` files.
5. Inspect ZuCo MATLAB/HDF5 file structure.
6. Extract word-level eye-tracking variables.
7. Clean the long-format table and define skipped-word and late-load targets.
8. Engineer lexical, positional, task, and early-fixation features.
9. Audit missingness, leakage risk, and class balance.
10. Train grouped machine-learning models with validation-only threshold and hyperparameter tuning.
11. Compare B-series models aligned to the same scientific question.
12. Run statistical robustness diagnostics, cross-validation, bootstrap intervals, and error analysis.
13. Save cleaned data, predictions, diagnostics, and a reproducibility manifest.

## Primary Question

Do FFD-based early fixation signals robustly improve prediction of high late processing load beyond lexical/contextual features?

## Outputs

The notebook writes outputs such as:

- `zuco_cleaned_eye_tracking_word_level.csv`
- `zuco_ml_ready_eye_tracking_word_level.csv`
- `model_B*_high_late_load_predictions.csv`
- `model_B_series_error_analysis.csv`
- `reproducibility_manifest.json`

These outputs are ignored by git and should be archived separately when reporting results.
