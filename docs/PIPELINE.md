# Pipeline Notes

The authoritative implementation is the installable `zuco2_pipeline` package under `src/`.

Run the pipeline with:

```bash
zuco2-pipeline run --config configs/default.yaml
```

## Stages

1. Load the YAML configuration.
2. Discover configured ZuCo `results{SUBJECT}_{TASK}.mat` files.
3. Extract word-level eye-tracking variables from MATLAB/HDF5 structures.
4. Build a model-ready table with lexical, positional, task, and first-fixation features.
5. Define skipped-word and high late-load targets.
6. Fit grouped train/validation/test splits using the configured grouping column.
7. Select model settings and decision thresholds on validation data only.
8. Evaluate the B-series models on the held-out test split.
9. Save file indices, extracted tables, model-ready tables, metrics, predictions, and a reproducibility manifest.

## Primary Question

Do first-fixation duration features improve prediction of high late word-level processing load beyond lexical and contextual predictors?

## Model Series

- `B0`: most-frequent-class dummy baseline.
- `B1`: lexical and positional baseline.
- `B2`: lexical baseline plus additive first-fixation features.
- `B3`: lexical baseline plus moderated first-fixation interactions.
- `B4`: nonlinear first-fixation model using a random forest classifier.

## Outputs

The package writes outputs such as:

- `input_file_index.csv`
- `zuco_extracted_word_level.csv`
- `zuco_ml_ready_word_level.csv`
- `model_B_series_metrics.csv`
- `model_B*_high_late_load_predictions.csv`
- `reproducibility_manifest.json`

These outputs are ignored by git and should be archived separately when reporting results.
