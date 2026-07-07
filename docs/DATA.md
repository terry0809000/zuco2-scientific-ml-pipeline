# Data Access and Layout

This repository does not redistribute ZuCo data.

The CLI expects local ZuCo MATLAB/HDF5 files named like:

```text
resultsYAC_NR.mat
resultsYAC_TSR.mat
```

For local runs, place files under:

```text
data/
```

Alternatively, edit `data_dir`, `subjects`, and `tasks` in `configs/default.yaml`.

The repository keeps only code, configuration, and documentation. Raw restricted files and generated analysis products should remain in local or institutionally approved storage.

## Files Excluded From Git

The repository intentionally excludes:

- raw `.mat`, `.h5`, and `.hdf5` files;
- cleaned word-level CSV outputs;
- prediction tables and diagnostics;
- figures;
- caches and local virtual environments.

Generated outputs are expected under the configured `output_dir`, which defaults to `outputs/`.
