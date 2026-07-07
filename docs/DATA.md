# Data Access and Layout

This repository does not redistribute ZuCo data.

The notebook expects local ZuCo MATLAB/HDF5 files named like:

```text
resultsYAC_NR.mat
resultsYAC_TSR.mat
```

For Google Colab, the default location is:

```text
/content/drive/MyDrive/ZuCo/
```

For local runs, place files under:

```text
data/
```

The notebook can optionally index OSF storage when `USE_OSF_DISCOVERY = True`. Downloading from OSF is disabled by default and should only be enabled after inspecting the file index because the dataset can be large.

## Files Excluded From Git

The repository intentionally excludes:

- raw `.mat`, `.h5`, and `.hdf5` files;
- cleaned word-level CSV outputs;
- prediction tables and diagnostics;
- figures;
- caches and local virtual environments.

Generated outputs are expected under the notebook's configured `OUTPUT_DIR`.
