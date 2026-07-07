#!/usr/bin/env python
"""Export code cells from the ZuCo 2.0 notebook into a single audit script."""

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/01_zuco2_scientific_ml_pipeline.ipynb")
OUTPUT_PATH = Path("scripts/zuco2_pipeline.py")


def main() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    chunks = [
        "# Auto-extracted from notebooks/01_zuco2_scientific_ml_pipeline.ipynb\n",
        "# The notebook remains the authoritative executable workflow.\n",
    ]

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.source.strip():
            chunks.append(f"\n# %% [cell {i}]\n")
            chunks.append(cell.source.rstrip() + "\n")

    OUTPUT_PATH.write_text("".join(chunks), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
