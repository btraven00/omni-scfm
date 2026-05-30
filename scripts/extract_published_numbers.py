#!/usr/bin/env python3
"""Extract the paper's published per-perturbation metrics into tidy CSV/JSON.

The reference numbers (Ahlmann-Eltze et al. 2025) ship as figure source-data
spreadsheets in ``vendor/paper/source_data/``. Their per-perturbation metric
sheets ("Panel A") use the columns
``dataset_name, seed, method, perturbation, train, r2, r2_delta, l2[, approach, label]``
— the same quantities our collector produces. This script normalises them to our
``scores.parquet`` schema so the two can be compared directly:

    r2 -> pearson, r2_delta -> pearson_delta, train -> split

Usage:
    python scripts/extract_published_numbers.py            # -> scratch/published/*.csv
    python scripts/extract_published_numbers.py --format json --out-dir somewhere
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Source spreadsheets and the sheet holding per-perturbation metrics.
SOURCES = {
    "single": ("single_perturbation_prediction.xlsx", "Panel A"),
    "double": ("perturbation_prediction.xlsx", "Panel A"),
}

RENAME = {"dataset_name": "dataset", "r2": "pearson", "r2_delta": "pearson_delta", "train": "split"}
COLUMNS = ["dataset", "seed", "method", "perturbation", "split",
           "pearson", "pearson_delta", "l2", "approach", "label"]


def extract(source_data: Path, key: str) -> pd.DataFrame:
    """Read one source spreadsheet's per-perturbation metrics, normalised."""
    fname, sheet = SOURCES[key]
    df = pd.read_excel(source_data / fname, sheet_name=sheet).rename(columns=RENAME)
    keep = [c for c in COLUMNS if c in df.columns]
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-data", default="vendor/paper/source_data",
                    help="dir with the paper's source-data xlsx (default: vendor/paper/source_data)")
    ap.add_argument("--out-dir", default="scratch/published",
                    help="output directory (default: scratch/published)")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    args = ap.parse_args()

    src, out = Path(args.source_data), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in SOURCES:
        try:
            df = extract(src, key)
        except FileNotFoundError:
            print(f"skip {key}: {SOURCES[key][0]} not found in {src}")
            continue
        dst = out / f"published_{key}.{args.format}"
        if args.format == "csv":
            df.to_csv(dst, index=False)
        else:
            df.to_json(dst, orient="records")
        print(f"{key}: {len(df)} rows, {df['method'].nunique()} methods, "
              f"{df['dataset'].nunique()} datasets -> {dst}")


if __name__ == "__main__":
    main()
