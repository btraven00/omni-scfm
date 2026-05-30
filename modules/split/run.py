#!/usr/bin/env python3
"""OmniBenchmark module: build the GEARS 'simulation' train/val/test split.

Seed-dependent. Reproduces the reference paper's
`PertData.prepare_split(split='simulation', seed=seed)` using the vendored,
torch-free `DataSplitter` (see src/omni_scfm/_vendor/gears_split.py).

Inputs:
  --data.h5ad PATH   processed AnnData from the `preprocess` stage
Params:
  --seed INT         split seed (paper: 1,2 single-pert; 1..5 double-pert)
Output:
  {name}.set2conditions.json   {"train": [...], "val": [...], "test": [...]}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import anndata as ad  # noqa: E402

from omni_scfm.cli import dataset_name, parse_ob_args, require  # noqa: E402
from omni_scfm.data import simulation_split  # noqa: E402


def main() -> None:
    args = parse_ob_args()
    output_dir = Path(require(args, "output_dir"))
    h5ad = Path(require(args, "data.h5ad", "data_h5ad"))
    name = dataset_name(args, "data.h5ad", "data_h5ad")
    seed = int(args.get("seed", 1))

    # Only obs is needed for the split; read backed to avoid loading X.
    adata = ad.read_h5ad(h5ad, backed="r")
    set2conditions = simulation_split(adata, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{name}.set2conditions.json"
    with open(out, "w", encoding="utf8") as fh:
        json.dump(set2conditions, fh, indent=2)

    sizes = {k: len(v) for k, v in set2conditions.items()}
    print(f"split: seed={seed} sizes={sizes} -> {out}")


if __name__ == "__main__":
    main()
