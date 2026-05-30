#!/usr/bin/env python3
"""OmniBenchmark method module: ground truth (observed condition means).

The reference target for scoring: observed mean expression per condition. Ports
`vendor/paper/benchmark/src/run_ground_truth_for_combinatorial_perturbations.py`
(reads the processed AnnData from the `preprocess` stage instead of via GEARS).

Inputs:
  --data.h5ad PATH   processed AnnData from `preprocess`
  (--split.set2conditions is provided by the stage but unused: ground truth is
   seed-independent.)
Outputs:
  {dataset}.predictions.json      {condition: [observed mean per gene]}
  {dataset}.gene_names.json
  {dataset}.predictions_se.json    {condition: [std/n_cells per gene]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import anndata as ad  # noqa: E402

from omni_scfm.cli import dataset_name, parse_ob_args, require  # noqa: E402
from omni_scfm.io import write_predictions  # noqa: E402
from omni_scfm.methods import gene_names_from, observed_condition_stats  # noqa: E402


def main() -> None:
    args = parse_ob_args()
    output_dir = Path(require(args, "output_dir"))
    h5ad = Path(require(args, "data.h5ad", "data_h5ad"))
    dataset = dataset_name(args, "data.h5ad", "data_h5ad")

    adata = ad.read_h5ad(h5ad)
    means, se, _ = observed_condition_stats(adata)
    write_predictions(output_dir, dataset, means, gene_names_from(adata), se=se)

    print(f"ground_truth: {len(means)} conditions, {adata.shape[1]} genes -> {dataset}.predictions.json")


if __name__ == "__main__":
    main()
