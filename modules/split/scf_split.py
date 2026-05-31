#!/usr/bin/env python3
"""Faithful port of `prepare_perturbation_data.py`'s `norman_from_scfoundation`
split branch (lines 100-115 + the JSON dump at 123-124).

The verbatim paper script CANNOT run for this dataset off-cluster: its
scfoundation branch hard-requires the forked GEARS 0.0.2 and absolute
`/g/huber/users/ahlmanne/...` paths (lines 43-49). This reproduces only its
split LOGIC with stock cell-gears — validated bit-exact vs the published
additive numbers (62/62 test+val perts, pearson 1.0). It is invoked by
`modules/split/run.sh` exactly like the verbatim script (same CWD/args), so the
GEARS GO-filter (`PertData.load`) runs identically.

Note on membership: the paper seeds numpy BEFORE `PertData.load` (line 30), and
`load` consumes RNG while building the cell-graph cache, so the exact train/test/val
membership depends on that consumption. Under `ob run` every seed runs in a fresh
working dir (cache rebuilt from scratch), so OUR split is deterministic and
reproducible; the precise membership may differ from the paper's by the RNG the
forked GEARS 0.0.2 `load` consumed. The additive *predictions/metrics are
split-independent and bit-exact* — see memory `norman-scfoundation-additive`.
"""
import argparse
import json

import numpy as np
from gears import PertData

p = argparse.ArgumentParser(description="scFoundation Norman double-pert split")
p.add_argument("--dataset_name", required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--working_dir", required=True)
p.add_argument("--result_id", required=True)
args = p.parse_args()

np.random.seed(args.seed)

pert_data = PertData("data/gears_pert_data")
pert_data.load(data_path="data/gears_pert_data/" + args.dataset_name)
adata = pert_data.adata

# --- paper lines 104-115 (verbatim logic) ---
conds = adata.obs["condition"].cat.remove_unused_categories().cat.categories.tolist()
single_pert = [x for x in conds if "ctrl" in x]
double_pert = np.setdiff1d(conds, single_pert).tolist()
double_training = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
double_test = np.setdiff1d(double_pert, double_training).tolist()
double_test = np.random.choice(double_test, size=len(double_test) // 2, replace=False).tolist()
double_holdout = np.setdiff1d(double_pert, double_training + double_test).tolist()
set2conditions = {
    "train": single_pert + double_training,
    "test": double_test,
    "val": double_holdout,
}

with open(args.working_dir + "/results/" + args.result_id, "w") as outfile:
    json.dump(set2conditions, outfile)
print(f"scf_split: {args.dataset_name} seed={args.seed} "
      f"train={len(set2conditions['train'])} test={len(set2conditions['test'])} "
      f"val={len(set2conditions['val'])}")
print("Python done")
