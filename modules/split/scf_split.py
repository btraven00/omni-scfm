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

Note on membership: the paper seeds numpy BEFORE `PertData.load` (line 30). On the
paper's cluster `load` found an existing cell-graph cache and drew nothing, so the
published split is exactly `seed -> choice` (a numpy-only recompute reproduces the
paper's test and val sets 31/31 for all five seeds). Under `ob run` every job has a
fresh working dir, `load` rebuilds the cell graphs and draws control cells from
np.random, which would shift the stream. So we seed again AFTER `load`.
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
# load() drew from np.random while building the cell graphs (fresh wd); the paper's
# run hit a cache and drew nothing. Re-seed so the choices below match the paper.
np.random.seed(args.seed)

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
