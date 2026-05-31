#!/usr/bin/env python3
"""Build a tiny norman fixture for local smoke-testing method modules.

A stripped-down slice of the GEARS-processed Norman dataset: a handful of
conditions (ctrl + a few singles + a few doubles, chosen so every double's two
single-gene components are present) and few cells per condition, but ALL genes
kept (GEARS/additive need the perturbed genes + GO coverage intact). Emits a
self-contained fixture that the method `run.sh`s consume verbatim:

  norman_mini.h5ad              perturb_processed-style AnnData (subset rows)
  norman_mini.gene_names.json   gene symbols in X-column order
  norman_mini.set2conditions.json  {train,val,test} split (doubles held out so
                                their singles stay in train -> additive works)
  norman_mini.go.csv            copied from the source dataset's go.csv

Not part of the benchmark — lives under scratch/ (gitignored). The plan is to
later "bless" a fixture like this into OB as a proper test dataset.

Usage:
  python scripts/make_mini_norman.py \
    --src out/download/norman/.*/preprocess/.../norman.h5ad \
    --go  out/download/norman/.*/preprocess/.../norman.go.csv \
    --out scratch/mini_norman [--n-cells 60 --n-doubles 6 --seed 0]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source norman perturb_processed h5ad")
    ap.add_argument("--go", default=None, help="source go.csv (copied verbatim)")
    ap.add_argument("--out", default="scratch/mini_norman", help="output fixture dir")
    ap.add_argument("--name", default="norman_mini", help="fixture dataset name")
    ap.add_argument("--n-cells", type=int, default=60, help="cells per kept condition")
    ap.add_argument("--n-doubles", type=int, default=6,
                    help="double perturbations to keep (3 train / 2 test / 1 val)")
    ap.add_argument("--max-genes", type=int, default=0,
                    help="if >0, keep only this many genes (all perturbed genes + "
                         "top-expressed to fill) and drop GEARS-only uns/layers, for a "
                         "tiny COMMITTABLE baseline-test fixture. 0 = keep all genes "
                         "(GEARS-capable, large; lives in scratch).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    adata = ad.read_h5ad(args.src)
    cond = adata.obs["condition"].astype(str)
    uniq = set(cond.unique())

    def single(g: str) -> str:
        return f"{g}+ctrl"

    singles = {c for c in uniq if c.endswith("+ctrl") and c != "ctrl"}
    # doubles whose BOTH single-gene components are available
    predictable_doubles = []
    for c in sorted(uniq):
        parts = c.split("+")
        if len(parts) == 2 and "ctrl" not in parts:
            if single(parts[0]) in singles and single(parts[1]) in singles:
                predictable_doubles.append(c)
    if len(predictable_doubles) < args.n_doubles:
        raise SystemExit(
            f"only {len(predictable_doubles)} additive-predictable doubles available, "
            f"need {args.n_doubles}")
    chosen_doubles = predictable_doubles[: args.n_doubles]

    needed_singles = sorted({single(g) for d in chosen_doubles for g in d.split("+")})
    keep_conds = ["ctrl"] + needed_singles + chosen_doubles

    # subsample cells per kept condition
    cond_vals = cond.values
    keep_idx = []
    for c in keep_conds:
        ci = np.flatnonzero(cond_vals == c)
        if ci.size > args.n_cells:
            ci = rng.choice(ci, args.n_cells, replace=False)
        keep_idx.append(ci)
    keep_idx = np.sort(np.concatenate(keep_idx))

    mini = adata[keep_idx].copy()
    mini.obs["condition"] = (
        mini.obs["condition"].astype("category").cat.remove_unused_categories()
    )

    # Optional gene trim -> tiny committable fixture for baseline entrypoint tests.
    # GEARS-only state (uns DE-index rankings, obsm/obsp/varm/varp, dense layers)
    # indexes the *original* gene axis, so it can't survive a gene subset; drop it.
    # mean/lpm read only X + obs.condition + var.gene_name, so they still run.
    if args.max_genes and args.max_genes < mini.n_vars:
        names = mini.var["gene_name"].astype(str)
        perturbed = {g for d in chosen_doubles + needed_singles for g in d.split("+")
                     if g != "ctrl"}
        keep_gene = names.isin(perturbed).to_numpy()
        # fill remaining budget with the most-expressed genes (stable, deterministic)
        import numpy as _np
        expr = _np.asarray(mini.X.mean(axis=0)).ravel()
        order = _np.argsort(-expr)
        for j in order:
            if keep_gene.sum() >= args.max_genes:
                break
            keep_gene[j] = True
        mini = mini[:, keep_gene].copy()
        for attr in ("uns", "obsm", "obsp", "varm", "varp", "layers"):
            getattr(mini, attr).clear()

    if args.max_genes:
        kept = set(mini.var["gene_name"].astype(str))

    # split: hold out the last 2 doubles for test, 1 for val; their singles stay
    # in train so additive can reconstruct them. All singles + ctrl go to train.
    test_doubles = chosen_doubles[-2:]
    val_doubles = chosen_doubles[-3:-2]
    train_doubles = chosen_doubles[:-3]
    set2 = {
        "train": ["ctrl"] + needed_singles + train_doubles,
        "val": val_doubles,
        "test": test_doubles,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    h5ad = out / f"{args.name}.h5ad"
    mini.write_h5ad(h5ad)
    gene_names = mini.var["gene_name"].astype(str).tolist() if "gene_name" in mini.var \
        else mini.var_names.astype(str).tolist()
    (out / f"{args.name}.gene_names.json").write_text(json.dumps(gene_names))
    (out / f"{args.name}.set2conditions.json").write_text(json.dumps(set2, indent=2))
    out_go = out / f"{args.name}.go.csv"
    if args.go and Path(args.go).exists():
        if args.max_genes:
            # keep only GO edges among kept genes, so the committable fixture's
            # go.csv is tiny too
            import pandas as pd
            go = pd.read_csv(args.go)
            gcols = [c for c in ("source", "target") if c in go.columns]
            if len(gcols) == 2:
                go = go[go["source"].isin(kept) & go["target"].isin(kept)]
            go.to_csv(out_go, index=False)
        else:
            shutil.copy(args.go, out_go)
    else:
        out_go.write_text("source,target,importance\n")

    print(f"wrote {h5ad}: {mini.n_obs} cells x {mini.n_vars} genes, "
          f"{len(keep_conds)} conditions")
    print(f"  train={len(set2['train'])} (ctrl+{len(needed_singles)} singles+"
          f"{len(train_doubles)} doubles)  val={set2['val']}  test={set2['test']}")


if __name__ == "__main__":
    main()
