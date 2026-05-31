#!/usr/bin/env python3
"""OmniBenchmark module: preprocess a downloaded GEARS dataset.

Seed-independent. Dataset-aware, mirroring the paper's own
`prepare_perturbation_data.py` (one script that branches on `dataset_name`):

  * Normal GEARS archive (adamson, norman): a `*.zip` already containing
    `<name>/perturb_processed.h5ad` + `go.csv`. We extract it and canonicalize
    condition names (sort genes within each), matching the paper's
    `normalize_condition_names` for plain norman.
  * scFoundation Norman (`*scfoundation*`): the raw figshare h5ad
    (`gse133344_…_withtotalcount.h5ad`, 19264 genes, total counts), which must be
    run through GEARS `PertData.new_data_process` to become `perturb_processed.h5ad`
    (paper lines 51-62 — reproduced with stock cell-gears, no forked GEARS / cluster
    paths). Conditions are NOT sorted here: the paper's scfoundation split uses raw
    condition order, and sorting would change the seeded train/test/val selection.

Hence this module runs in the **gears** env (needs cell-gears for new_data_process).

Emits (for every dataset):
  - {name}.h5ad             processed AnnData, re-written via anndata (adds the
                            modern root `encoding-type` so picklerick can read it).
  - {name}.gene_names.json  gene symbols in X-column order (from var.gene_name)
  - {name}.go.csv           the dataset's GO table (from the archive). For
                            scFoundation there is none — GEARS' GO data lives in
                            `gene2go.pkl` (used by split/additive); we write a
                            header-only placeholder to honour the output contract.

Inputs (OB flag = upstream output id):
  --data.raw PATH   path to the downloaded archive (.zip) or a .h5ad directly
Params: none.
"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Monorepo: make the shared library importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402

from omni_scfm.cli import dataset_name, parse_ob_args, require  # noqa: E402


def _find_h5ad(raw_path: Path, workdir: Path) -> Path:
    if raw_path.suffix == ".h5ad":
        return raw_path
    if zipfile.is_zipfile(raw_path):
        with zipfile.ZipFile(raw_path) as zf:
            members = [m for m in zf.namelist() if m.endswith("perturb_processed.h5ad")]
            if not members:
                members = [m for m in zf.namelist() if m.endswith(".h5ad")]
            if not members:
                raise SystemExit(f"no .h5ad found inside {raw_path}")
            target = workdir / "extracted.h5ad"
            with zf.open(members[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return target
    raise SystemExit(f"unrecognised raw input (expected .zip or .h5ad): {raw_path}")


def _extract_go(raw_path: Path, dest: Path) -> bool:
    """Pull go.csv out of the archive to `dest`; return False if absent."""
    if not zipfile.is_zipfile(raw_path):
        return False
    with zipfile.ZipFile(raw_path) as zf:
        members = [m for m in zf.namelist() if m.endswith("go.csv")]
        if not members:
            return False
        with zf.open(members[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return True


def _new_data_process_scfoundation(raw_h5ad: Path, name: str, workdir: Path):
    """Build GEARS `perturb_processed.h5ad` from the raw scFoundation Norman h5ad.

    Reproduces `prepare_perturbation_data.py` lines 51-62 with stock cell-gears
    (no forked GEARS 0.0.2 / cluster paths — validated bit-exact vs the published
    additive numbers). Returns the processed AnnData (conditions left in raw order).
    """
    import os

    import scanpy as sc
    from gears import PertData

    folder = workdir / "gears_pert_data"
    folder.mkdir(parents=True, exist_ok=True)
    # Reuse a cached gene2go if present, else GEARS downloads it (~9MB) itself.
    cache = os.environ.get("OMNI_GEARS_CACHE")
    if cache and Path(cache).is_dir():
        for pkl in Path(cache).glob("*.pkl"):
            shutil.copy(pkl, folder / pkl.name)

    adata = sc.read_h5ad(raw_h5ad)
    adata.uns["log1p"] = {"base": None}
    pert = PertData(str(folder))
    pert.new_data_process(dataset_name=name, adata=adata)
    return ad.read_h5ad(folder / name / "perturb_processed.h5ad")


def _canonicalize_sparse(adata) -> None:
    """Sort indices of any sparse X / layers in place.

    GEARS `new_data_process` (and some figshare h5ads) emit CSR/CSC matrices with
    unsorted within-major indices. anndata/scipy tolerate this, but R's Matrix
    package rejects it on read ("invalid dgCMatrix object: 'i' slot is not
    increasing within columns"), which breaks the R methods (mean, lpm). Sorting
    is a no-op when already canonical.
    """
    from scipy.sparse import issparse

    for mat in (adata.X, *adata.layers.values()):
        if issparse(mat) and hasattr(mat, "sort_indices") and not mat.has_sorted_indices:
            mat.sort_indices()


def main() -> None:
    args = parse_ob_args()
    output_dir = Path(require(args, "output_dir"))
    raw = Path(require(args, "data.raw", "data_raw"))
    name = dataset_name(args, "data.raw", "data_raw")
    scfoundation = "scfoundation" in name.lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = _find_h5ad(raw, Path(tmp))
        if scfoundation:
            # Raw figshare h5ad -> GEARS processing; keep raw condition order.
            adata = _new_data_process_scfoundation(h5ad, name, Path(tmp))
        else:
            adata = ad.read_h5ad(h5ad)
            # Canonicalize condition names (sort genes within each), matching the
            # paper's normalize_condition_names, so double-pert names are order-
            # independent (A+B == B+A) and consistent across split / ground_truth /
            # methods. No-op for single-pert datasets (symbols sort before 'ctrl').
            cond = adata.obs["condition"].astype(str)
            adata.obs["condition"] = pd.Categorical(["+".join(sorted(c.split("+"))) for c in cond])

        gene_names = (
            adata.var["gene_name"].astype(str).tolist()
            if "gene_name" in adata.var.columns
            else adata.var_names.astype(str).tolist()
        )

        _canonicalize_sparse(adata)  # sort sparse indices so R's Matrix accepts X
        out_h5ad = output_dir / f"{name}.h5ad"
        adata.write_h5ad(out_h5ad)  # adds modern encoding-type attrs (picklerick-readable)
        with open(output_dir / f"{name}.gene_names.json", "w", encoding="utf8") as fh:
            json.dump(gene_names, fh)
        has_go = _extract_go(raw, output_dir / f"{name}.go.csv")
        if not has_go:
            # No go.csv (e.g. scFoundation): GEARS' GO data lives in gene2go.pkl,
            # used by split/additive. Placeholder keeps the data.go output contract.
            (output_dir / f"{name}.go.csv").write_text("source,target,importance\n")

    print(f"preprocess: wrote {out_h5ad} ({adata.shape[0]} cells x {adata.shape[1]} genes), "
          f"{len(gene_names)} gene names, go.csv={'archive' if has_go else 'placeholder'}")


if __name__ == "__main__":
    main()
