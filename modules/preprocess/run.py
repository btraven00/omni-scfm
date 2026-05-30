#!/usr/bin/env python3
"""OmniBenchmark module: preprocess a downloaded GEARS dataset.

Seed-independent. Takes the raw archive from the `download` stage (a GEARS
`*.zip` containing `<name>/perturb_processed.h5ad`, `go.csv`) and emits:

  - {name}.h5ad             the processed expression AnnData, re-written via
                            anndata. This adds the modern root `encoding-type`
                            attribute the GEARS-era file lacks, so picklerick can
                            read it; GEARS/scanpy and our metrics read it identically.
  - {name}.gene_names.json  gene symbols in X-column order (from var.gene_name)
  - {name}.go.csv           the dataset's GO table, passed through for the split

TODO: once picklerick reads legacy AnnData h5ad (missing root encoding-type, e.g.
GEARS'), the downstream R methods could read the original archive h5ad directly
and this anndata re-write + go.csv passthrough could be dropped (simpler DAG).

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


def main() -> None:
    args = parse_ob_args()
    output_dir = Path(require(args, "output_dir"))
    raw = Path(require(args, "data.raw", "data_raw"))
    name = dataset_name(args, "data.raw", "data_raw")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        h5ad = _find_h5ad(raw, Path(tmp))
        adata = ad.read_h5ad(h5ad)

        # Canonicalize condition names (sort genes within each), matching the
        # paper's normalize_condition_names, so double-pert names are order-
        # independent (A+B == B+A) and consistent across split / ground_truth /
        # methods. No-op for single-pert datasets (gene symbols sort before 'ctrl').
        cond = adata.obs["condition"].astype(str)
        adata.obs["condition"] = pd.Categorical(["+".join(sorted(c.split("+"))) for c in cond])

        gene_names = (
            adata.var["gene_name"].astype(str).tolist()
            if "gene_name" in adata.var.columns
            else adata.var_names.astype(str).tolist()
        )

        out_h5ad = output_dir / f"{name}.h5ad"
        adata.write_h5ad(out_h5ad)  # adds modern encoding-type attrs (picklerick-readable)
        with open(output_dir / f"{name}.gene_names.json", "w", encoding="utf8") as fh:
            json.dump(gene_names, fh)
        has_go = _extract_go(raw, output_dir / f"{name}.go.csv")

    print(f"preprocess: wrote {out_h5ad} ({adata.shape[0]} cells x {adata.shape[1]} genes), "
          f"{len(gene_names)} gene names, go.csv={'yes' if has_go else 'absent'}")


if __name__ == "__main__":
    main()
