"""Collect per-condition metrics from an OmniBenchmark ``out/`` tree.

Walks the output directory, pairs each method's predictions with the matching
``ground_truth`` (the scoring target) and split, and writes a tidy
``scores.parquet``. Provenance comes from OB's own layout — no extra sidecars:

  dataset : the ``{dataset}.predictions.json`` filename stem
  method  : the module directory (``.../<stage>/<module>/.<hash>/<file>``)
  seed    : the nearest ancestor ``parameters.json`` carrying a ``seed`` key
            (``ground_truth`` has no split ancestor -> seed-independent)

This is the deliberately simple "gather 1+ runs, merge to parquet" collector;
the metric maths lives in :mod:`omni_scfm.metrics` so it can later be wrapped in
an OmniBenchmark gather stage unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from .metrics import as_gene_map, condition_metrics, normalize_condition, top_expressed_genes

_PRED_SUFFIXES = (".predictions.json.gz", ".predictions.json")

# Methods only meaningful for multi-gene (double) perturbations: their prediction
# for a single-gene perturbation is just that single's observed mean (leakage).
DOUBLE_ONLY_METHODS = {"additive"}


def _is_single(condition: str) -> bool:
    """A single-gene (or control) perturbation, e.g. 'GENE+ctrl' or 'ctrl'."""
    return "ctrl" in condition.split("+")


def _read_json(path: Path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as fh:
        return json.load(fh)


def _seed_for(path: Path) -> int | None:
    for parent in path.parents:
        pj = parent / "parameters.json"
        if pj.exists():
            try:
                params = _read_json(pj)
            except json.JSONDecodeError:
                continue
            if isinstance(params, dict) and "seed" in params:
                return int(params["seed"])
    return None


def _method_for(pred_path: Path) -> str:
    # .../<stage>/<module>/.<hash>/<dataset>.predictions.json
    return pred_path.parent.parent.name


def _dataset_for(path: Path) -> str:
    return path.name.split(".")[0]


def _pred_stem(name: str) -> str:
    for suf in _PRED_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name.split(".")[0]


def _load_predmap(pred_path: Path) -> dict[str, dict[str, float]]:
    preds = _read_json(pred_path)
    gn_path = pred_path.with_name(_pred_stem(pred_path.name) + ".gene_names.json")
    gene_names = _read_json(gn_path)
    # normalize_condition collapses duplicate keys (e.g. mean's recycled names).
    return {normalize_condition(k): as_gene_map(v, gene_names) for k, v in preds.items()}


def _run_id(out_dir: Path) -> str | None:
    """OmniBenchmark's per-run id from ``<out>/.metadata/manifest.json``.

    Tagging each row with it lets several runs' ``scores.parquet`` be concatenated
    and aggregated/compared by ``run_id`` (the dashboard merges on it).
    """
    manifest = out_dir / ".metadata" / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text()).get("run_id")
    except (json.JSONDecodeError, OSError):
        return None


def collect_scores(out_dir: str | Path, top_n: int = 1000) -> pd.DataFrame:
    out = Path(out_dir)
    run_id = _run_id(out)
    # Accept both gzipped and plain prediction files.
    pred_files = list(out.rglob("*.predictions.json.gz")) + list(out.rglob("*.predictions.json"))

    ground_truth: dict[str, dict] = {}
    method_files: list[tuple[Path, str, str, int | None]] = []
    for p in pred_files:
        method, ds, seed = _method_for(p), _dataset_for(p), _seed_for(p)
        if method == "ground_truth":
            ground_truth[ds] = _load_predmap(p)
        else:
            method_files.append((p, ds, method, seed))

    # split membership per (dataset, seed): condition -> 'train'|'val'|'test'
    split_labels: dict[tuple[str, int | None], dict[str, str]] = {}
    for sp in out.rglob("*.set2conditions.json"):
        ds, seed = _dataset_for(sp), _seed_for(sp)
        labels: dict[str, str] = {}
        for label, conds in _read_json(sp).items():
            for c in conds:
                labels[normalize_condition(c)] = label
        split_labels[(ds, seed)] = labels

    rows = []
    for p, ds, method, seed in method_files:
        observed = ground_truth.get(ds)
        if not observed or "ctrl" not in observed:
            continue
        baseline = observed["ctrl"]
        top_genes = top_expressed_genes(baseline, top_n)
        labels = split_labels.get((ds, seed), {})
        double_only = method in DOUBLE_ONLY_METHODS
        for cond, pred_map in _load_predmap(p).items():
            if cond not in observed:
                continue
            if double_only and _is_single(cond):
                continue  # additive on a single-pert is leakage (== observed mean)
            m = condition_metrics(pred_map, observed[cond], baseline, top_genes)
            rows.append(
                {"run_id": run_id, "dataset": ds, "seed": seed, "method": method,
                 "perturbation": cond, "split": labels.get(cond), **m}
            )

    return pd.DataFrame(
        rows,
        columns=["run_id", "dataset", "seed", "method", "perturbation", "split",
                 "n_genes", "pearson", "pearson_delta", "l2"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect metrics from a single OmniBenchmark run's output directory"
    )
    ap.add_argument("out_dir", nargs="?", default="out",
                    help="benchmark output directory for one run (default: out)")
    ap.add_argument("-o", "--output", default=None,
                    help="parquet path (default: <out_dir>/scores.parquet)")
    ap.add_argument("--top-n", type=int, default=1000, help="top-N expressed genes (default 1000)")
    args = ap.parse_args()

    output = args.output or str(Path(args.out_dir) / "scores.parquet")
    df = collect_scores(args.out_dir, args.top_n)
    df.to_parquet(output, index=False)
    n_methods = df["method"].nunique() if len(df) else 0
    print(f"collect: {len(df)} rows, {n_methods} method(s) from {args.out_dir} -> {output}")


if __name__ == "__main__":
    main()
