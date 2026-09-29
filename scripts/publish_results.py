#!/usr/bin/env python3
"""Assemble a results-repo payload from a benchmark run.

The benchmark *code* lives in omni-scfm; the benchmark *results* are published as a
separate, versioned data artifact (github.com/btraven00/omni-scfm-results) that the
dashboard — and anyone else — consumes by URL. This script is the bridge: it copies
the run products + reference data into a target dir and writes a provenance manifest.

    python scripts/publish_results.py ../omni-scfm-results \
        [--scores out/scores.parquet ...] [--scatter out/scatter.parquet ...] \
        [--run-commit RUN_ID=SHA] [--run-note RUN_ID=TEXT]

To ADD a run to what is already published, pass the published tables first:
    --scores ../omni-scfm-results/scores.parquet other/out/scores.parquet (same for --scatter);
later files win on (dataset, seed, method, perturbation[, gene]).

Then commit + push that dir. Re-run after each `ob run` + `pixi run collect` to refresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# a row's identity within the published tables (run_id is provenance, not identity)
KEYS = {"scores.parquet": ["dataset", "seed", "method", "perturbation"],
        "scatter.parquet": ["dataset", "seed", "method", "perturbation", "gene"]}


def _merge(paths: list[str], keys: list[str]) -> pd.DataFrame:
    """Concatenate runs in order; a later run replaces an earlier one's rows (same keys)."""
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def _kv(pairs: list[str]) -> dict[str, str]:
    return dict(p.split("=", 1) for p in pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="results-repo checkout dir")
    ap.add_argument("--scores", nargs="+", default=[str(REPO / "out" / "scores.parquet")],
                    help="one or more runs' scores.parquet, merged in order (later wins)")
    ap.add_argument("--scatter", nargs="+", default=[str(REPO / "out" / "scatter.parquet")])
    ap.add_argument("--run-commit", action="append", default=[], metavar="RUN_ID=SHA",
                    help="benchmark commit that produced a run (default: previous manifest, else HEAD)")
    ap.add_argument("--run-note", action="append", default=[], metavar="RUN_ID=TEXT")
    args = ap.parse_args()

    target = Path(args.target)
    (target / "reference").mkdir(parents=True, exist_ok=True)
    prev = json.loads((target / "manifest.json").read_text()) if (target / "manifest.json").exists() else {}

    # run products (the actual results) — read everything before writing: the target's own
    # current parquet is usually one of the inputs
    tables = {"scores.parquet": _merge(args.scores, KEYS["scores.parquet"]),
              "scatter.parquet": _merge(args.scatter, KEYS["scatter.parquet"])}
    # reference inputs the dashboard overlays (paper numbers, gene network centrality)
    reference = {
        "reference/published_single.csv": REPO / "scratch" / "published" / "published_single.csv",
        "reference/published_double.csv": REPO / "scratch" / "published" / "published_double.csv",
        "reference/gene_centrality.csv": REPO / "data" / "gene_centrality.csv",
    }

    files: dict[str, dict] = {}
    for rel, df in tables.items():
        df.to_parquet(target / rel, index=False)
        files[rel] = {"bytes": (target / rel).stat().st_size, "md5": _md5(target / rel)}
    for rel, src in reference.items():
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        files[rel] = {"bytes": dst.stat().st_size, "md5": _md5(dst)}

    # provenance: the publishing code state + one entry per run in the merged table
    scores = tables["scores.parquet"]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    prev_runs = {r["run_id"]: r for r in prev.get("runs", [])}
    commits, notes = _kv(args.run_commit), _kv(args.run_note)
    runs = []
    for rid, g in scores.groupby("run_id", sort=False):
        old = prev_runs.get(rid, {})
        # a manifest from before `runs` existed described a single run by benchmark_commit
        commit = commits.get(rid) or old.get("benchmark_commit") or (
            prev.get("benchmark_commit") if not prev.get("runs") and prev else None) or head
        run = {"run_id": rid, "benchmark_commit": commit,
               "datasets": sorted(g["dataset"].unique().tolist()),
               "methods": sorted(g["method"].unique().tolist()),
               "seeds": sorted(int(s) for s in g["seed"].dropna().unique())}
        if note := notes.get(rid, old.get("note")):
            run["note"] = note
        runs.append(run)
    manifest = {
        "benchmark": "omni-scfm",
        "benchmark_repo": "https://github.com/btraven00/omni-scfm",
        "benchmark_commit": head,  # the code that published this set; per-run commits below
        "generated": date.today().isoformat(),
        "datasets": sorted(scores["dataset"].unique().tolist()),
        "methods": sorted(scores["method"].unique().tolist()),
        "seeds": sorted(int(s) for s in scores["seed"].dropna().unique()),
        "n_score_rows": int(len(scores)),
        "runs": runs,
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"published {len(files)} files to {target}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
