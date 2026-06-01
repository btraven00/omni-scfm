#!/usr/bin/env python3
"""Assemble a results-repo payload from a benchmark run.

The benchmark *code* lives in omni-scfm; the benchmark *results* are published as a
separate, versioned data artifact (github.com/btraven00/omni-scfm-results) that the
dashboard — and anyone else — consumes by URL. This script is the bridge: it copies
the run products + reference data into a target dir and writes a provenance manifest.

    python scripts/publish_results.py ../omni-scfm-results \
        [--scores out/scores.parquet] [--scatter out/scatter.parquet]

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="results-repo checkout dir")
    ap.add_argument("--scores", default=str(REPO / "out" / "scores.parquet"))
    ap.add_argument("--scatter", default=str(REPO / "out" / "scatter.parquet"))
    args = ap.parse_args()

    target = Path(args.target)
    (target / "reference").mkdir(parents=True, exist_ok=True)

    # run products (the actual results)
    products = {"scores.parquet": Path(args.scores), "scatter.parquet": Path(args.scatter)}
    # reference inputs the dashboard overlays (paper numbers, gene network centrality)
    reference = {
        "reference/published_single.csv": REPO / "scratch" / "published" / "published_single.csv",
        "reference/published_double.csv": REPO / "scratch" / "published" / "published_double.csv",
        "reference/gene_centrality.csv": REPO / "data" / "gene_centrality.csv",
    }

    files: dict[str, dict] = {}
    for rel, src in {**products, **reference}.items():
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        files[rel] = {"bytes": dst.stat().st_size, "md5": _md5(dst)}

    # provenance from the scores table + the benchmark repo state
    scores = pd.read_parquet(args.scores)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "benchmark": "omni-scfm",
        "benchmark_repo": "https://github.com/btraven00/omni-scfm",
        "benchmark_commit": commit,
        "generated": date.today().isoformat(),
        "datasets": sorted(scores["dataset"].unique().tolist()),
        "methods": sorted(scores["method"].unique().tolist()),
        "seeds": sorted(int(s) for s in scores["seed"].dropna().unique()),
        "n_score_rows": int(len(scores)),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"published {len(files)} files to {target}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
