"""Pure data helpers for the dashboard (no streamlit import — unit-testable).

Our OB method-module ids (`mean`, `lpm_selftrained`, …) match the paper's method
names, so `scores.parquet` and the published reference join directly on
``(dataset, seed, method, perturbation)``.
"""

from __future__ import annotations

import pandas as pd

METRIC_LABELS = {
    "pearson_delta": "Pearson delta",
    "pearson": "Pearson",
    "l2": "L2",
}


def load_scores(path: str) -> pd.DataFrame:
    """Load one or more `scores.parquet` (concatenated if a list)."""
    if isinstance(path, (list, tuple)):
        return pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    return pd.read_parquet(path)


def leaderboard(df: pd.DataFrame, split: str = "test", metric: str = "pearson_delta") -> pd.DataFrame:
    """Per (dataset, method) summary of a metric over the chosen split."""
    sub = df[df["split"] == split].dropna(subset=[metric])
    out = (
        sub.groupby(["dataset", "method"])[metric]
        .agg(median="median", mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["dataset", "median"], ascending=[True, False])
    )
    return out


def method_comparison(df: pd.DataFrame, method_x: str, method_y: str,
                      metric: str = "pearson_delta", split: str = "test") -> pd.DataFrame:
    """Pair two methods' per-perturbation metric for a scatter (x vs y).

    Returns rows ``dataset, seed, perturbation, x, y, winner`` for perturbations
    both methods scored on the given split. ``winner`` is the method with the
    higher metric (the better one for pearson/pearson_delta).
    """
    sub = df[df["split"] == split].dropna(subset=[metric])
    keys = ["dataset", "seed", "perturbation"]
    x = sub[sub["method"] == method_x][keys + [metric]].rename(columns={metric: "x"})
    y = sub[sub["method"] == method_y][keys + [metric]].rename(columns={metric: "y"})
    m = x.merge(y, on=keys, how="inner")
    m["winner"] = (m["y"] > m["x"]).map({True: method_y, False: method_x})
    return m


def reproduction(scores: pd.DataFrame, published: pd.DataFrame,
                 metric: str = "pearson_delta") -> pd.DataFrame:
    """Join our scores to the paper's published numbers per perturbation.

    Returns rows with `<metric>_ours`, `<metric>_paper`, and `abs_diff`, for the
    methods/datasets present in both.
    """
    keys = ["dataset", "seed", "method", "perturbation"]
    o = scores[keys + [metric]].rename(columns={metric: f"{metric}_ours"})
    p = published[keys + [metric]].rename(columns={metric: f"{metric}_paper"})
    m = o.merge(p, on=keys, how="inner")
    m["abs_diff"] = (m[f"{metric}_ours"] - m[f"{metric}_paper"]).abs()
    return m


def reproduction_summary(repro: pd.DataFrame, metric: str = "pearson_delta") -> pd.DataFrame:
    """Per (dataset, method) agreement stats from `reproduction`."""
    if repro.empty:
        return pd.DataFrame(columns=["dataset", "method", "n", "max_abs_diff", "mean_abs_diff", "corr"])
    rows = []
    for (ds, meth), g in repro.groupby(["dataset", "method"]):
        corr = g[f"{metric}_ours"].corr(g[f"{metric}_paper"])
        rows.append({"dataset": ds, "method": meth, "n": len(g),
                     "max_abs_diff": g["abs_diff"].max(),
                     "mean_abs_diff": g["abs_diff"].mean(), "corr": corr})
    return pd.DataFrame(rows)
