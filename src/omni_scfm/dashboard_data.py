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
