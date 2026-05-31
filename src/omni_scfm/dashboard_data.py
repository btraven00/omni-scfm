"""Pure data helpers for the dashboard (no streamlit import — unit-testable).

Our OB method-module ids (`mean`, `lpm_selftrained`, …) match the paper's method
names, so `scores.parquet` and the published reference join directly on
``(dataset, seed, method, perturbation)``.
"""

from __future__ import annotations

import numpy as np
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


def load_scatter(path: str) -> pd.DataFrame:
    """Load one or more `scatter.parquet` (per-gene Δ-vs-Δ points)."""
    if isinstance(path, (list, tuple)):
        return pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    return pd.read_parquet(path)


def _swarm_1d(y: np.ndarray, bin_frac: float = 0.02, dx: float = 1.0) -> np.ndarray:
    """Beeswarm x-offsets for 1-D `y` (Altair has no native beeswarm).

    Classic swarm placement: take points in y order and give each the x-offset of
    smallest magnitude that doesn't collide (stay `d` apart) with any already-placed
    point within `d` in y, where `d = bin_frac * y-range` is the collision diameter.
    Dense bands fan out horizontally; isolated points sit on the centre line. Returns
    one offset per input point, in input order; magnitude is in y-units (Altair's
    xOffset scale fits the spread to the band, so only relative width matters).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    offsets = np.zeros(n)
    if n == 0:
        return offsets
    finite = y[np.isfinite(y)]
    span = (finite.max() - finite.min()) if finite.size else 0.0
    d = span * bin_frac if span > 0 else 1.0
    placed: list[tuple[float, float]] = []           # (y, x) already positioned
    for idx in np.argsort(y, kind="stable"):
        yi = y[idx]
        if not np.isfinite(yi):
            continue
        near = [px for (py, px) in placed if abs(yi - py) < d]
        if not near:
            xi = 0.0
        else:
            # smallest-|x| candidate that clears every near point by >= d
            cands = sorted({0.0, *(px + s * d for px in near for s in (-1, 1))},
                           key=lambda v: (abs(v), v))
            xi = next(c for c in cands
                      if all(abs(c - px) >= d - 1e-9 for px in near))
        placed.append((yi, xi))
        offsets[idx] = xi * dx
    return offsets


def add_swarm_offsets(df: pd.DataFrame, value_col: str, group_cols: list[str],
                      out_col: str = "swarm", bin_frac: float = 0.02,
                      dx: float = 1.0) -> pd.DataFrame:
    """Add `out_col` = per-group beeswarm x-offset for `value_col` (see `_swarm_1d`)."""
    df = df.copy()
    df[out_col] = 0.0
    for _, idx in df.groupby(group_cols, observed=True, sort=False).groups.items():
        df.loc[idx, out_col] = _swarm_1d(df.loc[idx, value_col].to_numpy(), bin_frac, dx)
    return df


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


def per_perturbation(df: pd.DataFrame, split: str = "test",
                     metric: str = "pearson_delta") -> pd.DataFrame:
    """Per (dataset, method, perturbation) metric, averaged across seeds."""
    sub = df[df["split"] == split].dropna(subset=[metric])
    return (
        sub.groupby(["dataset", "method", "perturbation"])[metric]
        .mean().reset_index()
    )


def hardest_table(per_pert: pd.DataFrame, metric: str = "pearson_delta") -> pd.DataFrame:
    """Wide (dataset, perturbation) x method table + row mean, hardest first."""
    wide = per_pert.pivot_table(index=["dataset", "perturbation"],
                                columns="method", values=metric)
    wide["mean"] = wide.mean(axis=1)
    return wide.sort_values("mean").reset_index()


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
