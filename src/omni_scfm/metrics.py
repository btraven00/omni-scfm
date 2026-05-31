"""Perturbation-prediction metrics (faithful to the paper's analysis notebooks).

Per (method, perturbation), over the top-N genes ranked by control expression:
  - pearson        = cor(pred, obs)
  - pearson_delta  = cor(pred - baseline, obs - baseline)      # the headline metric
  - l2             = sqrt(sum((pred - obs)^2))
where `baseline` is the observed control (`ctrl`) mean and `obs` is the
ground-truth condition mean. Genes are aligned by name; condition names are
normalised so a method keyed by 'GENE' and ground truth keyed by 'GENE+ctrl'
match. See vendor/paper/notebooks/single_perturbation_analysis.Rmd.
"""

from __future__ import annotations

import re

import numpy as np

GeneMap = dict[str, float]


def normalize_condition(cond: str) -> str:
    """Canonicalise a condition name (mirrors the notebooks' perturbation_split).

    Split on '+'/'_' (max 2 parts): all-ctrl/empty -> 'ctrl'; two parts kept;
    a single gene gets '+ctrl' appended. So 'GENE' and 'GENE+ctrl' both ->
    'GENE+ctrl', 'A+B' -> 'A+B', 'ctrl' -> 'ctrl'.
    """
    parts = re.split(r"[+_]", cond, maxsplit=1)
    if all(p in ("ctrl", "") for p in parts):
        return "ctrl"
    if len(parts) == 2:
        return "+".join(parts)
    return "+".join(parts + ["ctrl"])


def as_gene_map(values: list[float], gene_names: list[str]) -> GeneMap:
    """Pair a value vector with its gene names (last wins on duplicate names)."""
    return dict(zip(gene_names, values))


def top_expressed_genes(baseline: GeneMap, n: int = 1000) -> list[str]:
    """Top-`n` gene names by descending control expression (ties: input order)."""
    return [g for g, _ in sorted(baseline.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def _as_float_array(m: GeneMap, genes: list[str]) -> np.ndarray:
    """Pull `genes` from a gene map as floats; non-numeric entries -> NaN.

    Methods can legitimately emit `NA`/null predictions (e.g. lpm on conditions
    it cannot embed). R serialises `NA` as the string ``"NA"``, so a value may be
    a string here; coerce to NaN rather than crashing. Any NaN then propagates to
    a NaN metric (R's `cor(use="everything")` does the same), excluding the
    condition from leaderboards without dropping the whole run.
    """
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    return np.array([f(m[x]) for x in genes], dtype=float)


def _cor(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def condition_metrics(
    pred: GeneMap, obs: GeneMap, baseline: GeneMap, genes: list[str]
) -> dict[str, float]:
    """Pearson / Pearson-delta / L2 for one condition over `genes` (intersection)."""
    g = [x for x in genes if x in pred and x in obs and x in baseline]
    p = _as_float_array(pred, g)
    o = _as_float_array(obs, g)
    b = _as_float_array(baseline, g)
    return {
        "n_genes": len(g),
        "pearson": _cor(p, o),
        "pearson_delta": _cor(p - b, o - b),
        "l2": float(np.sqrt(np.sum((p - o) ** 2))) if len(g) else float("nan"),
    }
