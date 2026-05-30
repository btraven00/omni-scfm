"""Method implementations and shared helpers.

The ground-truth "method" reports observed per-condition mean expression — the
target every predictive method is scored against. Faithful to
``vendor/paper/benchmark/src/run_ground_truth_for_combinatorial_perturbations.py``
(reads expression directly instead of via GEARS PertData).
"""

from __future__ import annotations

import numpy as np

try:  # SciPy is pulled in by scanpy/anndata; guard just in case.
    from scipy.sparse import issparse
except Exception:  # pragma: no cover
    def issparse(_):  # type: ignore
        return False


def gene_names_from(adata) -> list[str]:
    """Gene symbols in X-column order (var.gene_name, else var_names)."""
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def observed_condition_stats(adata):
    """Observed mean (and standard error) of expression per condition.

    Mirrors the reference script: ``mean = X.mean(0)`` per condition and
    ``se = X.std(0) / n_cells``. Returns ``(means, se, n_cells)`` dicts keyed by
    the full GEARS condition string (e.g. ``"GENE+ctrl"``, ``"ctrl"``).
    """
    cond = adata.obs["condition"].astype(str).to_numpy()
    conditions = list(dict.fromkeys(cond))  # unique, preserving first-seen order

    means: dict[str, list[float]] = {}
    se: dict[str, list[float]] = {}
    n_cells: dict[str, int] = {}
    for c in conditions:
        sub = _dense(adata[cond == c].X)
        n = sub.shape[0]
        means[c] = np.asarray(sub.mean(axis=0)).ravel().tolist()
        se[c] = (np.asarray(sub.std(axis=0)).ravel() / n).tolist()
        n_cells[c] = int(n)
    return means, se, n_cells
