"""Tests for ground-truth observed-condition statistics."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from omni_scfm.methods import gene_names_from, observed_condition_stats


def _adata(cond_to_rows, var_names=None, gene_name=None):
    rows, conds = [], []
    for c, mat in cond_to_rows.items():
        for r in mat:
            rows.append(r)
            conds.append(c)
    X = np.asarray(rows, dtype="float32")
    var = pd.DataFrame(index=var_names or [f"v{i}" for i in range(X.shape[1])])
    if gene_name is not None:
        var["gene_name"] = gene_name
    return ad.AnnData(X=X, obs=pd.DataFrame({"condition": pd.Categorical(conds)}), var=var)


def test_observed_means_and_se():
    a = _adata({
        "ctrl": [[1.0, 2.0], [3.0, 4.0]],          # mean [2,3], pop-std [1,1], n=2 -> se [.5,.5]
        "G1+ctrl": [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]],  # mean [10,0], std 0 -> se 0
    })
    means, se, n = observed_condition_stats(a)

    assert means["ctrl"] == [2.0, 3.0]
    assert se["ctrl"] == [0.5, 0.5]
    assert means["G1+ctrl"] == [10.0, 0.0]
    assert se["G1+ctrl"] == [0.0, 0.0]
    assert n == {"ctrl": 2, "G1+ctrl": 3}


def test_covers_all_conditions():
    a = _adata({"ctrl": [[1, 1]], "A+ctrl": [[2, 2]], "B+ctrl": [[3, 3]]})
    means, _, _ = observed_condition_stats(a)
    assert set(means) == {"ctrl", "A+ctrl", "B+ctrl"}
    assert all(len(v) == 2 for v in means.values())


def test_gene_names_prefers_gene_name_col():
    a = _adata({"ctrl": [[0, 0]]}, var_names=["ENS1", "ENS2"], gene_name=["SYM1", "SYM2"])
    assert gene_names_from(a) == ["SYM1", "SYM2"]


def test_gene_names_falls_back_to_var_names():
    a = _adata({"ctrl": [[0, 0]]}, var_names=["ENS1", "ENS2"])
    assert gene_names_from(a) == ["ENS1", "ENS2"]
