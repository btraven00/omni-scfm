"""Tests for the GEARS-faithful simulation split (omni_scfm.data)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from omni_scfm.data import simulation_split
from omni_scfm._vendor.gears_split import (
    DataSplitter,
    parse_any_pert,
    parse_single_pert,
)


def _toy_adata(single_genes, combos=(), cells_per_cond=3, n_genes=8, seed=0):
    """Build a minimal AnnData with a `condition` column GEARS expects.

    Conditions follow GEARS naming: singles as 'GENE+ctrl', combos as 'A+B',
    plus a 'ctrl' group.
    """
    rng = np.random.default_rng(seed)
    conds = ["ctrl"] + [f"{g}+ctrl" for g in single_genes] + list(combos)
    rows = []
    for c in conds:
        rows += [c] * cells_per_cond
    obs = pd.DataFrame({"condition": pd.Categorical(rows)})
    X = rng.random((len(rows), n_genes)).astype("float32")
    return ad.AnnData(X=X, obs=obs)


def test_split_partitions_all_single_perts():
    genes = [f"G{i}" for i in range(20)]
    adata = _toy_adata(genes)
    s = simulation_split(adata, seed=1)

    assert set(s) == {"train", "val", "test"}
    # ctrl is forced into train.
    assert "ctrl" in s["train"]
    # Every condition appears exactly once across the three sets.
    all_conds = [c for v in s.values() for c in v]
    assert len(all_conds) == len(set(all_conds)), "splits overlap"
    assert set(all_conds) == set(adata.obs["condition"].astype(str)), "missing conditions"
    # Non-trivial holdout.
    assert len(s["test"]) > 0 and len(s["val"]) > 0


def test_split_is_deterministic_per_seed():
    adata = _toy_adata([f"G{i}" for i in range(20)])
    assert simulation_split(adata, seed=1) == simulation_split(adata, seed=1)


def test_split_varies_with_seed():
    adata = _toy_adata([f"G{i}" for i in range(20)])
    assert set(simulation_split(adata, 1)["test"]) != set(simulation_split(adata, 2)["test"])


def test_split_handles_combos():
    # Mixed single + combo conditions exercise the combo_seen{0,1,2} branches.
    genes = [f"G{i}" for i in range(12)]
    combos = ["G0+G1", "G2+G3", "G4+G5", "G6+G7"]
    adata = _toy_adata(genes, combos=combos)
    s = simulation_split(adata, seed=3)
    all_conds = [c for v in s.values() for c in v]
    assert len(all_conds) == len(set(all_conds))
    assert set(all_conds) == set(adata.obs["condition"].astype(str))


def test_uses_prepare_split_defaults():
    # simulation_split must match a direct DataSplitter call with the GEARS
    # prepare_split('simulation') defaults (train_gene_set_size/combo frac=0.75).
    genes = [f"G{i}" for i in range(20)]
    adata = _toy_adata(genes)
    ours = simulation_split(adata.copy(), seed=7)

    ds = DataSplitter(adata.copy(), split_type="simulation")
    a2, _ = ds.split_data(
        train_gene_set_size=0.75, combo_seen2_train_frac=0.75, seed=7,
        test_perts=None, only_test_set_perts=False,
    )
    ref = {
        k: v.unique().tolist()
        for k, v in dict(a2.obs.groupby("split").agg({"condition": lambda x: x}).condition).items()
    }
    assert ours == ref


@pytest.mark.parametrize(
    "pert,expected",
    [("GENE+ctrl", "GENE"), ("ctrl+GENE", "GENE")],
)
def test_parse_single_pert(pert, expected):
    assert parse_single_pert(pert) == expected


def test_parse_any_pert():
    assert parse_any_pert("FOO+ctrl") == ["FOO"]
    assert parse_any_pert("A+B") == ["A", "B"]
