"""Tests for the dashboard data helpers (omni_scfm.dashboard_data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omni_scfm.dashboard_data import (
    _swarm_1d,
    add_swarm_offsets,
    hardest_table,
    leaderboard,
    method_comparison,
    per_perturbation,
    reproduction,
    reproduction_summary,
)


def test_swarm_1d_one_offset_per_point_and_dodges_overlaps():
    y = np.array([1.0, 1.0, 1.0, 5.0])      # three coincident + one far
    off = _swarm_1d(y, bin_frac=0.5, dx=1.0)
    assert off.shape == (4,)
    # the three equal points land in one row and get distinct offsets (no overlap)
    assert len(set(off[:3])) == 3
    assert 0.0 in off[:3]                   # first in a row sits on the centre line
    assert off[3] == 0.0                    # lone far point centred in its own row


def test_add_swarm_offsets_is_per_group():
    df = pd.DataFrame({"m": ["a", "a", "a", "b"], "v": [1.0, 1.0, 1.0, 2.0]})
    out = add_swarm_offsets(df, "v", ["m"])
    assert "swarm" in out.columns and len(out) == 4
    assert out.loc[out.m == "a", "swarm"].nunique() == 3   # dodged within group a
    assert out.loc[out.m == "b", "swarm"].iloc[0] == 0.0   # singleton centred
    # original columns preserved, input not mutated
    assert "swarm" not in df.columns


def _scores():
    return pd.DataFrame([
        {"dataset": "adamson", "seed": 1, "method": "mean", "perturbation": "A+ctrl",
         "split": "test", "pearson_delta": 0.8, "pearson": 0.99, "l2": 3.0},
        {"dataset": "adamson", "seed": 1, "method": "mean", "perturbation": "B+ctrl",
         "split": "test", "pearson_delta": 0.6, "pearson": 0.98, "l2": 4.0},
        {"dataset": "adamson", "seed": 1, "method": "lpm_selftrained", "perturbation": "A+ctrl",
         "split": "test", "pearson_delta": 0.9, "pearson": 0.99, "l2": 2.0},
        {"dataset": "adamson", "seed": 1, "method": "mean", "perturbation": "C+ctrl",
         "split": "train", "pearson_delta": 0.5, "pearson": 0.9, "l2": 5.0},
    ])


def test_leaderboard_ranks_by_median_test():
    lb = leaderboard(_scores(), split="test")
    assert list(lb["method"]) == ["lpm_selftrained", "mean"]   # 0.9 > median(0.8,0.6)=0.7
    mean_row = lb[lb["method"] == "mean"].iloc[0]
    assert mean_row["n"] == 2 and mean_row["median"] == pytest.approx(0.7)


def test_leaderboard_excludes_other_splits():
    lb = leaderboard(_scores(), split="test")
    assert lb[lb["method"] == "mean"]["n"].iloc[0] == 2   # the train row excluded


def test_reproduction_join_and_summary():
    scores = _scores()
    published = pd.DataFrame([
        {"dataset": "adamson", "seed": 1, "method": "mean", "perturbation": "A+ctrl",
         "pearson_delta": 0.8000005},
        {"dataset": "adamson", "seed": 1, "method": "mean", "perturbation": "B+ctrl",
         "pearson_delta": 0.6000003},
    ])
    rep = reproduction(scores, published)
    assert len(rep) == 2 and rep["abs_diff"].max() < 1e-6
    summ = reproduction_summary(rep)
    row = summ.iloc[0]
    assert row["method"] == "mean" and row["n"] == 2
    assert row["max_abs_diff"] < 1e-6


def test_per_perturbation_averages_seeds():
    df = _scores().copy()
    # add a seed-2 row for mean/A+ctrl so the average is exercised
    extra = df[(df.method == "mean") & (df.perturbation == "A+ctrl")].copy()
    extra["seed"] = 2
    extra["pearson_delta"] = 0.6
    pp = per_perturbation(pd.concat([df, extra], ignore_index=True), split="test")
    a = pp[(pp.method == "mean") & (pp.perturbation == "A+ctrl")].iloc[0]
    assert a["pearson_delta"] == pytest.approx(0.7)   # mean(0.8, 0.6)


def test_hardest_table_sorted_ascending_with_mean():
    pp = per_perturbation(_scores(), split="test")
    ht = hardest_table(pp)
    assert list(ht["mean"]) == sorted(ht["mean"])     # hardest (lowest) first
    assert "mean" in ht.columns and "mean" in set(ht.columns) - {"dataset", "perturbation"}


def test_method_comparison_pairs_and_picks_winner():
    comp = method_comparison(_scores(), "mean", "lpm_selftrained", split="test")
    # only A+ctrl is scored by both methods on the test split
    assert list(comp["perturbation"]) == ["A+ctrl"]
    row = comp.iloc[0]
    assert row["x"] == 0.8 and row["y"] == 0.9
    assert row["winner"] == "lpm_selftrained"   # 0.9 > 0.8


def test_reproduction_empty_when_no_overlap():
    scores = _scores()
    published = pd.DataFrame([
        {"dataset": "norman", "seed": 1, "method": "gears", "perturbation": "X+Y",
         "pearson_delta": 0.5},
    ])
    assert reproduction(scores, published).empty
    assert reproduction_summary(reproduction(scores, published)).empty
