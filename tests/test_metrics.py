"""Tests for metric maths (omni_scfm.metrics) and the collector (omni_scfm.collect)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from omni_scfm.collect import collect_scores
from omni_scfm.metrics import (
    condition_metrics,
    normalize_condition,
    top_expressed_genes,
)


@pytest.mark.parametrize("raw,expected", [
    ("GENE", "GENE+ctrl"),
    ("GENE+ctrl", "GENE+ctrl"),
    ("ctrl", "ctrl"),
    ("A+B", "A+B"),
    ("GENE_ctrl", "GENE+ctrl"),
    # GEARS emits condition keys joined with '_' (and bare gene names for
    # singles); these must normalise to the same '+'-joined form as the
    # ground truth so predictions join correctly. See run_gears.py output.
    ("AHR_FEV", "AHR+FEV"),         # underscore-joined double
    ("BAK1_BCL2L11", "BAK1+BCL2L11"),
    ("AHR", "AHR+ctrl"),            # bare single-gene perturbation
    ("ctrl_ctrl", "ctrl"),          # all-control collapses
])
def test_normalize_condition(raw, expected):
    assert normalize_condition(raw) == expected


def test_normalize_condition_order_preserved_not_sorted():
    """normalize_condition must NOT reorder genes — 'A+B' and 'B+A' stay distinct
    as written (canonicalisation/sorting happens upstream in preprocess, not here),
    so GEARS keys map 1:1 to how the split/ground-truth name them."""
    assert normalize_condition("FEV_AHR") == "FEV+AHR"
    assert normalize_condition("AHR_FEV") == "AHR+FEV"


def test_top_expressed_genes_by_descending_baseline():
    base = {"a": 5.0, "b": 1.0, "c": 3.0, "d": 2.0}
    assert top_expressed_genes(base, 2) == ["a", "c"]


def test_condition_metrics_perfect_prediction():
    genes = ["g0", "g1", "g2", "g3"]
    obs = {g: v for g, v in zip(genes, [1.0, 2.0, 3.0, 4.0])}
    base = {g: 0.0 for g in genes}
    m = condition_metrics(obs, obs, base, genes)  # pred == obs
    assert m["n_genes"] == 4
    assert m["pearson"] == pytest.approx(1.0)
    assert m["pearson_delta"] == pytest.approx(1.0)
    assert m["l2"] == pytest.approx(0.0)


def test_condition_metrics_delta_with_nonzero_baseline():
    genes = ["g0", "g1", "g2"]
    base = {g: 1.0 for g in genes}
    obs = {"g0": 2.0, "g1": 3.0, "g2": 4.0}   # delta 1,2,3
    pred = {"g0": 2.0, "g1": 3.0, "g2": 4.0}
    m = condition_metrics(pred, obs, base, genes)
    assert m["pearson_delta"] == pytest.approx(1.0)
    assert m["l2"] == pytest.approx(0.0)


def test_condition_metrics_l2_and_scale_invariance():
    genes = ["g0", "g1", "g2"]
    base = {g: 0.0 for g in genes}
    obs = {"g0": 1.0, "g1": 2.0, "g2": 3.0}
    pred = {"g0": 2.0, "g1": 4.0, "g2": 6.0}   # 2x obs
    m = condition_metrics(pred, obs, base, genes)
    assert m["pearson"] == pytest.approx(1.0)        # correlation is scale-invariant
    assert m["l2"] == pytest.approx(math.sqrt(1 + 4 + 9))


def test_condition_metrics_zero_variance_is_nan():
    genes = ["g0", "g1"]
    base = {g: 0.0 for g in genes}
    obs = {"g0": 0.0, "g1": 0.0}                      # obs - base is constant 0
    pred = {"g0": 1.0, "g1": 2.0}
    m = condition_metrics(pred, obs, base, genes)
    assert math.isnan(m["pearson_delta"])


# --- collector on a synthetic OB out/ tree ----------------------------------

def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _build_out_tree(root: Path):
    genes = ["g0", "g1", "g2", "g3"]
    base = root / "download/adamson/.h/preprocess/gears_h5ad/.default"

    # ground_truth (seed-independent): observed means per condition
    gt = base / "ground_truth/ground_truth/.default"
    _write(gt / "adamson.predictions.json",
           {"ctrl": [0, 0, 0, 0], "A+ctrl": [1, 2, 3, 4], "B+ctrl": [4, 3, 2, 1]})
    _write(gt / "adamson.gene_names.json", genes)

    # split (seed 1): A held out as test, B/ctrl in train
    sp = base / "split/simulation/.s1"
    _write(sp / "parameters.json", {"seed": 1})
    _write(sp / "adamson.set2conditions.json",
           {"train": ["ctrl", "B+ctrl"], "val": [], "test": ["A+ctrl"]})

    # method 'mean' under that split (clean condition keys, like the R script)
    mean = sp / "methods/mean/.default"
    _write(mean / "adamson.predictions.json",
           {"ctrl": [0, 0, 0, 0], "A": [1, 2, 3, 4], "B": [0, 0, 0, 0]})
    _write(mean / "adamson.gene_names.json", genes)


def test_collect_scores_end_to_end(tmp_path):
    _build_out_tree(tmp_path)
    df = collect_scores(tmp_path)

    # ground_truth must not appear as a scored method.
    assert set(df["method"]) == {"mean"}
    assert set(df["seed"]) == {1}

    # The held-out perturbation, scored, with the right split + perfect delta.
    a = df[df["perturbation"] == "A+ctrl"].iloc[0]
    assert a["split"] == "test"
    assert a["pearson_delta"] == pytest.approx(1.0)
    assert a["n_genes"] == 4

    # mean keyed 'A' was normalised to 'A+ctrl' to match ground truth.
    assert "A+ctrl" in set(df["perturbation"])


def test_collect_reads_gzipped_predictions(tmp_path):
    # write_predictions emits .predictions.json.gz; the collector must read it.
    from omni_scfm.io import write_predictions

    genes = ["g0", "g1", "g2", "g3"]
    base = tmp_path / "download/adamson/.h/preprocess/gears_h5ad/.default"
    gt = base / "ground_truth/ground_truth/.default"
    write_predictions(gt, "adamson",
                      {"ctrl": [0, 0, 0, 0], "A+ctrl": [1, 2, 3, 4]}, genes)
    assert (gt / "adamson.predictions.json.gz").exists()

    sp = base / "split/simulation/.s1"
    _write(sp / "parameters.json", {"seed": 1})
    _write(sp / "adamson.set2conditions.json", {"train": ["ctrl"], "test": ["A+ctrl"]})
    mean = sp / "methods/mean/.default"
    write_predictions(mean, "adamson", {"ctrl": [0, 0, 0, 0], "A": [1, 2, 3, 4]}, genes)

    df = collect_scores(tmp_path)
    a = df[df["perturbation"] == "A+ctrl"].iloc[0]
    assert a["method"] == "mean" and a["split"] == "test"
    assert a["pearson_delta"] == pytest.approx(1.0)


def test_additive_scored_on_doubles_only(tmp_path):
    """A DOUBLE_ONLY method (additive) is scored on doubles, never on singles."""
    genes = ["g0", "g1", "g2", "g3"]
    base = tmp_path / "download/norman/.h/preprocess/gears_h5ad/.default"

    gt = base / "ground_truth/ground_truth/.default"
    _write(gt / "norman.predictions.json",
           {"ctrl": [0, 0, 0, 0], "A+ctrl": [1, 2, 3, 4],
            "B+ctrl": [4, 3, 2, 1], "A+B": [5, 5, 5, 5]})
    _write(gt / "norman.gene_names.json", genes)

    sp = base / "split/simulation/.s1"
    _write(sp / "parameters.json", {"seed": 1})
    _write(sp / "norman.set2conditions.json",
           {"train": ["ctrl", "A+ctrl", "B+ctrl"], "val": [], "test": ["A+B"]})

    # additive predicts every condition, incl. singles (which are leakage).
    add = sp / "methods/additive/.default"
    _write(add / "norman.predictions.json",
           {"ctrl": [0, 0, 0, 0], "A+ctrl": [1, 2, 3, 4],
            "B+ctrl": [4, 3, 2, 1], "A+B": [5, 5, 5, 5]})
    _write(add / "norman.gene_names.json", genes)

    df = collect_scores(tmp_path)
    add_df = df[df["method"] == "additive"]
    # only the double is scored; singles + ctrl are dropped.
    assert set(add_df["perturbation"]) == {"A+B"}
