"""publish_results merges runs: a later run replaces rows with the same key, others stay."""
import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "publish_results", Path(__file__).resolve().parents[1] / "scripts" / "publish_results.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)


def test_merge_later_run_wins(tmp_path):
    row = dict(dataset="adamson", seed=1, perturbation="A+ctrl", split="test")
    old = pd.DataFrame([{**row, "run_id": "r1", "method": "mean", "pearson_delta": 0.1},
                        {**row, "run_id": "r1", "method": "gears", "pearson_delta": 0.2}])
    new = pd.DataFrame([{**row, "run_id": "r2", "method": "gears", "pearson_delta": 0.9},
                        {**row, "run_id": "r2", "method": "geneformer", "pearson_delta": 0.5}])
    old.to_parquet(tmp_path / "old.parquet"); new.to_parquet(tmp_path / "new.parquet")
    m = pr._merge([tmp_path / "old.parquet", tmp_path / "new.parquet"], pr.KEYS["scores.parquet"])
    got = m.set_index("method")[["run_id", "pearson_delta"]].to_dict("index")
    assert got == {"mean": {"run_id": "r1", "pearson_delta": 0.1},
                   "gears": {"run_id": "r2", "pearson_delta": 0.9},
                   "geneformer": {"run_id": "r2", "pearson_delta": 0.5}}
