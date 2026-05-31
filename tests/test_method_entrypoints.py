"""End-to-end smoke tests for the method `run.sh` entrypoints.

Runs each baseline method's wrapper on the committed tiny norman fixture
(`tests/fixtures/norman_tiny/`, ~1.4 MB) in its own pixi env and checks the
output contract: a gzipped {condition: [per-gene value]} prediction map plus a
gene_names.json, with vectors the width of the fixture's gene panel.

These are integration tests: each method needs its pixi env (`.pixi/envs/{r,gears}`)
and `additive` needs a GEARS gene2go cache. Any missing prerequisite -> skip, so
the suite stays green on a machine without the envs while still guarding the
entrypoints wherever they can actually run. GEARS is intentionally not covered
here — it trains on a GPU (device='cuda') and is validated separately.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "norman_tiny"
NAME = "norman_tiny"
N_GENES = 200  # must match the committed fixture's gene panel


def _env_bin(name: str) -> Path | None:
    p = REPO / ".pixi" / "envs" / name / "bin"
    return p if p.is_dir() else None


def _gene2go_dir() -> Path | None:
    for c in (REPO / "scratch" / "scf" / "pertdata",
              REPO / "scratch" / "gears_run" / "data" / "gears_pert_data"):
        if (c / "gene2go_all.pkl").exists():
            return c
    return None


def _run(method: str, env_bin: Path, extra_flags: list[str], extra_env: dict | None = None):
    out = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["PATH"] = f"{env_bin}:{env['PATH']}"
    if extra_env:
        env.update(extra_env)
    cmd = [
        "bash", f"modules/methods/{method}/run.sh",
        "--output_dir", str(out), "--name", NAME,
        "--data.h5ad", str(FIX / f"{NAME}.h5ad"),
        "--split.set2conditions", str(FIX / f"{NAME}.set2conditions.json"),
        "--seed", "1",
    ] + extra_flags
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    return out, proc


def _assert_predictions(out: Path, proc):
    assert proc.returncode == 0, f"run.sh failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    preds = out / f"{NAME}.predictions.json.gz"
    names = out / f"{NAME}.gene_names.json"
    assert preds.exists() and names.exists()
    with gzip.open(preds, "rt") as fh:
        d = json.load(fh)
    assert d, "empty prediction map"
    assert all(len(v) == N_GENES for v in d.values()), "prediction vectors != gene panel width"
    assert len(json.loads(names.read_text())) == N_GENES


@pytest.mark.integration
def test_mean_entrypoint():
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    out, proc = _run("mean", rb, [])
    _assert_predictions(out, proc)


@pytest.mark.integration
def test_lpm_entrypoint():
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    out, proc = _run("lpm", rb, [])
    _assert_predictions(out, proc)


@pytest.mark.integration
def test_additive_entrypoint():
    gb = _env_bin("gears")
    if not (gb and (gb / "python").exists()):
        pytest.skip("gears pixi env not available")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    out, proc = _run(
        "additive", gb,
        ["--data.go", str(FIX / f"{NAME}.go.csv")],
        extra_env={"OMNI_GEARS_CACHE": str(cache)},
    )
    _assert_predictions(out, proc)
