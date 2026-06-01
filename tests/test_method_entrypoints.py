"""End-to-end smoke tests for the method `run.sh` entrypoints.

Runs each baseline method's wrapper on the committed tiny norman fixture
(`tests/fixtures/norman_tiny/`, ~1.4 MB) in its own pixi env and checks the
output contract: a gzipped {condition: [per-gene value]} prediction map plus a
gene_names.json, with vectors the width of the fixture's gene panel.

These are integration tests: each method needs its conda/pixi env and some need a
GEARS gene2go cache. Any missing prerequisite -> skip, so the suite stays green on a
machine without the envs while still guarding the entrypoints wherever they can
actually run. Selected/tracked via the `integration` marker (see pyproject.toml).

GPU methods are GPU-gated rather than excluded: `cpa` runs here when both a CPA env
and a CUDA device are present (the norman_tiny train is ~5s), else it skips. GEARS is
still not covered — its env is heavier and it's validated separately.

Portability note: these tests are parametrized only by method name + run.sh path +
the committed fixture, so when a module graduates to its own repo the matching test
travels with it unchanged (just move the function + the fixture dir).
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


def _cpa_env_bin() -> Path | None:
    """Canonical: the pixi env (`.pixi/envs/cpa-gpu`, from `pixi install -e cpa-gpu`).
    Override: `OMNI_CPA_ENV_BIN` for the OB run box, where OB builds envs/cpa-gpu.yml
    into a snakemake conda prefix rather than a pixi env. (No scratch fallback — see
    AGENTS.md: tests never depend on scratch/.)"""
    if (ov := os.environ.get("OMNI_CPA_ENV_BIN")) and (Path(ov) / "python").exists():
        return Path(ov)
    return _env_bin("cpa-gpu")


def _has_cuda(env_bin: Path) -> bool:
    """run_cpa.py calls torch.cuda.get_device_name() at startup, so without a GPU it
    crashes rather than skips — probe first."""
    env = os.environ.copy()
    env["PATH"] = f"{env_bin}:{env['PATH']}"
    env["PYTHONNOUSERSITE"] = "1"
    r = subprocess.run([str(env_bin / "python"), "-c",
                        "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                       env=env, capture_output=True)
    return r.returncode == 0


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


@pytest.mark.integration
def test_cpa_entrypoint():
    cb = _cpa_env_bin()
    if cb is None:
        pytest.skip("cpa env not available (set OMNI_CPA_ENV_BIN or build envs/cpa-gpu.yml)")
    if not _has_cuda(cb):
        pytest.skip("no CUDA device (run_cpa.py trains on GPU)")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    out, proc = _run(
        "cpa", cb,
        ["--data.go", str(FIX / f"{NAME}.go.csv")],
        extra_env={"OMNI_CPA_CACHE": str(cache)},
    )
    _assert_predictions(out, proc)
