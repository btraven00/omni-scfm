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


def _scf_env_bin() -> Path | None:
    if (ov := os.environ.get("OMNI_SCFOUNDATION_ENV_BIN")) and (Path(ov) / "python").exists():
        return Path(ov)
    return _env_bin("scfoundation-gpu")


def _scf_ckpt() -> Path | None:
    p = Path(os.environ.get("OMNI_SCFOUNDATION_CKPT", REPO / "data" / "scfoundation" / "models.ckpt"))
    return p if p.exists() else None


@pytest.mark.integration
def test_scfoundation_env_imports():
    """Env-good guard (no GPU/checkpoint needed): the deps import and the VENDORED
    forked GEARS resolves at 0.0.2 (shadowing pip cell-gears) — what run_scfoundation.py
    asserts. Skips only if the env is absent."""
    sb = _scf_env_bin()
    if sb is None:
        pytest.skip("scfoundation env not available (set OMNI_SCFOUNDATION_ENV_BIN or build envs/scfoundation-gpu.yml)")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = (f"{REPO/'vendor'/'scfoundation'/'scfoundation_gears'}:"
                         f"{REPO/'vendor'/'scfoundation'/'model'}")
    check = (
        "import torch, torch_geometric, einops, local_attention, scanpy, gears, gears.version;"
        "assert gears.version.__version__=='0.0.2', gears.version.__version__;"
        "assert 'vendor/scfoundation' in gears.__file__, gears.__file__;"
        "print('ok')"
    )
    r = subprocess.run([str(sb / "python"), "-c", check], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"scfoundation env import failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.integration
def test_scfoundation_entrypoint():
    sb = _scf_env_bin()
    if sb is None:
        pytest.skip("scfoundation env not available")
    if not _has_cuda(sb):
        pytest.skip("no CUDA device (run_scfoundation.py trains on GPU)")
    ckpt = _scf_ckpt()
    if ckpt is None:
        pytest.skip("no scFoundation checkpoint (OMNI_SCFOUNDATION_CKPT / fetch-scfoundation-model)")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    # norman_tiny isn't scFoundation-shaped: the forked GEARS needs obs['total_count'] +
    # uns['non_zeros_gene_idx'], which are pre-baked in scFoundation's distributed
    # _withtotalcount h5ad but DROPPED by our stock 0.1.2 preprocess. Validate on a
    # scFoundation-preprocessed substrate (set OMNI_SCFOUNDATION_FIXTURE to its dir).
    fixture = os.environ.get("OMNI_SCFOUNDATION_FIXTURE")
    if not fixture:
        pytest.skip("norman_tiny lacks scFoundation fields (obs.total_count, uns.non_zeros_gene_idx); "
                    "set OMNI_SCFOUNDATION_FIXTURE to a scFoundation-preprocessed tiny dataset dir")
    fx = Path(fixture); name = fx.name
    out = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["PATH"] = f"{sb}:{env['PATH']}"
    env.update({"OMNI_GEARS_CACHE": str(cache), "OMNI_SCFOUNDATION_CKPT": str(ckpt),
                "OMNI_SCF_EPOCHS": "1"})
    proc = subprocess.run(
        ["bash", "modules/methods/scfoundation/run.sh", "--output_dir", str(out),
         "--name", name, "--data.h5ad", str(fx / f"{name}.h5ad"),
         "--data.go", str(fx / f"{name}.go.csv"),
         "--split.set2conditions", str(fx / f"{name}.set2conditions.json"), "--seed", "1"],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"run.sh failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    assert (out / f"{name}.predictions.json.gz").exists()


def _scgpt_env_bin() -> Path | None:
    if (ov := os.environ.get("OMNI_SCGPT_ENV_BIN")) and (Path(ov) / "python").exists():
        return Path(ov)
    return _env_bin("scgpt-gpu")


def _scgpt_model() -> Path | None:
    p = Path(os.environ.get("OMNI_SCGPT_MODEL", REPO / "data" / "scgpt" / "scGPT_human"))
    return p if (p / "best_model.pt").exists() else None


@pytest.mark.integration
def test_scgpt_env_imports():
    """Env-good guard (no GPU/checkpoint needed): the bit-faithful deps import and resolve
    at the paper's pins — scgpt 0.2.1 classes, the classic torchtext.vocab.Vocab API (so no
    shim is needed), cell-gears 0.0.2 (run_scgpt.py asserts it), and flash-attn 1.0.4 is
    *installed* (metadata check — importing the CUDA ext needs a GPU). For the
    norman_from_scfoundation path the VENDORED fork shadows cell-gears on sys.path, so we
    also check it resolves to 0.0.2 from vendor/scfoundation. Skips only if the env is absent."""
    sb = _scgpt_env_bin()
    if sb is None:
        pytest.skip("scgpt env not available (set OMNI_SCGPT_ENV_BIN or build envs/scgpt-gpu.yml)")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    # Mirror the module: vendored forked GEARS 0.0.2 ahead of the env's cell-gears.
    env["PYTHONPATH"] = (f"{REPO/'vendor'/'scfoundation'/'scfoundation_gears'}:"
                         f"{REPO/'vendor'/'scfoundation'/'model'}")
    check = (
        "import torch, torch_geometric, scanpy;"
        "from torchtext.vocab import Vocab;"                       # classic API run_scgpt.py imports
        "from scgpt.model import TransformerGenerator;"
        "from scgpt.tokenizer.gene_tokenizer import GeneVocab;"
        "from importlib.metadata import version;"
        "assert version('flash-attn')=='1.0.4', version('flash-attn');"
        "import gears, gears.version;"
        "assert gears.version.__version__=='0.0.2', gears.version.__version__;"
        "assert 'vendor/scfoundation' in gears.__file__, gears.__file__;"
        "print('ok')"
    )
    r = subprocess.run([str(sb / "python"), "-c", check], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"scgpt env import failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.integration
def test_scgpt_entrypoint():
    sb = _scgpt_env_bin()
    if sb is None:
        pytest.skip("scgpt env not available")
    if not _has_cuda(sb):
        pytest.skip("no CUDA device (run_scgpt.py trains on GPU)")
    model = _scgpt_model()
    if model is None:
        pytest.skip("no scGPT checkpoint (OMNI_SCGPT_MODEL / fetch-scgpt-model)")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    # Scope is norman_from_scfoundation, so the fixture must be scFoundation-shaped (the
    # forked GEARS needs obs['total_count'] + uns['non_zeros_gene_idx'] — same gate as the
    # scfoundation entrypoint test). Set OMNI_SCGPT_FIXTURE to its dir.
    fixture = os.environ.get("OMNI_SCGPT_FIXTURE") or os.environ.get("OMNI_SCFOUNDATION_FIXTURE")
    if not fixture:
        pytest.skip("norman_tiny lacks scFoundation fields; set OMNI_SCGPT_FIXTURE to a "
                    "scFoundation-preprocessed tiny dataset dir")
    fx = Path(fixture); name = fx.name
    out = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["PATH"] = f"{sb}:{env['PATH']}"
    env.update({"OMNI_GEARS_CACHE": str(cache), "OMNI_SCGPT_MODEL": str(model),
                "OMNI_SCGPT_EPOCHS": "1", "OMNI_SCGPT_BATCH": "8"})  # tiny + fast for the smoke test
    proc = subprocess.run(
        ["bash", "modules/methods/scgpt/run.sh", "--output_dir", str(out),
         "--name", name, "--data.h5ad", str(fx / f"{name}.h5ad"),
         "--data.go", str(fx / f"{name}.go.csv"),
         "--split.set2conditions", str(fx / f"{name}.set2conditions.json"), "--seed", "1"],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"run.sh failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    assert (out / f"{name}.predictions.json.gz").exists()


# --- scgpt: checkpoint resolution --------------------------------------------
# The weights arrive as an omni-huggingface manifest (json pointing into the shared HF
# cache) or as an unpacked dir. Resolving that needs no GPU, no env and no real checkpoint.

def _fake_ckpt(tmp: Path) -> Path:
    d = tmp / "scGPT_human"; d.mkdir()
    for f in ("best_model.pt", "vocab.json", "args.json"):
        (d / f).write_text("x")
    return d


def _run_scgpt(*args: str, env_extra: dict | None = None):
    env = os.environ.copy()
    env.pop("OMNI_SCGPT_MODEL", None)
    env.update(env_extra or {})
    return subprocess.run(["bash", "modules/methods/scgpt/run.sh", *args],
                          cwd=REPO, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("as_manifest", [False, True], ids=["dir", "manifest"])
def test_scgpt_resolves_checkpoint(as_manifest: bool):
    tmp = Path(tempfile.mkdtemp())
    ckpt = _fake_ckpt(tmp)
    handed = ckpt
    if as_manifest:
        handed = tmp / "scgpt_human_hf.json"
        handed.write_text(json.dumps({"repo": "perturblab/scgpt-human", "snapshot": str(ckpt)}))
    h5ad = tmp / "norman_tiny.h5ad"; h5ad.write_text("not really an h5ad")
    split = tmp / "norman_tiny.set2conditions.json"; split.write_text("{}")
    # An --output_dir under an out/ makes DATA_ROOT the empty tmp dir, so the run stops at
    # the gene2go check — which sits AFTER the checkpoint gate, i.e. resolution succeeded.
    proc = _run_scgpt("--output_dir", str(tmp / "out" / "x"), "--data.h5ad", str(h5ad),
                      "--split.set2conditions", str(split),
                      env_extra={"OMNI_SCGPT_MODEL": str(handed)})
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "gene2go.pkl missing" in proc.stderr, proc.stderr


def test_scgpt_rejects_incomplete_checkpoint():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "empty").mkdir()
    proc = _run_scgpt("--output_dir", str(tmp / "out"), "--data.h5ad", "x.h5ad",
                      "--split.set2conditions", "s.json",
                      env_extra={"OMNI_SCGPT_MODEL": str(tmp / "empty")})
    assert proc.returncode == 3 and "incomplete" in proc.stderr, proc.stderr


def test_scgpt_rejects_bogus_manifest():
    """A json without "snapshot" must fail loudly, not silently fall through."""
    tmp = Path(tempfile.mkdtemp())
    bogus = tmp / "scgpt_human_hf.json"; bogus.write_text('{"repo": "x"}')
    proc = _run_scgpt("--output_dir", str(tmp / "out"), "--data.h5ad", "x.h5ad",
                      "--split.set2conditions", "s.json",
                      env_extra={"OMNI_SCGPT_MODEL": str(bogus)})
    assert proc.returncode == 3, proc.stdout + proc.stderr
