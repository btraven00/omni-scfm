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
import random
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


def _has_r_pkg(env_bin: Path, pkg: str) -> bool:
    r = subprocess.run([str(env_bin / "Rscript"), "-e",
                        f'quit(status = !requireNamespace("{pkg}", quietly = TRUE))'],
                       capture_output=True)
    return r.returncode == 0


def _gene2go_dir() -> Path | None:
    for c in (REPO / "data" / "godata",  # the side-load (pixi run fetch-godata)
              REPO / "scratch" / "scf" / "pertdata",
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
@pytest.mark.parametrize("method", ["lpm_randomPertEmb", "lpm_randomGeneEmb"])
def test_lpm_random_embedding_entrypoints(method):
    """The two nulls: same lpm script, one embedding replaced by rnorm. No side-load."""
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    out, proc = _run(method, rb, [])
    _assert_predictions(out, proc)


@pytest.mark.integration
def test_lpm_pert_embedding_entrypoint(tmp_path):
    """The precomputed-embedding path (lpm_{gears,k562,rpe1}PertEmb all share it): a tsv
    whose header is the perturbation names. Synthesised here — the real ones need the
    Dataverse side-loads — so this guards the wiring, not the embedding's content."""
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    conds = sorted({c.split("+ctrl")[0] for c in
                    json.loads((FIX / f"{NAME}.set2conditions.json").read_text())["train"]})
    rng = random.Random(0)
    tsv = tmp_path / "emb.tsv"
    tsv.write_text("\t".join(conds) + "\n" +
                   "\n".join("\t".join(f"{rng.gauss(0, 1):.6f}" for _ in conds)
                             for _ in range(10)) + "\n")
    out, proc = _run("lpm_k562PertEmb", rb, [], extra_env={"OMNI_PERT_EMB": str(tsv)})
    _assert_predictions(out, proc)


@pytest.mark.integration
def test_lpm_gene_embedding_entrypoint(tmp_path):
    """The precomputed GENE-embedding path (lpm_{scgpt,scFoundation}GeneEmb share it): a
    tsv whose header is gene names, rows = dims. Synthesised (the real ones are built from
    the model checkpoints), half the panel missing — like a real model vocabulary."""
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    genes = json.loads((FIX / f"{NAME}.gene_names.json").read_text())[::2] + ["NOT_A_GENE"]
    rng = random.Random(0)
    tsv = tmp_path / "emb.tsv"
    tsv.write_text("\t".join(genes) + "\n" +
                   "\n".join("\t".join(f"{rng.gauss(0, 1):.6f}" for _ in genes)
                             for _ in range(10)) + "\n")
    out, proc = _run("lpm_scgptGeneEmb", rb, [], extra_env={"OMNI_GENE_EMB": str(tsv)})
    # NOT the full-panel contract: run_linear_pretrained_model.R predicts only the genes
    # the embedding covers (paper behaviour, e.g. 4399/5060 adamson genes for scGPT), and
    # gene_names.json says which — the collector scores by name.
    assert proc.returncode == 0, f"run.sh failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    names = json.loads((out / f"{NAME}.gene_names.json").read_text())
    with gzip.open(out / f"{NAME}.predictions.json.gz", "rt") as fh:
        d = json.load(fh)
    assert d and sorted(names) == sorted(genes[:-1])
    assert all(len(v) == len(names) for v in d.values())


@pytest.mark.integration
def test_pert_embedding_builder_pca(tmp_path):
    """modules/godata/pert_embedding --kind pca on the fixture: what feeds
    lpm_{k562,rpe1}PertEmb, built from a dataset instead of Replogle. (--kind gears isn't
    covered: it needs the 338MB go_essential_all.csv side-load.)"""
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists()):
        pytest.skip("r pixi env not available")
    env = os.environ.copy()
    env["PATH"] = f"{rb}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", "modules/godata/pert_embedding/run.sh", "--output_dir", str(tmp_path),
         "--kind", "pca", "--data.h5ad", str(FIX / f"{NAME}.h5ad"), "--pca_dim", "4"],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout[-800:]}\n{proc.stderr[-800:]}"
    rows = (tmp_path / f"{NAME}.tsv").read_text().splitlines()
    assert len(rows) == 5                                  # header + pca_dim
    header = rows[0].split("\t")
    # The header must be PERTURBATION names (lpm matches them against clean_condition),
    # not gene names — that swap would silently produce an all-NA prediction table.
    sets = json.loads((FIX / f"{NAME}.set2conditions.json").read_text())
    want = {c.replace("+ctrl", "") for v in sets.values() for c in v} | {"ctrl"}
    assert set(header) == want


@pytest.mark.integration
@pytest.mark.parametrize("method,var,hint", [
    ("lpm_k562PertEmb", "OMNI_PERT_EMB", "make-pert-emb"),
    ("lpm_scgptGeneEmb", "OMNI_GENE_EMB", "make-gene-emb"),
])
def test_lpm_embedding_missing_side_load(method, var, hint):
    """Missing side-load must fail loudly (exit 3), not silently predict nothing."""
    proc = subprocess.run(
        ["bash", f"modules/methods/{method}/run.sh", "--output_dir", "/tmp/unused",
         "--data.h5ad", str(FIX / f"{NAME}.h5ad")],
        cwd=REPO, env={**os.environ, var: "/nonexistent.tsv"},
        capture_output=True, text=True)
    assert proc.returncode == 3, proc.stderr
    assert hint in proc.stderr


@pytest.mark.integration
def test_transfer_entrypoint():
    """Self-transfer: the fixture is its own reference dataset, so every perturbation
    matches and no prediction column is NA. Exercises the whole cross-dataset path
    (two h5ads staged, --reference_data threaded through) without the ~1GB Replogle
    side-load — with a real reference the unmatched columns are NA by design."""
    rb = _env_bin("r")
    if not (rb and (rb / "Rscript").exists() and (rb / "python3").exists()):
        pytest.skip("r pixi env not available")
    if not _has_r_pkg(rb, "lemur"):
        pytest.skip("r env predates the lemur dep (rebuild it: pixi install -e r)")
    out, proc = _run("transfer", rb, [], extra_env={"OMNI_REPLOGLE_H5AD": str(FIX / f"{NAME}.h5ad")})
    _assert_predictions(out, proc)
    with gzip.open(out / f"{NAME}.predictions.json.gz", "rt") as fh:
        preds = json.load(fh)
    # rjson writes R's NA as the string "NA" (metrics.py coerces it to NaN); with the
    # fixture as its own reference nothing should be unmatched.
    assert all(isinstance(v[0], float) for v in preds.values()), "unexpected NA in self-transfer"
    # The vendored script emits var_names (Ensembl); run.sh relabels to the symbols the
    # collector joins on. Without this the metrics silently come out NaN.
    assert json.loads((out / f"{NAME}.gene_names.json").read_text()) == \
        json.loads((FIX / f"{NAME}.gene_names.json").read_text())


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


# --- geneformer ----------------------------------------------------------------

def _geneformer_env_bin() -> Path | None:
    if (ov := os.environ.get("OMNI_GENEFORMER_ENV_BIN")) and (Path(ov) / "python").exists():
        return Path(ov)
    return _env_bin("geneformer-gpu")


def _geneformer_snapshot() -> Path | None:
    m = Path(os.environ.get("OMNI_GENEFORMER_HF", REPO / "data" / "geneformer" / "geneformer_hf.json"))
    if not m.exists():
        return None
    snap = Path(json.loads(m.read_text())["snapshot"])
    return snap if (snap / "gf-12L-95M-i4096" / "model.safetensors").exists() else None


@pytest.mark.integration
def test_geneformer_env_imports():
    """Env-good guard (no GPU): the pinned geneformer snapshot imports in the env with
    everything run_geneformer.py pulls from it, at the paper's transformers/gears pins."""
    gb, snap = _geneformer_env_bin(), _geneformer_snapshot()
    if gb is None:
        pytest.skip("geneformer env not available (set OMNI_GENEFORMER_ENV_BIN or build envs/geneformer-gpu.yml)")
    if snap is None:
        pytest.skip("no Geneformer snapshot (pixi run -e hf fetch-geneformer-hf)")
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(snap)})
    check = (
        "from geneformer import TranscriptomeTokenizer, Classifier, InSilicoPerturber;"
        "from geneformer import perturber_utils as pu;"
        "from geneformer.emb_extractor import get_embs;"
        "import transformers, gears.version, geneformer;"
        "assert transformers.__version__=='4.47.0', transformers.__version__;"
        "assert gears.version.__version__=='0.1.2', gears.version.__version__;"
        f"assert geneformer.__file__.startswith('{snap}'), geneformer.__file__"
    )
    r = subprocess.run([str(gb / "python"), "-c", check], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"geneformer env import failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.integration
def test_geneformer_entrypoint():
    gb = _geneformer_env_bin()
    if gb is None:
        pytest.skip("geneformer env not available")
    if not _has_cuda(gb):
        pytest.skip("no CUDA device (run_geneformer.py fine-tunes on GPU)")
    if _geneformer_snapshot() is None:
        pytest.skip("no Geneformer snapshot (pixi run -e hf fetch-geneformer-hf)")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    out, proc = _run("geneformer", gb, [], {"OMNI_GEARS_CACHE": str(cache)})
    _assert_predictions(out, proc)


# --- uce -------------------------------------------------------------------------

def _uce_env_bin() -> Path | None:
    if (ov := os.environ.get("OMNI_UCE_ENV_BIN")) and (Path(ov) / "python").exists():
        return Path(ov)
    return _env_bin("uce-blackwell") or _env_bin("uce-gpu")


def _uce_model_files(ckpt: str = "4layer_model.torch") -> Path | None:
    p = Path(os.environ.get("OMNI_UCE_MODEL_FILES", REPO / "data" / "uce" / "model_files"))
    return p if (p / ckpt).exists() and (p / "protein_embeddings").is_dir() else None


def _run_uce(*args: str, env_extra: dict | None = None):
    env = os.environ.copy()
    env.update(env_extra or {})
    return subprocess.run(["bash", "modules/methods/uce/run.sh", *args],
                          cwd=REPO, env=env, capture_output=True, text=True)


def test_uce_patch_targets_exist():
    """run.sh sed-patches three cluster paths in the vendored run_uce.py; if a submodule
    bump moves them the patch silently no-ops, so pin that they are there."""
    src = (REPO / "vendor" / "paper" / "benchmark" / "src" / "run_uce.py").read_text()
    assert "cd /home/ahlmanne/prog/UCE" in src
    assert "/home/ahlmanne/data/universal_cell_embedding/4layer_model.torch" in src
    assert "/home/ahlmanne/data/universal_cell_embedding/33l_8ep_1024t_1280.torch" in src


def test_uce_rejects_bad_model_type():
    proc = _run_uce("--output_dir", "/tmp/x", "--data.h5ad", "x.h5ad",
                    "--split.set2conditions", "s.json", "--model_type", "12layers")
    assert proc.returncode == 2 and "4layers or 33layers" in proc.stderr, proc.stderr


def test_uce_reports_missing_model_files():
    tmp = Path(tempfile.mkdtemp())
    proc = _run_uce("--output_dir", str(tmp / "out" / "x"), "--data.h5ad", "x.h5ad",
                    "--split.set2conditions", "s.json", "--model_type", "33layers",
                    env_extra={"OMNI_UCE_MODEL_FILES": str(tmp)})
    assert proc.returncode == 3 and "fetch-uce-model" in proc.stderr, proc.stderr


@pytest.mark.integration
def test_uce_env_imports():
    """Env-good guard (no GPU/weights): the vendored UCE modules import in the env."""
    ub = _uce_env_bin()
    if ub is None:
        pytest.skip("uce env not available (set OMNI_UCE_ENV_BIN or build envs/uce-*.yml)")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    check = ("import evaluate, model, eval_data, utils, accelerate, scanpy, gears.version;"
             "assert gears.version.__version__=='0.1.2'")
    r = subprocess.run([str(ub / "python"), "-c", check], cwd=REPO / "vendor" / "uce",
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"uce env import failed:\n{r.stdout}\n{r.stderr}"


@pytest.mark.integration
def test_uce_entrypoint():
    ub = _uce_env_bin()
    if ub is None:
        pytest.skip("uce env not available")
    if not _has_cuda(ub):
        pytest.skip("no CUDA device (run_uce.py calls torch.cuda.get_device_name())")
    mf = _uce_model_files()
    if mf is None:
        pytest.skip("no UCE model files (pixi run -e omnidata fetch-uce-model)")
    cache = _gene2go_dir()
    if cache is None:
        pytest.skip("no GEARS gene2go cache (scratch/scf/pertdata)")
    # batch 25 fits a 24GB card (the paper's 100 needs ~80GB); embeddings are batch-independent
    out, proc = _run("uce", ub, ["--model_type", "4layers"],
                     {"OMNI_GEARS_CACHE": str(cache), "OMNI_UCE_MODEL_FILES": str(mf),
                      "OMNI_UCE_BATCH": os.environ.get("OMNI_UCE_BATCH", "25")})
    _assert_predictions(out, proc)
