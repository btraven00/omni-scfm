#!/usr/bin/env bash
# Snakemake conda POST-DEPLOY hook for envs/scgpt-gpu.yml.
#
# Snakemake runs `<envfile-basename>.post-deploy.sh` AFTER it creates the conda env, with
# that env ACTIVE ($CONDA_PREFIX pointing at it). We build flash-attn 1.0.4 HERE rather
# than in the env yml because flash-attn's setup.py does `import torch` at build time, and
# the yml's pip list is built under BUILD ISOLATION (a fresh env with no torch) — which
# fails with "ModuleNotFoundError: No module named 'torch'". By the time this runs, torch
# 1.13.1 is installed in the env, so `--no-build-isolation` finds it and the env-local
# cuda-nvcc 11.7 (a conda dep of the env) compiles the CUDA extension against cu117.
#
# Bit-faithful: still flash-attn==1.0.4 (the paper's pin). If THIS compile fails on the
# box, the pre-authorized fallback is to relax ONLY flash-attn (a prebuilt wheel that still
# exposes FlashMHA, or set use_fast_transformer=False in run_scgpt.py) — keep the rest faithful.
set -euo pipefail

# flash-attn's setup.py locates CUDA via CUDA_HOME / nvcc on PATH. cuda-nvcc installs nvcc
# into $CONDA_PREFIX/bin and the toolkit under $CONDA_PREFIX, so point CUDA_HOME there.
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

echo "scgpt-gpu.post-deploy: building flash-attn 1.0.4 (CUDA_HOME=$CUDA_HOME, nvcc=$(command -v nvcc || echo MISSING))"
python -m pip install --no-build-isolation --no-deps flash-attn==1.0.4

# Sanity: the package is importable-as-metadata (the CUDA ext import itself needs a GPU at
# runtime, so don't import it here — env creation may run on a CPU-only head node).
python -c "from importlib.metadata import version; assert version('flash-attn')=='1.0.4', version('flash-attn'); print('scgpt-gpu.post-deploy: flash-attn', version('flash-attn'), 'installed')"
