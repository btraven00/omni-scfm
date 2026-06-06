#!/usr/bin/env bash
# Snakemake conda POST-DEPLOY hook for envs/scgpt-gpu.yml.
#
# Snakemake runs `<envfile-basename>.post-deploy.sh` AFTER it creates the conda env, with
# that env ACTIVE ($CONDA_PREFIX pointing at it). We install FOUR packages here, out of the
# yml's pip list, because conda installs that list with full dependency resolution / build
# isolation and these four break it:
#   - cell-gears 0.0.2 / scgpt 0.2.1 / torchtext 0.14.0: over-strict pins that collide with
#     torch==1.13.1 (torchtext pins torch==1.13.0 exactly) -> ResolutionImpossible. Install
#     --no-deps; their real runtime deps are already in the env from the yml.
#   - flash-attn 1.0.4: setup.py `import torch` at build time fails under BUILD ISOLATION.
#     By now torch 1.13.1 is installed, so --no-build-isolation finds it and the env-local
#     cuda-nvcc 11.7 compiles the CUDA ext against cu117.
#
# Bit-faithful: same paper pins. If the flash-attn 1.0.4 compile fails on the box, the
# pre-authorized fallback is to relax ONLY flash-attn (a prebuilt wheel that still exposes
# FlashMHA, or set use_fast_transformer=False in run_scgpt.py) — keep the rest faithful.
set -euo pipefail

# flash-attn's setup.py locates CUDA via CUDA_HOME / nvcc on PATH. cuda-nvcc installs nvcc
# into $CONDA_PREFIX/bin and the toolkit under $CONDA_PREFIX, so point CUDA_HOME there.
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

echo "scgpt-gpu.post-deploy: installing scgpt/cell-gears/torchtext (--no-deps)"
python -m pip install --no-deps scgpt==0.2.1 cell-gears==0.0.2 torchtext==0.14.0

echo "scgpt-gpu.post-deploy: building flash-attn 1.0.4 (CUDA_HOME=$CUDA_HOME, nvcc=$(command -v nvcc || echo MISSING))"
python -m pip install --no-build-isolation --no-deps flash-attn==1.0.4

# Sanity: metadata is queryable (don't IMPORT flash-attn's CUDA ext here — env creation may
# run on a CPU-only head node; and the vendored forked GEARS that scgpt uses for
# norman_from_scfoundation shadows cell-gears at runtime via PYTHONPATH).
python -c "from importlib.metadata import version; \
assert version('flash-attn')=='1.0.4' and version('scgpt')=='0.2.1' and version('torchtext')=='0.14.0'; \
print('scgpt-gpu.post-deploy: flash-attn/scgpt/torchtext installed OK')"
