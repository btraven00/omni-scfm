#!/usr/bin/env bash
# Regenerate envs/gears.yml with the FULL pinned pip closure.
#
# `pixi workspace export conda-environment` writes only the *direct* deps, so
# conda's pip re-resolution at `ob run` time can drop transitive ones (e.g.
# typing_extensions) and fail. We instead pin the entire `pip freeze` of the
# solved pixi gears env so the OB-built conda env matches it exactly.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

{
  echo "name: gears"
  echo "channels:"
  echo "- conda-forge"
  echo "- bioconda"
  echo "dependencies:"
  echo "- python 3.10.*"
  echo "- pip"
  echo "- pip:"
  echo "  # CPU torch index (avoids the ~2GB CUDA wheel)."
  echo "  - --extra-index-url https://download.pytorch.org/whl/cpu"
  pixi run -e gears pip freeze \
    | grep -E '==' | grep -v '@ file' | grep -vE '^(pip|setuptools|wheel)==' \
    | sed 's/^/  - /'
} > envs/gears.yml

echo "wrote envs/gears.yml ($(grep -c '  - ' envs/gears.yml) pip pins)"
