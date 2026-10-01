#!/usr/bin/env bash
# Every side-load (reference data + pretrained weights) from scratch, hash-pinned, in
# dependency order. Idempotent: each fetcher skips what is already present and verified.
# Keeps going past a failure and lists what failed at the end (exit 1 if anything did).
#
#   pixi run fetch-all                       # everything
#   OMNI_FETCH_SKIP="uce scfoundation" pixi run fetch-all
#
# Not included: make-pert-emb-rpe1 (its Replogle RPE1 input has no pinned md5 yet).
set -uo pipefail
cd "$(dirname "$0")/.."
px() { env -u PIXI_PROJECT_MANIFEST pixi run --manifest-path "$PWD/pixi.toml" "$@"; }

steps=(
  "godata         default   fetch-godata"
  "go-essential   default   fetch-go-essential"
  "replogle       default   fetch-replogle"
  "pert-emb-gears r         make-pert-emb-gears"
  "pert-emb-k562  r         make-pert-emb-k562"
  "scfoundation   default   fetch-scfoundation-model"
  "scgpt          hf        fetch-scgpt-model-hf"
  "gene-emb-scf   scfoundation-gpu make-gene-emb-scfoundation"
  "gene-emb-scgpt scgpt-gpu        make-gene-emb-scgpt"
  "geneformer     hf        fetch-geneformer-hf"
  "scbert         hf        fetch-scbert-hf"
  "uce            omnidata  fetch-uce-model"
)
failed=()
for s in "${steps[@]}"; do
  read -r name env task <<<"$s"
  [[ " ${OMNI_FETCH_SKIP:-} " == *" $name "* ]] && { echo "== skip $name"; continue; }
  echo "== $name ($task, env $env)"
  px -e "$env" "$task" || failed+=("$name")
done
((${#failed[@]})) && { echo "fetch-all: FAILED: ${failed[*]}" >&2; exit 1; }
echo "fetch-all: all side-loads present and verified"
