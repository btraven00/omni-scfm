#!/usr/bin/env bash
# Shared runner for the lpm variants that swap in a PRECOMPUTED perturbation embedding
# (paper: lpm_gearsPertEmb / lpm_k562PertEmb / lpm_rpe1PertEmb).
#
#   lpm_pert_emb.sh <embedding_name> --output_dir D --data.h5ad H ...
#
# Same run_linear_pretrained_model.R as lpm_selftrained, with --pert_embedding pointed at
# a tsv instead of "training_data" — the paper's one-variable-at-a-time test of whether a
# fancier P buys anything. The tsv is SIDE-LOADED (built once by
# modules/godata/pert_embedding), resolved from the user repo like every other side-load.
#
# Perturbations missing from the embedding predict NA by design (they get an all-NA
# column), which metrics.py coerces to NaN — same story as lpm on double perts.
#
# Env: OMNI_PERT_EMB   override the embedding path outright
set -euo pipefail

name="$1"; shift
REPO="$(pwd)"

# Side-loaded data lives in the USER repo (parent of out/), not the OB-staged module
# cwd — recover it from --output_dir, falling back to $REPO for local runs (AGENTS.md).
output_dir=""
for ((i = 1; i <= $#; i++)); do
  j=$((i + 1))
  [[ ${!i} == "--output_dir" ]] && output_dir="${!j}"
done
DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"

emb="${OMNI_PERT_EMB:-$DATA_ROOT/data/embeddings/$name.tsv}"
[[ -f $emb ]] || {
  echo "lpm_pert_emb.sh: no perturbation embedding at '$emb' — build it with one of" \
       "\`pixi run -e r make-pert-emb-{gears,k562,rpe1}\` (modules/godata/pert_embedding)" >&2; exit 3; }

exec bash modules/methods/run_r_method.sh run_linear_pretrained_model.R \
  --pert_embedding "$emb" "$@"
