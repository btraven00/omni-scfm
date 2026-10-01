#!/usr/bin/env bash
# Shared runner for the lpm variants that swap in a PRECOMPUTED embedding (paper:
# lpm_{gears,k562,rpe1}PertEmb swap P; lpm_{scgpt,scFoundation}GeneEmb swap G).
#
#   lpm_emb.sh pert|gene <embedding_name> --output_dir D --data.h5ad H ...
#
# Same run_linear_pretrained_model.R as lpm_selftrained, with --pert_embedding or
# --gene_embedding pointed at a tsv instead of "training_data" — the paper's
# one-variable-at-a-time test of whether a fancier P or G buys anything. The tsv is
# SIDE-LOADED (built once by modules/godata/{pert,gene}_embedding), resolved from the
# user repo like every other side-load.
#
# Perturbations missing from a P embedding predict NA by design (they get an all-NA
# column), which metrics.py coerces to NaN — same story as lpm on double perts. Genes
# missing from a G embedding are dropped from the fit (the R script matches by name).
#
# Env: OMNI_PERT_EMB / OMNI_GENE_EMB   override the embedding path outright
set -euo pipefail

kind="$1"; name="$2"; shift 2
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

case "$kind" in
  pert) emb="${OMNI_PERT_EMB:-$DATA_ROOT/data/embeddings/$name.tsv}"
        hint="\`pixi run -e r make-pert-emb-{gears,k562,rpe1}\` (modules/godata/pert_embedding)" ;;
  gene) emb="${OMNI_GENE_EMB:-$DATA_ROOT/data/embeddings/$name.tsv}"
        hint="\`pixi run -e {scgpt,scfoundation}-gpu make-gene-emb-{scgpt,scfoundation}\` (modules/godata/gene_embedding)" ;;
  *) echo "lpm_emb.sh: first argument must be pert or gene (got '$kind')" >&2; exit 2 ;;
esac
[[ -f $emb ]] || {
  echo "lpm_emb.sh: no $kind embedding at '$emb' — build it with one of $hint" >&2; exit 3; }

exec bash modules/methods/run_r_method.sh run_linear_pretrained_model.R \
  "--${kind}_embedding" "$emb" "$@"
