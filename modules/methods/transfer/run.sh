#!/usr/bin/env bash
# OmniBenchmark method module: transfer perturbation prediction (cross-dataset linear).
#
# Runs run_transfer_perturbation_prediction.R VERBATIM via the shared R runner.
# Pseudobulks the target and a REFERENCE dataset, PCAs each, ridge-regresses the
# target's embedding on the reference's over perturbations seen in both (training
# conditions only), then decodes held-out perturbations through that map:
# "gene X did this in Replogle K562, so it should do that here".
#
# CPU only (pseudobulk + PCA + ridge; no torch). The reference is big — Replogle
# K562 essential is ~162k cells — so allow a few GB of RAM.
#
# SINGLE-PERT ONLY: perturbations are matched by target-gene name, and a double
# "A+B" never matches Replogle's singles, so on a double-pert dataset every
# prediction column stays NA. Scoped to adamson in benchmark.yaml.
#
# The reference is SIDE-LOADED (`pixi run fetch-replogle`), not an OB input: it is a
# second dataset feeding every target's lineage, which OB 0.5.1 can't wire (same
# reason as gene2go — see modules/godata/gene2go/run.sh).
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env:
#   OMNI_REPLOGLE_H5AD          reference h5ad (default data/replogle/replogle_k562_essential.h5ad)
set -euo pipefail

REPO="$(pwd)"

# Side-loaded data lives in the USER repo (parent of out/), not the OB-staged module
# cwd — recover it from --output_dir, falling back to $REPO for local runs (AGENTS.md).
output_dir="" ; data_h5ad=""
for ((i = 1; i <= $#; i++)); do
  j=$((i + 1))
  case "${!i}" in
    --output_dir)            output_dir="${!j}" ;;
    --data.h5ad|--data_h5ad) data_h5ad="${!j}"  ;;
  esac
done
DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"

ref="${OMNI_REPLOGLE_H5AD:-$DATA_ROOT/data/replogle/replogle_k562_essential.h5ad}"
[[ -f $ref ]] || {
  echo "transfer/run.sh: no reference dataset at '$ref' — run \`pixi run fetch-replogle\`" \
       "(or set OMNI_REPLOGLE_H5AD)" >&2; exit 3; }

bash modules/methods/run_r_method.sh run_transfer_perturbation_prediction.R \
  --reference_h5ad "$ref" "$@"

# Relabel genes to symbols. Unlike the other vendored R scripts, this one never does
# `rownames(sce) <- rowData(sce)$gene_name`, so it emits the h5ad's var_names (Ensembl
# IDs) while ground_truth + every other method emit symbols — and the collector joins
# predictions to truth by gene NAME, so leaving them would silently score NaN.
# Positional: pred rows are psce's, i.e. all genes in X-column order, which is exactly
# the order of preprocess's sibling gene_names.json. Length mismatch => refuse.
ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
python3 - "${data_h5ad%.h5ad}.gene_names.json" "$output_dir/$ds.gene_names.json" <<'PY'
import json, sys
ref, emitted = sys.argv[1], sys.argv[2]
try:
    with open(ref) as fh:
        symbols = json.load(fh)
except OSError as e:
    sys.exit(f"transfer: cannot relabel genes, no preprocess gene_names.json ({e})")
with open(emitted) as fh:
    got = json.load(fh)
if len(got) != len(symbols):
    sys.exit(f"transfer: gene count {len(got)} != preprocess {len(symbols)}; refusing to relabel")
if got != symbols:
    with open(emitted, "w") as fh:
        json.dump(symbols, fh)
    print(f"transfer: relabelled {len(symbols)} genes to symbols ({got[0]} -> {symbols[0]})")
PY
