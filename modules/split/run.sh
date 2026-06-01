#!/usr/bin/env bash
# OmniBenchmark split module: GEARS-faithful train/val/test split.
#
# Runs the paper's prepare_perturbation_data.py VERBATIM in the GEARS env. This is
# required for a bit-exact split: GEARS' PertData.load() GO-filters perturbations
# (drops those whose gene is not in the GO graph) BEFORE splitting, which a
# re-implementation can't reproduce without GEARS' gene2go vocabulary. The script
# also handles each dataset's split logic (Adamson 'simulation', Norman custom).
#
# Inputs (OB):
#   --data.h5ad PATH  processed AnnData from `preprocess` (GEARS reads it identically
#                     to the original archive -> bit-exact split; verified)
#   --data.go PATH    the dataset's go.csv (from `preprocess`)
#   --seed INT        split seed (paper: 1,2 single-pert; 1..5 double-pert)
# Output:
#   {dataset}.set2conditions.json   {"train","val","test"}
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.10 leaks into the env otherwise

REPO="$(pwd)"
PREP="$REPO/vendor/paper/benchmark/src/prepare_perturbation_data.py"
SCF_SPLIT="$REPO/modules/split/scf_split.py"

output_dir="" ; data_h5ad="" ; data_go="" ; seed="1" ; gene2go=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)              output_dir="$2"; shift 2 ;;
    --name)                    shift 2 ;;
    --data.h5ad|--data_h5ad)   data_h5ad="$2";  shift 2 ;;
    --data.go|--data_go)       data_go="$2";    shift 2 ;;
    --godata.gene2go|--godata_gene2go|--gene2go|--gene2go_all) gene2go="$2"; shift 2 ;;
    --seed)                    seed="$2";       shift 2 ;;
    *)                         shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad ]] || {
  echo "split/run.sh: need --output_dir and --data.h5ad" >&2; exit 2; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
# Assemble the GEARS pert-data folder so load(<ds>) uses our data (no re-download):
# the processed h5ad as perturb_processed.h5ad + the dataset's go.csv. COPY (not
# symlink): the norman branch normalizes condition names and rewrites this file,
# which through a symlink would corrupt the upstream preprocess output.
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
[[ -n $data_go ]] && cp "$data_go" "$wd/data/gears_pert_data/$ds/go.csv"
# Skip GEARS' ~9MB gene2go re-download. gene2go_all.pkl is SIDE-LOADED (fetched once
# by `pixi run fetch-godata` -> data/godata/), not an OB stage: OB 0.5.1 can't wire a
# single global artifact into per-dataset lineages (a parallel-root input is silently
# dropped from argv). Precedence: --gene2go flag > data/godata default > OMNI_GEARS_CACHE
# (the integration-test fixture, which doesn't run the full OB graph).
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
gene2go="${gene2go:-$REPO/data/godata/gene2go_all.pkl}"
[[ -f $gene2go ]] && cp "$gene2go" "$wd/data/gears_pert_data/gene2go_all.pkl"

rid="result"
# scFoundation Norman: the paper's verbatim script can't run off-cluster for this
# dataset (forked GEARS 0.0.2 + absolute /g/huber paths), so use our faithful port
# of its split branch (lines 100-115). All other datasets run the script verbatim.
case "$ds" in
  *scfoundation*) SCRIPT="$SCF_SPLIT" ;;
  *)              SCRIPT="$PREP" ;;
esac
( cd "$wd" && python "$SCRIPT" \
    --dataset_name "$ds" --seed "$seed" \
    --working_dir "$wd" --result_id "$rid" )

mkdir -p "$output_dir"
cp "$wd/results/$rid" "$output_dir/$ds.set2conditions.json"
echo "split: $ds seed=$seed -> $ds.set2conditions.json"
