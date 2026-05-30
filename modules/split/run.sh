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
#   --data.raw PATH   the GEARS dataset archive from `download` (<ds>/perturb_processed.h5ad, go.csv)
#   --seed INT        split seed (paper: 1,2 single-pert; 1..5 double-pert)
# Output:
#   {dataset}.set2conditions.json   {"train","val","test"}
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.10 leaks into the env otherwise

REPO="$(pwd)"
PREP="$REPO/vendor/paper/benchmark/src/prepare_perturbation_data.py"

output_dir="" ; data_raw="" ; seed="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)            output_dir="$2"; shift 2 ;;
    --name)                  shift 2 ;;
    --data.raw|--data_raw)   data_raw="$2";   shift 2 ;;
    --seed)                  seed="$2";       shift 2 ;;
    *)                       shift ;;
  esac
done
[[ -n $output_dir && -n $data_raw ]] || {
  echo "split/run.sh: need --output_dir and --data.raw" >&2; exit 2; }

ds=$(basename "$data_raw"); ds="${ds%%.*}"

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data" "$wd/results"
# The GEARS archive contains "<ds>/perturb_processed.h5ad" (+ go.csv); extracting
# into the pert-data folder lets GEARS load(<ds>) use it instead of re-downloading.
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$data_raw" "$wd/data/gears_pert_data"
# Optional: reuse a cached gene2go / pert-gene graph to skip the ~9MB download.
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi

rid="result"
( cd "$wd" && python "$PREP" \
    --dataset_name "$ds" --seed "$seed" \
    --working_dir "$wd" --result_id "$rid" )

mkdir -p "$output_dir"
cp "$wd/results/$rid" "$output_dir/$ds.set2conditions.json"
echo "split: $ds seed=$seed -> $ds.set2conditions.json"
