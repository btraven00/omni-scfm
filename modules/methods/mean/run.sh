#!/usr/bin/env bash
# OmniBenchmark method module: mean prediction baseline.
#
# Runs the paper's run_mean_prediction.R VERBATIM (via wrapper.R) by recreating
# the directory layout that script expects and translating OmniBenchmark's CLI:
#   --output_dir / --name / --data.h5ad / --split.set2conditions
# into the script's --dataset_name / --test_train_config_id / --working_dir /
# --result_id.
set -euo pipefail

REPO="$(pwd)"                         # OB runs entrypoints from the repo/module root
HERE="$REPO/modules/methods/mean"

output_dir="" ; data_h5ad="" ; split=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                          output_dir="$2"; shift 2 ;;
    --name)                                shift 2 ;;
    --data.h5ad|--data_h5ad)               data_h5ad="$2";  shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2"; shift 2 ;;
    *)                                     shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "mean/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

# {dataset} wildcard from the input filename (api>=0.5: --name is the module id).
ds=$(basename "$data_h5ad"); ds="${ds%%.*}"

# Recreate the layout run_mean_prediction.R expects, under a temp working dir:
#   <wd>/data/gears_pert_data/<ds>/perturb_processed.h5ad   (the expression data)
#   <wd>/results/<cfg>                                      (the split json)
wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
ln -sf "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

export OMNI_VENDORED_SCRIPT="$REPO/vendor/paper/benchmark/src/run_mean_prediction.R"
# Run from $wd so the script's relative data/ path resolves.
( cd "$wd" && Rscript "$HERE/wrapper.R" \
    --dataset_name "$ds" \
    --test_train_config_id "$cfg" \
    --working_dir "$wd" \
    --result_id "$rid" )

mkdir -p "$output_dir"
# The vendored script writes one prediction per *gene* (replicate(nrow(psce),…)),
# i.e. ~5060 byte-identical copies of the constant profile keyed by recycled
# condition names. gzip can't span that (>32 KB window), so we collapse the
# duplicate keys first — lossless: json.load keeps the unique conditions, the
# values are unchanged — then gzip. Keep gene names plain (small).
python3 - "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz" <<'PY'
import gzip, json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    preds = json.load(f)            # duplicate keys collapse to the unique conditions
with gzip.open(dst, "wt", encoding="utf8") as g:
    json.dump(preds, g)
PY
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "mean: wrote $ds.predictions.json.gz ($(printf '%s' "$(wc -c < "$output_dir/$ds.predictions.json.gz")") bytes)"
