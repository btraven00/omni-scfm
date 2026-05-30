#!/usr/bin/env bash
# Shared runner for the paper's R method scripts (mean, lpm, ...).
#
#   run_r_method.sh <vendored_script.R> --output_dir D --name N \
#       --data.h5ad H --split.set2conditions S [--seed K]
#
# Runs the vendored script VERBATIM via wrapper.R (which swaps the reader to
# picklerick), translating OmniBenchmark's CLI into the GEARS working-dir layout
# the scripts expect, then dedups + gzips the predictions. Adding another R method
# is a one-line wrapper that calls this with its script name.
#
# Reads the preprocess (anndata-rewritten) h5ad — picklerick can read that but not
# the legacy GEARS archive h5ad. Lineage: download->preprocess->split->methods.
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local site-packages leak into the conda env otherwise

script_name="$1"; shift
REPO="$(pwd)"                                   # OB runs entrypoints from the repo root
WRAPPER="$REPO/modules/methods/wrapper.R"
VENDORED="$REPO/vendor/paper/benchmark/src/$script_name"

output_dir="" ; data_h5ad="" ; split="" ; seed=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --name)                                        shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "run_r_method.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"

# Seed: explicit --seed wins, else read the split output's sibling parameters.json
# (OB records {"seed": N} there); default 1.
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
# The preprocess h5ad (anndata-rewritten) is what the scripts read via picklerick;
# place it where they look: data/gears_pert_data/<ds>/perturb_processed.h5ad.
ln -sf "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

export OMNI_VENDORED_SCRIPT="$VENDORED"
( cd "$wd" && Rscript "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" )

mkdir -p "$output_dir"
# Collapse any duplicate keys (e.g. mean's recycled names) and gzip; keep names plain.
python3 - "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz" <<'PY'
import gzip, json, sys
with open(sys.argv[1]) as f:
    preds = json.load(f)
with gzip.open(sys.argv[2], "wt", encoding="utf8") as g:
    json.dump(preds, g)
PY
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "$script_name: $ds seed=$seed -> $ds.predictions.json.gz"
