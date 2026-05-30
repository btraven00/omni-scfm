#!/usr/bin/env bash
# OmniBenchmark method module: additive model (double-perturbation linear baseline).
#
# Runs the paper's run_additive_model.py VERBATIM in the GEARS env: predicts a
# double perturbation as baseline + sum of the single-gene effects. Uses GEARS
# PertData.load (hence the GEARS folder + go.csv) and the per-seed split.
#
# Double-pert only: on single-pert datasets every "double" is a single, so the
# prediction trivially equals the observed mean (leakage) — exclude it there.
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --data.go PATH              the dataset's go.csv (from `preprocess`)
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene additive prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.10 leaks into the env otherwise

REPO="$(pwd)"
SCRIPT="$REPO/vendor/paper/benchmark/src/run_additive_model.py"

output_dir="" ; data_h5ad="" ; data_go="" ; split="" ; seed=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --name)                                        shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --data.go|--data_go)                           data_go="$2";    shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "additive/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
[[ -n $data_go ]] && cp "$data_go" "$wd/data/gears_pert_data/$ds/go.csv"
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

( cd "$wd" && python "$SCRIPT" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "additive: $ds seed=$seed -> $ds.predictions.json.gz"
