#!/usr/bin/env bash
# OmniBenchmark method module: GEARS (graph-enhanced gene-perturbation predictor).
#
# Runs the paper's run_gears.py VERBATIM in a CUDA-enabled GEARS env. GEARS trains
# a GNN per (dataset, split, seed) on GPU (device='cuda' is hardcoded in the
# script), then predicts every condition's expression vector.
#
# Staging mirrors additive: the script wants a GEARS PertData layout under
# data/gears_pert_data/<name>/ (perturb_processed.h5ad + GO data) and the per-seed
# split at results/<config>. We stage those, run with cwd=workdir, then gzip.
#
# scFoundation note: run_gears.py special-cases the *string*
# "norman_from_scfoundation" to load a forked GEARS 0.0.2 + scFoundation
# embeddings (a separate method). We are wiring *stock* GEARS on that substrate,
# so we stage it under a non-magic folder name -> the script takes its normal
# stock-0.1.2 load path. The scFoundation-embedding variant stays a future
# `scfoundation` module (needs the fork + weights).
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --data.go PATH              the dataset's go.csv (from `preprocess`)
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env knobs:
#   OMNI_GEARS_EPOCHS  training epochs (default 20, the paper's value)
#   OMNI_GEARS_CACHE   dir with GEARS gene2go *.pkl to avoid re-downloading
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene GEARS prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.10 leaks into the env otherwise

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"   # applies scipy/pandas compat shims
export OMNI_VENDORED_SCRIPT="$REPO/vendor/paper/benchmark/src/run_gears.py"
EPOCHS="${OMNI_GEARS_EPOCHS:-20}"

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
  echo "gears/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

# Internal load name: avoid the script's "norman_from_scfoundation" fork branch so
# stock GEARS 0.1.2 is used. Known GEARS names keep their name (the script's
# pert_data.load(name) path); everything else loads via data_path.
gname="$ds"
[[ $ds == "norman_from_scfoundation" ]] && gname="norman_from_scf"

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$gname" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$gname/perturb_processed.h5ad"
[[ -n $data_go ]] && cp "$data_go" "$wd/data/gears_pert_data/$gname/go.csv"
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$gname" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" --epochs "$EPOCHS" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "gears: $ds (load name=$gname) seed=$seed epochs=$EPOCHS -> $ds.predictions.json.gz"
