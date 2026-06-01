#!/usr/bin/env bash
# OmniBenchmark method module: CPA (Compositional Perturbation Autoencoder).
#
# Runs the paper's run_cpa.py VERBATIM in a CUDA-enabled env. CPA trains an
# autoencoder per (dataset, split, seed) on GPU (model.train(use_gpu=True),
# max_epochs=2000 with early stopping — all hyperparams are inline in the script),
# then predicts each condition's mean expression vector.
#
# Scope: norman_from_scfoundation ONLY — the only substrate the paper reports CPA
# on (doubles). adamson / plain norman are excluded at the dataset modules in
# benchmark.yaml; the collector's DOUBLE_ONLY filter is the backstop.
#
# Staging mirrors gears: the script wants a GEARS PertData layout under
# data/gears_pert_data/<name>/ (perturb_processed.h5ad + GO data) and the per-seed
# split at results/<config>. We stage those, run with cwd=workdir, then gzip.
#
# scFoundation note: run_cpa.py special-cases the *string* "norman_from_scfoundation"
# to sys.path-insert a forked GEARS 0.0.2 (a cluster path) and assert that version.
# We run *stock* cell-gears 0.1.2 on the bit-exact-reproduced substrate, so we stage
# under a non-magic folder name -> the script takes its normal else branch and
# asserts 0.1.2 (which our env satisfies). See norman-scfoundation-additive notes.
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --data.go PATH              the dataset's go.csv (from `preprocess`)
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env knobs:
#   OMNI_CPA_CACHE   dir with GEARS gene2go *.pkl to avoid re-downloading
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene CPA prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1   # ~/.local/lib/python3.10 leaks into the env otherwise

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"   # reused: runpy runner + compat shims
export OMNI_VENDORED_SCRIPT="$REPO/vendor/paper/benchmark/src/run_cpa.py"

output_dir="" ; data_h5ad="" ; data_go="" ; split="" ; seed="" ; gene2go=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --name)                                        shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --data.go|--data_go)                           data_go="$2";    shift 2 ;;
    --godata.gene2go|--godata_gene2go|--gene2go|--gene2go_all) gene2go="$2"; shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "cpa/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

# Internal load name: avoid the script's "norman_from_scfoundation" fork branch so
# stock cell-gears 0.1.2 is used (the else branch asserts 0.1.2).
gname="$ds"
[[ $ds == "norman_from_scfoundation" ]] && gname="norman_from_scf"

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/data/gears_pert_data/$gname" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$gname/perturb_processed.h5ad"
[[ -n $data_go ]] && cp "$data_go" "$wd/data/gears_pert_data/$gname/go.csv"
# gene2go_all.pkl side-loaded (pixi run fetch-godata -> data/godata/); see split/run.sh.
if [[ -n "${OMNI_CPA_CACHE:-}" && -d "$OMNI_CPA_CACHE" ]]; then
  cp -n "$OMNI_CPA_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
gene2go="${gene2go:-$REPO/data/godata/gene2go_all.pkl}"
[[ -f $gene2go ]] && cp "$gene2go" "$wd/data/gears_pert_data/gene2go_all.pkl"
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$gname" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "cpa: $ds (load name=$gname) seed=$seed -> $ds.predictions.json.gz"
