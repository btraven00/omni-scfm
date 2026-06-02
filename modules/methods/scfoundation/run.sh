#!/usr/bin/env bash
# OmniBenchmark method module: scFoundation (foundation-model perturbation predictor).
#
# scFoundation = the forked GEARS 0.0.2 with a scFoundation `maeautobin` encoder loaded
# as the backbone (run_scfoundation.py). Trains/fine-tunes on GPU per (split, seed).
# Scoped to norman_from_scfoundation (the paper's only scFoundation substrate; that's
# what the whole `norman_from_scfoundation` dataset exists for).
#
# Unlike additive/gears/cpa there is no rename-dodge: scFoundation IS the fork, so we
# run the VENDORED fork (vendor/scfoundation/scfoundation_gears, gears.version==0.0.2)
# + encoder (vendor/scfoundation/model). The vendored run_scfoundation.py hardcodes
# three cluster paths (two sys.path inserts + the checkpoint), so — like the split's
# scfoundation port — we sed-patch ONLY those three absolute paths and run the rest
# verbatim:
#   scfoundation_gears/ -> vendor/scfoundation/scfoundation_gears
#   model/              -> vendor/scfoundation/model
#   models.ckpt         -> $OMNI_SCFOUNDATION_CKPT (side-loaded, see fetch-scfoundation-model)
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --data.go PATH              the dataset's go.csv (from `preprocess`)
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env knobs:
#   OMNI_SCFOUNDATION_CKPT  path to the scFoundation models.ckpt (default data/scfoundation/models.ckpt)
#   OMNI_SCF_EPOCHS         fine-tune epochs (default 15, the paper's value)
#   OMNI_GEARS_CACHE        dir with a gene2go *.pkl (else data/godata side-load)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene scFoundation prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"            # scipy/pandas shim + runpy
VENDORED="$REPO/vendor/paper/benchmark/src/run_scfoundation.py"
FORK="$REPO/vendor/scfoundation/scfoundation_gears"
MODEL="$REPO/vendor/scfoundation/model"
CKPT="${OMNI_SCFOUNDATION_CKPT:-$REPO/data/scfoundation/models.ckpt}"
EPOCHS="${OMNI_SCF_EPOCHS:-15}"
BATCH="${OMNI_SCF_BATCH:-6}"   # paper default 6; lower it (e.g. 1) to fit a small GPU

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
    --batch_size)                                  BATCH="$2";      shift 2 ;;  # explicit in benchmark.yaml
    --epochs)                                      EPOCHS="$2";     shift 2 ;;  # explicit in benchmark.yaml
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "scfoundation/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }
[[ -f $CKPT ]] || {
  echo "scfoundation/run.sh: checkpoint not found at '$CKPT'. Set OMNI_SCFOUNDATION_CKPT or run" >&2
  echo "  OMNI_SCFOUNDATION_URL=<http(s)|file://...> pixi run fetch-scfoundation-model" >&2; exit 3; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
# scFoundation loads via data_path = data/gears_pert_data/<ds> (no name dodge — the
# script's else-branch handles non-norman/adamson/dixit names by data_path).
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
# scFoundation needs the REAL GO perturbation graph. get_go_auto reads go.csv if present,
# else builds it from gene2go via a pure-Python O(n^2) Jaccard over ~9976 essential genes
# (~99M pairs => HOURS on CPU; the paper's 5-day SLURM cap absorbed this). So we SIDE-LOAD
# the precomputed go_essential_all.csv (the file the original GEARS read; ~338MB) and stage
# it as the dataset's go.csv -> get_go_auto just reads it (seconds). Precedence:
#   OMNI_SCF_GO env > data/godata/go_essential_all.csv (side-load) > a non-placeholder
#   --data.go (>1 line) > else the fork computes it (the slow path; avoid on a single GPU).
GO_ESSENTIAL="${OMNI_SCF_GO:-$REPO/data/godata/go_essential_all.csv}"
if [[ -f $GO_ESSENTIAL ]]; then
  cp "$GO_ESSENTIAL" "$wd/data/gears_pert_data/$ds/go.csv"
elif [[ -n $data_go && $(wc -l < "$data_go") -gt 1 ]]; then
  cp "$data_go" "$wd/data/gears_pert_data/$ds/go.csv"
fi
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
gene2go="${gene2go:-$REPO/data/godata/gene2go_all.pkl}"
[[ -f $gene2go ]] && cp "$gene2go" "$wd/data/gears_pert_data/gene2go_all.pkl"
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

# Patch ONLY the three hardcoded cluster paths -> our vendored dirs + side-loaded ckpt.
patched="$wd/run_scfoundation.py"
sed -e "s#[^\"']*scfoundation_gears/#$FORK/#g" \
    -e "s#[^\"']*scfoundation/model/#$MODEL/#g" \
    -e "s#/home/ahlmanne/huber/data/scfoundation_model/models.ckpt#$CKPT#g" \
    -e "s#^batch_size=6#batch_size=$BATCH#" \
    -e "s#^test_batch_size=6#test_batch_size=$BATCH#" \
    "$VENDORED" > "$patched"
export OMNI_VENDORED_SCRIPT="$patched"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" --epochs "$EPOCHS" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "scfoundation: $ds seed=$seed epochs=$EPOCHS ckpt=$CKPT -> $ds.predictions.json.gz"
