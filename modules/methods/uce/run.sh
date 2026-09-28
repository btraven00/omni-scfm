#!/usr/bin/env bash
# OmniBenchmark method module: UCE (Universal Cell Embeddings; `uce` = 4 layers, `uce33` = 33).
#
# run_uce.py (verbatim): for each condition, take 100 control cells, overwrite the target
# genes' expression with values sampled from that condition's cells, embed them with the
# pretrained UCE model (one `python eval_single_anndata.py` subprocess per condition), and
# ridge-regress expression on the mean embedding. Inference only — no fine-tuning.
#
# The script hardcodes cluster paths, sed-patched to:
#   cd /home/ahlmanne/prog/UCE                    -> a per-job copy of vendor/uce (@8227a65)
#   /home/ahlmanne/data/universal_cell_embedding/ -> the side-loaded model_files
#   scfoundation_gears/ + model/ inserts          -> vendor/scfoundation (norman_from_scfoundation)
# UCE resolves its token / protein-embedding files relative to its cwd (./model_files/), so
# the per-job copy gets a model_files/ of symlinks into the side-load (read-only, shared).
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
#   --model_type 4layers|33layers   (benchmark.yaml parameter; paper method uce / uce33)
# Env knobs:
#   OMNI_UCE_MODEL_FILES  dir from `pixi run fetch-uce-model` (default data/uce/model_files)
#   OMNI_GEARS_CACHE      dir with a gene2go *.pkl (else data/godata side-load)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"
VENDORED="$REPO/vendor/paper/benchmark/src/run_uce.py"
UCE_CODE="$REPO/vendor/uce"
FORK="$REPO/vendor/scfoundation/scfoundation_gears"
MODEL="$REPO/vendor/scfoundation/model"
MTYPE="4layers"

output_dir="" ; data_h5ad="" ; split="" ; seed="" ; gene2go=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --godata.gene2go|--godata_gene2go|--gene2go|--gene2go_all) gene2go="$2"; shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    --model_type)                                  MTYPE="$2";      shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "uce/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }
case "$MTYPE" in
  4layers)  ckpt="4layer_model.torch" ;;
  33layers) ckpt="33l_8ep_1024t_1280.torch" ;;
  *) echo "uce/run.sh: --model_type must be 4layers or 33layers, not '$MTYPE'" >&2; exit 2 ;;
esac

DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"
MF="${OMNI_UCE_MODEL_FILES:-$DATA_ROOT/data/uce/model_files}"
for f in species_chrom.csv species_offsets.pkl all_tokens.torch protein_embeddings "$ckpt"; do
  [[ -e "$MF/$f" ]] || {
    echo "uce/run.sh: '$MF/$f' missing. Run: pixi run fetch-uce-model (or set OMNI_UCE_MODEL_FILES)." >&2; exit 3; }
done

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
export TMPDIR="$wd"   # the script's per-condition TemporaryDirectory()s die with $wd
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
gene2go="${gene2go:-$DATA_ROOT/data/godata/gene2go_all.pkl}"
if [[ -f $gene2go ]]; then
  cp "$gene2go" "$wd/data/gears_pert_data/gene2go_all.pkl"
  cp "$gene2go" "$wd/data/gears_pert_data/gene2go.pkl"
fi
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
fi
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

# Per-job UCE checkout: vendored code + model_files/ = its committed csv + symlinks to the side-load.
cp -r "$UCE_CODE" "$wd/uce"
for f in "$MF"/*; do ln -s "$(realpath "$f")" "$wd/uce/model_files/$(basename "$f")"; done

patched="$wd/run_uce.py"
sed -e "s#cd /home/ahlmanne/prog/UCE#cd $wd/uce#" \
    -e "s#/home/ahlmanne/data/universal_cell_embedding/#$(realpath "$MF")/#g" \
    -e "s#[^\"']*scfoundation_gears/#$FORK/#g" \
    -e "s#[^\"']*scfoundation/model/#$MODEL/#g" \
    "$VENDORED" > "$patched"
grep -q "cd $wd/uce" "$patched" || { echo "uce/run.sh: UCE-dir patch did not apply" >&2; exit 5; }
export OMNI_VENDORED_SCRIPT="$patched"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" --model_type "$MTYPE" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "uce: $ds seed=$seed model_type=$MTYPE -> $ds.predictions.json.gz"
