#!/usr/bin/env bash
# OmniBenchmark method module: scGPT (transformer foundation-model perturbation predictor).
#
# scGPT = a pretrained scGPT `TransformerGenerator` (whole-human checkpoint) fine-tuned
# per (split, seed) on the GEARS perturbation dataloader (run_scgpt.py). Trains on GPU.
# BIT-FAITHFUL to the paper: use_fast_transformer=True (flash-attn 1.0.4), batch_size=64.
#
# Like scfoundation, run_scgpt.py hardcodes cluster paths and has no CLI hook for them,
# so we sed-patch ONLY those and run the rest verbatim:
#   load_model = ".../scGPT_human"   -> $SCGPT_MODEL_DIR (side-loaded checkpoint DIR)
# and, FOR norman_from_scfoundation ONLY (the script's `if dataset == norman_from_scfoundation`
# branch sys.path-inserts the SAME forked GEARS 0.0.2 as scfoundation), the two inserts:
#   scfoundation_gears/ -> vendor/scfoundation/scfoundation_gears
#   model/              -> vendor/scfoundation/model
# plus the batch-size literals (paper 64; lower to fit a smaller GPU):
#   batch_size = 64 / eval_batch_size = 64 -> $BATCH
#
# SCOPE: norman_from_scfoundation only (first faithful cut) — it reuses the already
# vendored fork and the scF substrate's gene2go. The paper also reports scGPT on
# adamson/norman (the script's else-branch, which needs essential_all_data_pert_genes.pkl
# + the env's stock cell-gears 0.0.2); broadening scope is a documented TODO ([[scgpt-method]]).
#
# Unlike scfoundation, scGPT does NOT call GEARS.model_initialize / get_go_auto, so it
# needs NO GO perturbation graph (no go_essential side-load). It only needs gene2go.pkl
# (for the pert_names lookup at run_scgpt.py:237) — staged below.
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --data.go PATH              the dataset's go.csv (accepted but UNUSED — scGPT builds no GO graph)
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env knobs:
#   OMNI_SCGPT_MODEL  path to the scGPT_human checkpoint DIR (default data/scgpt/scGPT_human)
#   OMNI_SCGPT_EPOCHS fine-tune epochs (default 15, the paper's value)
#   OMNI_SCGPT_BATCH  batch size (default 64, the paper's value)
#   OMNI_GEARS_CACHE  dir with a gene2go *.pkl (else data/godata side-load)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene scGPT prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"           # scipy/pandas shim + runpy
VENDORED="$REPO/vendor/paper/benchmark/src/run_scgpt.py"
FORK="$REPO/vendor/scfoundation/scfoundation_gears"        # only used by the norman_from_scfoundation branch
MODEL="$REPO/vendor/scfoundation/model"
EPOCHS="${OMNI_SCGPT_EPOCHS:-15}"
BATCH="${OMNI_SCGPT_BATCH:-64}"   # paper default 64; lower it to fit a smaller GPU (BatchNorm-free, so any >=1 is fine)

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
  echo "scgpt/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

# OB runs the module from its staged commit dir, so $REPO=$(pwd) is NOT the user repo.
# The side-loaded (gitignored) data/ lives in the user repo = parent of out/, recovered
# from --output_dir (an absolute path under the real out/). Fall back to $REPO locally.
DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"
SCGPT_MODEL="${OMNI_SCGPT_MODEL:-$DATA_ROOT/data/scgpt/scGPT_human}"
[[ -f "$SCGPT_MODEL/best_model.pt" && -f "$SCGPT_MODEL/vocab.json" && -f "$SCGPT_MODEL/args.json" ]] || {
  echo "scgpt/run.sh: scGPT checkpoint dir incomplete at '$SCGPT_MODEL' (need best_model.pt + vocab.json + args.json)." >&2
  echo "  Set OMNI_SCGPT_MODEL or run: OMNI_SCGPT_URL='file:///abs/scGPT_human' pixi run fetch-scgpt-model" >&2; exit 3; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
# run_scgpt.py loads via data_path = data/gears_pert_data/<ds> (its else-branch handles
# non-norman/adamson/dixit names by data_path), so stage the AnnData there.
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"

# gene2go: for norman_from_scfoundation the forked PertData.__init__ opens gene2go.pkl at
# the gears_pert_data FOLDER level, and run_scgpt.py:237 opens gene2go.pkl at the DATASET
# dir level — stage it at both (byte-identical to gene2go_all.pkl, same md5).
gene2go="${gene2go:-$DATA_ROOT/data/godata/gene2go_all.pkl}"
if [[ -f $gene2go ]]; then
  cp "$gene2go" "$wd/data/gears_pert_data/gene2go_all.pkl"
  cp "$gene2go" "$wd/data/gears_pert_data/gene2go.pkl"
  cp "$gene2go" "$wd/data/gears_pert_data/$ds/gene2go.pkl"
  cp "$gene2go" "$wd/data/gears_pert_data/$ds/gene2go_all.pkl"
fi
if [[ -n "${OMNI_GEARS_CACHE:-}" && -d "$OMNI_GEARS_CACHE" ]]; then
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/" 2>/dev/null || true
  cp -n "$OMNI_GEARS_CACHE"/*.pkl "$wd/data/gears_pert_data/$ds/" 2>/dev/null || true
fi
[[ -f "$wd/data/gears_pert_data/$ds/gene2go.pkl" ]] || {
  echo "scgpt/run.sh: gene2go.pkl missing. Run 'pixi run fetch-godata' (writes" >&2
  echo "  data/godata/gene2go_all.pkl) or set OMNI_GEARS_CACHE to a dir with gene2go.pkl." >&2; exit 4; }
cfg="config"; rid="result"
cp "$split" "$wd/results/$cfg"

# Patch ONLY the hardcoded cluster paths -> our vendored dirs + side-loaded checkpoint,
# and the batch-size literals. The fork-path subs are no-ops for non-scfoundation datasets.
patched="$wd/run_scgpt.py"
sed -e "s#/home/ahlmanne/huber/data/scgpt_models/scGPT_human#$SCGPT_MODEL#g" \
    -e "s#[^\"']*scfoundation_gears/#$FORK/#g" \
    -e "s#[^\"']*scfoundation/model/#$MODEL/#g" \
    -e "s#^batch_size = 64#batch_size = $BATCH#" \
    -e "s#^eval_batch_size = 64#eval_batch_size = $BATCH#" \
    "$VENDORED" > "$patched"
export OMNI_VENDORED_SCRIPT="$patched"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" --epochs "$EPOCHS" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "scgpt: $ds seed=$seed epochs=$EPOCHS batch=$BATCH model=$SCGPT_MODEL -> $ds.predictions.json.gz"
