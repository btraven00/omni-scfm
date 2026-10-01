#!/usr/bin/env bash
# OmniBenchmark method module: scBERT (Performer pretrained on PanglaoDB; Yang et al. 2022).
#
# run_scbert.py (verbatim): re-lay the data out in scBERT's 16906-gene order, fine-tune the
# pretrained model as a perturbation classifier on <=5000 cells (the paper's
# finetune_modified.py via `torch.distributed.launch`, 1 GPU), then for each condition
# embed 100 control cells with the target genes' expression swapped in (the paper's
# attn_sum_save_modified.py, one subprocess per condition) and ridge-regress expression on
# the embedding. Both helpers run via os.system, so a crash there only surfaces later as a
# missing finetune_best.pth / full_attn_sum.npy — read the full log.
#
# Hardcoded cluster paths, sed-patched to:
#   cd /home/ahlmanne/prog/scBERT        -> a per-job copy of vendor/scbert (@262fd4b9) +
#                                           the paper's two *_modified.py scripts, as the
#                                           paper did. Per job because finetune pickles its
#                                           label encoding into the cwd.
#   .../panglao_human.h5ad               -> a 0-cell stub with scBERT's gene axis (the script
#                                           reads nothing else from the 11 GB original)
#   .../panglao_pretrain.pth             -> the side-loaded checkpoint (fetch-scbert-hf)
#   scfoundation_gears/ + model/ inserts -> vendor/scfoundation (norman_from_scfoundation)
# performer_pytorch loads ../data/gene2vec_16906.npy relative to its cwd: rebuilt per job
# from the checkpoint, byte-identical to the author's file (see side_inputs.py).
#
# Faithful quirks kept: the fine-tune seed is hardcoded (2021) in finetune_modified.py, so
# --seed only drives run_scbert.py's numpy sampling; each embedding is the mean over a
# 100-row matrix of which 3 rows are filled (attn_sum_save_modified.py).
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
#   --finetuning_epochs N       (benchmark.yaml parameter; paper = the script default 100)
# Env knobs:
#   OMNI_SCBERT_MODEL    manifest (json with "snapshot") or the .pth itself
#                        (default data/scbert/scbert_hf.json)
#   OMNI_SCBERT_EPOCHS   fine-tune epochs if --finetuning_epochs is not given (default 100)
#   OMNI_GEARS_CACHE     dir with a gene2go *.pkl (else data/godata side-load)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"
PAPER_SRC="$REPO/vendor/paper/benchmark/src"
HERE="$REPO/modules/methods/scbert"
FORK="$REPO/vendor/scfoundation/scfoundation_gears"
MODEL="$REPO/vendor/scfoundation/model"
EPOCHS="${OMNI_SCBERT_EPOCHS:-100}"

output_dir="" ; data_h5ad="" ; split="" ; seed="" ; gene2go=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --godata.gene2go|--godata_gene2go|--gene2go|--gene2go_all) gene2go="$2"; shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    --finetuning_epochs)                           EPOCHS="$2";     shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "scbert/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"
ckpt="${OMNI_SCBERT_MODEL:-$DATA_ROOT/data/scbert/scbert_hf.json}"
if [[ $ckpt == *.json && -f $ckpt ]]; then
  ckpt="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$ckpt")/panglao_pretrain.pth"
fi
[[ -f $ckpt ]] || {
  echo "scbert/run.sh: no scBERT checkpoint at '$ckpt'. Run: pixi run -e hf fetch-scbert-hf (or set OMNI_SCBERT_MODEL)." >&2; exit 3; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
export TMPDIR="$wd"   # the script's mkdtemp()/TemporaryDirectory()s die with $wd
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

# Per-job scBERT checkout (cwd of both helpers) + its side inputs. ../data/ from the
# checkout is $wd/data, which also holds gears_pert_data/ — that's fine.
cp -r "$REPO/vendor/scbert" "$wd/scbert"
cp "$PAPER_SRC/finetune_modified.py" "$PAPER_SRC/attn_sum_save_modified.py" "$wd/scbert/"
python "$HERE/side_inputs.py" "$ckpt" "$HERE/panglao_human_genes.txt" \
  "$wd/panglao_human.h5ad" "$wd/data/gene2vec_16906.npy"

patched="$wd/run_scbert.py"
sed -e "s#cd /home/ahlmanne/prog/scBERT#cd $wd/scbert#g" \
    -e "s#/home/ahlmanne/data/scbert/panglao_human.h5ad#$wd/panglao_human.h5ad#" \
    -e "s#/home/ahlmanne/projects/perturbation_prediction-benchmark/data/panglao_pretrain.pth#$ckpt#" \
    -e "s#[^\"']*scfoundation_gears/#$FORK/#g" \
    -e "s#[^\"']*scfoundation/model/#$MODEL/#g" \
    "$PAPER_SRC/run_scbert.py" > "$patched"
if grep -nE "/home/ahlmanne|/g/huber" "$patched" >&2; then
  echo "scbert/run.sh: unpatched cluster path(s) above — upstream script changed?" >&2; exit 5
fi
export OMNI_VENDORED_SCRIPT="$patched"

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" --finetuning_epochs "$EPOCHS" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "scbert: $ds seed=$seed finetuning_epochs=$EPOCHS -> $ds.predictions.json.gz"
