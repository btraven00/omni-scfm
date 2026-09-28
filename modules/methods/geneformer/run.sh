#!/usr/bin/env bash
# OmniBenchmark method module: Geneformer (transformer foundation model, gf-12L-95M-i4096).
#
# run_geneformer.py (verbatim): fine-tune the pretrained model as a cell-state classifier
# on the training conditions, in-silico perturb (delete / overexpress) the target genes in
# control cells, take the mean CLS embedding per condition, and ridge-regress expression on
# it. Trains on GPU (paper: H100, 120GB RAM, 50h budget).
#
# The script hardcodes cluster paths with no CLI hook, so we sed-patch ONLY those:
#   /home/ahlmanne/prog/Geneformer/...   -> the pinned HF snapshot (code + weights)
#   scfoundation_gears/ + model/ inserts -> vendor/scfoundation (norman_from_scfoundation only)
# plus ONE generalisation: the "already has Ensembl IDs" branch keys on a hardcoded list of
# dataset NAMES (adamson, norman, replogle_*). We key it on the var index instead
# (all-ENSG), which is the same decision for every paper dataset and also correct for
# datasets the list doesn't know (e.g. the norman_tiny fixture).
#
# --perturbation_type: the paper used `delete` on single-pert datasets and `overexpress` on
# double-pert ones. Default `auto` picks it the same way from the split (any condition with
# two non-ctrl genes -> overexpress).
#
# Inputs (OB):
#   --data.h5ad PATH            processed AnnData from `preprocess`
#   --split.set2conditions PATH {"train","val","test"} from `split` (per seed)
# Env knobs:
#   OMNI_GENEFORMER_HF  omni-huggingface manifest (default data/geneformer/geneformer_hf.json,
#                       from `pixi run -e hf fetch-geneformer-hf`)
#   OMNI_GEARS_CACHE    dir with a gene2go *.pkl (else data/godata side-load)
# Output:
#   {dataset}.predictions.json.gz   {condition: [per-gene prediction]}
#   {dataset}.gene_names.json
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/gears_wrapper.py"
VENDORED="$REPO/vendor/paper/benchmark/src/run_geneformer.py"
FORK="$REPO/vendor/scfoundation/scfoundation_gears"
MODEL="$REPO/vendor/scfoundation/model"
PTYPE="auto" ; NCELLS=300

output_dir="" ; data_h5ad="" ; split="" ; seed="" ; gene2go=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)                                  output_dir="$2"; shift 2 ;;
    --data.h5ad|--data_h5ad)                       data_h5ad="$2";  shift 2 ;;
    --godata.gene2go|--godata_gene2go|--gene2go|--gene2go_all) gene2go="$2"; shift 2 ;;
    --split.set2conditions|--split_set2conditions) split="$2";      shift 2 ;;
    --seed)                                        seed="$2";       shift 2 ;;
    --perturbation_type)                           PTYPE="$2";      shift 2 ;;
    --ncells_training)                             NCELLS="$2";     shift 2 ;;
    *)                                             shift ;;
  esac
done
[[ -n $output_dir && -n $data_h5ad && -n $split ]] || {
  echo "geneformer/run.sh: need --output_dir, --data.h5ad, --split.set2conditions" >&2; exit 2; }

DATA_ROOT="${output_dir%%/out/*}"
[[ -z $DATA_ROOT || $DATA_ROOT == "$output_dir" ]] && DATA_ROOT="$REPO"
manifest="${OMNI_GENEFORMER_HF:-$DATA_ROOT/data/geneformer/geneformer_hf.json}"
[[ -f $manifest ]] || {
  echo "geneformer/run.sh: no Geneformer manifest at '$manifest'. Run: pixi run -e hf fetch-geneformer-hf" >&2; exit 3; }
SNAP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$manifest")
[[ -f "$SNAP/gf-12L-95M-i4096/model.safetensors" && -f "$SNAP/geneformer/ensembl_mapping_dict_gc95M.pkl" ]] || {
  echo "geneformer/run.sh: snapshot '$SNAP' incomplete (HF cache cleared?) — re-run fetch-geneformer-hf." >&2; exit 3; }

ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
if [[ -z $seed ]]; then
  pj="$(dirname "$split")/parameters.json"
  [[ -f $pj ]] && seed=$(grep -oE '"seed"[[:space:]]*:[[:space:]]*[0-9]+' "$pj" | grep -oE '[0-9]+' | head -1)
  seed="${seed:-1}"
fi
if [[ $PTYPE == auto ]]; then
  PTYPE=$(python -c '
import json, sys
conds = [c for v in json.load(open(sys.argv[1])).values() for c in v]
print("overexpress" if any(len([g for g in c.split("+") if g != "ctrl"]) > 1 for c in conds) else "delete")' "$split")
fi

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
export TMPDIR="$wd"   # the script's tempfile.mkdtemp() dirs (tokenized data, fine-tuned model) die with $wd
mkdir -p "$wd/data/gears_pert_data/$ds" "$wd/results"
cp "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
# gene2go: stock PertData reads gene2go_all.pkl at the folder level (downloads it otherwise);
# the fork (norman_from_scfoundation) reads gene2go.pkl — stage both names.
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

patched="$wd/run_geneformer.py"
sed -e "s#/home/ahlmanne/prog/Geneformer/#$SNAP/#g" \
    -e "s#[^\"']*scfoundation_gears/#$FORK/#g" \
    -e "s#[^\"']*scfoundation/model/#$MODEL/#g" \
    -e "s#^if args.dataset_name in \['adamson', 'norman', 'replogle_k562_essential', 'replogle_rpe1_essential'\]:#if new_adata.var.index.str.startswith('ENSG').all():#" \
    "$VENDORED" > "$patched"
grep -q "startswith('ENSG')" "$patched" || { echo "geneformer/run.sh: Ensembl-branch patch did not apply" >&2; exit 5; }
export OMNI_VENDORED_SCRIPT="$patched"
export PYTHONPATH="$SNAP${PYTHONPATH:+:$PYTHONPATH}"   # `import geneformer` -> the pinned snapshot
# Everything is local (snapshot + staged data): forbid Hub/datasets network calls per run.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

( cd "$wd" && python "$WRAPPER" \
    --dataset_name "$ds" --test_train_config_id "$cfg" \
    --working_dir "$wd" --result_id "$rid" --seed "$seed" \
    --perturbation_type "$PTYPE" --ncells_training "$NCELLS" )

mkdir -p "$output_dir"
python -c "import gzip,json,sys; json.dump(json.load(open(sys.argv[1])), gzip.open(sys.argv[2],'wt',encoding='utf8'))" \
    "$wd/results/$rid/all_predictions.json" "$output_dir/$ds.predictions.json.gz"
cp "$wd/results/$rid/gene_names.json" "$output_dir/$ds.gene_names.json"
echo "geneformer: $ds seed=$seed perturbation_type=$PTYPE ncells_training=$NCELLS -> $ds.predictions.json.gz"
