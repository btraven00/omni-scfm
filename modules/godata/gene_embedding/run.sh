#!/usr/bin/env bash
# Side-load builder: GENE EMBEDDINGS for lpm_scgptGeneEmb / lpm_scFoundationGeneEmb.
#
# The paper swaps lpm's gene embedding G for a foundation model's (run_perturbation_
# benchmark.R:35-36). Neither extractor runs the model on data — both are weight lookups,
# so they need no GPU and no target dataset, and are built ONCE like gene2go
# ([[ob-no-global-input]]). The vendored scripts run verbatim except for path patches:
#
#   --kind scgpt         extract_gene_embedding_scgpt.py — model.encoder(every vocab id),
#                        i.e. scGPT's token embedding + LayerNorm (512 dims x ~60k tokens).
#                        Env scgpt-gpu (after its postinstall's first line; flash-attn is
#                        NOT needed — the transformer never runs). Patched: the ckpt dir.
#   --kind scfoundation  extract_gene_embedding_scfoundation.py — the checkpoint's
#                        model.pos_emb.weight (768 dims x 19264 genes + 3 special tokens).
#                        Env scfoundation-gpu. Patched: the ckpt path, and the gene names
#                        read from the vendored OS_scRNA_gene_index.19264.tsv instead of
#                        scFoundation's GEARS/data/demo.h5ad — verified identical (same
#                        19264 names, same order; demo.h5ad @ 397631c, git blob c29c324).
#
# Usage: pixi run -e scgpt-gpu make-gene-emb-scgpt / pixi run -e scfoundation-gpu make-gene-emb-scfoundation
# Output: <output_dir>/<name>.tsv   (header = gene names, rows = dims)
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
output_dir="" ; kind="" ; name=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --kind)       kind="$2";       shift 2 ;;
    --name)       name="$2";       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir && -n $kind ]] || {
  echo "gene_embedding/run.sh: need --output_dir and --kind scgpt|scfoundation" >&2; exit 2; }

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/results"
SRC="$REPO/vendor/paper/benchmark/src"

case "$kind" in
  scgpt)
    name="${name:-scgpt_gene}"
    # same resolution as modules/methods/scgpt/run.sh: an omni-huggingface manifest or a dir
    model="${OMNI_SCGPT_MODEL:-$REPO/data/scgpt/scgpt_human_hf.json}"
    [[ -e $model ]] || model="$REPO/data/scgpt/scGPT_human"
    [[ -f $model ]] && model=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$model")
    [[ -f $model/best_model.pt && -f $model/vocab.json && -f $model/args.json ]] || {
      echo "gene_embedding: no scGPT checkpoint at '$model' — run \`pixi run -e hf fetch-scgpt-model-hf\`" >&2; exit 3; }
    sed "s|/home/ahlmanne/huber/data/scgpt_models/scGPT_human|$model|" \
      "$SRC/extract_gene_embedding_scgpt.py" > "$wd/extract.py"
    ;;
  scfoundation)
    name="${name:-scfoundation_gene}"
    ckpt="${OMNI_SCFOUNDATION_CKPT:-$REPO/data/scfoundation/models.ckpt}"
    [[ -f $ckpt ]] || {
      echo "gene_embedding: no scFoundation checkpoint at '$ckpt' — run \`pixi run fetch-scfoundation-model\`" >&2; exit 3; }
    genes="$REPO/vendor/scfoundation/model/OS_scRNA_gene_index.19264.tsv"
    sed -e "s|^demo_adata = ad.read_h5ad(.*|demo_genes = pd.read_csv('$genes', sep='\\\\t')|" \
        -e "s|^gene_names = demo_adata.var.gene_name.tolist()|gene_names = demo_genes.gene_name.tolist()|" \
        -e "s|/home/ahlmanne/huber/data/scfoundation_model/models.ckpt|$ckpt|" \
      "$SRC/extract_gene_embedding_scfoundation.py" > "$wd/extract.py"
    ;;
  *) echo "gene_embedding/run.sh: --kind must be scgpt or scfoundation (got '$kind')" >&2; exit 2 ;;
esac
# every path patch must have landed, or the script would read a cluster path
if grep -q "/home/ahlmanne" "$wd/extract.py"; then
  echo "gene_embedding: unpatched cluster path in the $kind extractor (upstream script changed?)" >&2; exit 4
fi

( cd "$wd" && python extract.py --working_dir "$wd" --result_id emb )

mkdir -p "$output_dir"
cp "$wd/results/emb" "$output_dir/$name.tsv"
echo "gene_embedding: $kind -> $output_dir/$name.tsv ($(head -1 "$output_dir/$name.tsv" | tr '\t' '\n' | wc -l) genes)"
