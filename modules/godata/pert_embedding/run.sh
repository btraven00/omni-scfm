#!/usr/bin/env bash
# Side-load builder: PERTURBATION EMBEDDINGS for the lpm_*PertEmb variants.
#
# The paper swaps lpm's perturbation embedding P for one derived from somewhere else
# and asks whether the number moves (run_perturbation_benchmark.R:36-38). Two vendored
# extractors produce those, both CPU, both run here VERBATIM via wrapper.R:
#
#   --kind gears  extract_pert_embedding_from_gears.R — spectral embedding of GEARS' GO
#                 Jaccard graph (go_essential_all.csv). NB the script parses --pca_dim
#                 and then ignores it: the embedding is hardcoded to 2 dims.
#   --kind pca    extract_pert_embedding_pca.R — pseudobulk + PCA of a reference dataset
#                 (Replogle K562/RPE1), one column per perturbation.
#
# Neither depends on the target dataset or the seed, so like gene2go these are built
# ONCE and side-loaded rather than expressed as an OB stage ([[ob-no-global-input]]).
# NB the pca extractor calls set.seed(pa$seed) but never declares --seed, so it seeds
# with NULL and irlba starts from a random init: the embedding is stable in span but
# not bit-identical between builds. Build once, keep the artifact.
#
# Usage (r env — `pixi run -e r <task>`):
#   make-pert-emb-gears / make-pert-emb-k562 / make-pert-emb-rpe1
# Output: <output_dir>/<name>.tsv   (header = perturbation names, rows = dims)
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(pwd)"
WRAPPER="$REPO/modules/methods/wrapper.R"

output_dir="" ; kind="" ; data_h5ad="" ; go_csv="" ; pca_dim="" ; name=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)            output_dir="$2"; shift 2 ;;
    --kind)                  kind="$2";       shift 2 ;;
    --data.h5ad|--data_h5ad) data_h5ad="$2";  shift 2 ;;
    --go_essential)          go_csv="$2";     shift 2 ;;
    --pca_dim)               pca_dim="$2";    shift 2 ;;
    --name)                  name="$2";       shift 2 ;;
    *)                       shift ;;
  esac
done
[[ -n $output_dir && -n $kind ]] || {
  echo "pert_embedding/run.sh: need --output_dir and --kind gears|pca" >&2; exit 2; }

wd=$(mktemp -d); trap 'rm -rf "$wd"' EXIT
mkdir -p "$wd/results"
export TMPDIR="$wd/tmp"; mkdir -p "$TMPDIR"
args=()

case "$kind" in
  gears)
    script="extract_pert_embedding_from_gears.R"
    name="${name:-gears_go}"
    go_csv="${go_csv:-$REPO/data/godata/go_essential_all.csv}"
    [[ -f $go_csv ]] || {
      echo "pert_embedding: no GO graph at '$go_csv' — run \`pixi run fetch-go-essential\`" >&2; exit 3; }
    # The script reads it from the GEARS folder layout, under the dataset-ish name
    # "go_essential_all" (extract_pert_embedding_from_gears.R:22).
    mkdir -p "$wd/data/gears_pert_data/go_essential_all"
    ln -sf "$(realpath "$go_csv")" "$wd/data/gears_pert_data/go_essential_all/go_essential_all.csv"
    ;;
  pca)
    script="extract_pert_embedding_pca.R"
    [[ -n $data_h5ad ]] || { echo "pert_embedding: --kind pca needs --data.h5ad" >&2; exit 2; }
    [[ -f $data_h5ad ]] || {
      echo "pert_embedding: no dataset at '$data_h5ad' — run \`pixi run fetch-replogle\`" >&2; exit 3; }
    ds=$(basename "$data_h5ad"); ds="${ds%%.*}"
    name="${name:-$ds}"
    mkdir -p "$wd/data/gears_pert_data/$ds"
    ln -sf "$(realpath "$data_h5ad")" "$wd/data/gears_pert_data/$ds/perturb_processed.h5ad"
    args+=(--dataset_name "$ds")
    ;;
  *) echo "pert_embedding/run.sh: --kind must be gears or pca (got '$kind')" >&2; exit 2 ;;
esac
[[ -n $pca_dim ]] && args+=(--pca_dim "$pca_dim")

# Unlike the method scripts, these write a single FILE at results/<result_id>.
export OMNI_VENDORED_SCRIPT="$REPO/vendor/paper/benchmark/src/$script"
( cd "$wd" && Rscript "$WRAPPER" --working_dir "$wd" --result_id emb "${args[@]}" )

mkdir -p "$output_dir"
cp "$wd/results/emb" "$output_dir/$name.tsv"
echo "pert_embedding: $script -> $output_dir/$name.tsv ($(head -1 "$output_dir/$name.tsv" | tr '\t' '\n' | wc -l) perturbations)"
