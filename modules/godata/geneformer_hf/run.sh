#!/usr/bin/env bash
# Side-load fetcher: Geneformer (code + gf-12L-95M-i4096 weights) from the HUGGING FACE HUB.
#
# The paper's geneformer_env installed `geneformer==0.1.0` from a git checkout of
# ctheodoris/Geneformer and pointed run_geneformer.py at that checkout's
# gf-12L-95M-i4096/ and geneformer/ensembl_mapping_dict_gc95M.pkl. The env yml was frozen
# on 2025-01-26 and the next code change landed 2025-01-27 (c81f6f9), so the checkout was
# 01d3ea89 (2025-01-10). We fetch exactly that revision — the package dir (whose token /
# median / ensembl dictionaries are LFS files) and the 95M weights — into the shared HF
# cache and write a MANIFEST (same mechanism as ../scgpt_model_hf). modules/methods/
# geneformer puts `snapshot` on PYTHONPATH, so nothing is pip-installed.
#
# Usage:  pixi run -e hf fetch-geneformer-hf        # -> data/geneformer/geneformer_hf.json
# Output: <output_dir>/geneformer_hf.json
set -euo pipefail

REPO_HF="${OMNI_HF_REPO:-ctheodoris/Geneformer}"
REV="${OMNI_HF_REVISION:-01d3ea8993c2a49704375d7ee6175210cfa33f11}"
MODULE_URL="${OMNI_HF_MODULE_URL:-https://github.com/omnibenchmark/omni-huggingface}"
MODULE_COMMIT="${OMNI_HF_MODULE_COMMIT:-dd042df5956fff3a022dde98a0fffa3e5517c503}"

# LFS sha256 oids at $REV (the files run_geneformer.py and the tokenizer actually read).
SHAS=(
  "gf-12L-95M-i4096/model.safetensors:4365ba23e393fcfa0e65a94ac64a0983cd788bd23a8d4914f4ab66f85cfe043c"
  "geneformer/ensembl_mapping_dict_gc95M.pkl:0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c"
  "geneformer/token_dictionary_gc95M.pkl:67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f"
  "geneformer/gene_median_dictionary_gc95M.pkl:a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5"
)

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "geneformer_hf/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
manifest="$(cd "$output_dir" && pwd)/geneformer_hf.json"
py=$(command -v python3 || command -v python)

"$py" -c 'import huggingface_hub' 2>/dev/null || {
  echo "geneformer_hf/run.sh: huggingface_hub not importable — run via 'pixi run -e hf fetch-geneformer-hf'." >&2; exit 3; }

module="${OMNI_HF_MODULE:-}"
if [[ -z $module ]]; then
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  git clone --quiet "$MODULE_URL" "$tmp/omni-huggingface"
  git -C "$tmp/omni-huggingface" checkout --quiet "$MODULE_COMMIT"
  module="$tmp/omni-huggingface"
fi

"$py" "$module/run.py" --repo "$REPO_HF" --repo_type model --revision "$REV" \
  --files "geneformer/*,gf-12L-95M-i4096/*" --output "manifest=$manifest"

snap=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$manifest")
for f in "${SHAS[@]}"; do
  got=$(sha256sum "$snap/${f%%:*}" | cut -d' ' -f1)
  [[ $got == "${f##*:}" ]] || { echo "geneformer_hf: ${f%%:*} sha256 $got != ${f##*:}; refusing." >&2; exit 1; }
done
echo "geneformer_hf: $REPO_HF@${REV:0:7} verified -> $manifest (cache: $snap)"
