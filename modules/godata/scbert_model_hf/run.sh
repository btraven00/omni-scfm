#!/usr/bin/env bash
# Side-load fetcher: the pretrained scBERT checkpoint (panglao_pretrain.pth), from the
# HUGGING FACE HUB — same mechanism as ../scgpt_model_hf (omni-huggingface, pinned).
#
# Upstream (TencentAILabHealthcare/scBERT) ships it via a login-walled WeChat Drive link;
# the author later posted a Google Drive copy (scBERT issue #46, Drive id
# 1_Pgk_o8AtQtoXr_ZLQx0eJYoSWzjxC8f). Nothing upstream publishes a checksum, so we mirrored
# those exact bytes to btraven/scbert-panglao-pretrain (GPL-3.0 — scBERT's LICENSE covers
# the weights; the repo carries it) and pin the sha256 of the Drive download here: that
# hash, not the host, is the identity. Any mirror serving the same bytes passes.
#
# Writes a MANIFEST, not a copy: the file lands once in the shared HF cache and the
# manifest records `snapshot`, the cache path. Clearing the cache invalidates it — re-run.
#
# Usage:  pixi run -e hf fetch-scbert-hf      # -> data/scbert/scbert_hf.json
# Output: <output_dir>/scbert_hf.json
set -euo pipefail

REPO_HF="${OMNI_HF_REPO:-btraven/scbert-panglao-pretrain}"
REV="${OMNI_HF_REVISION:-2cab9c47d58740f8599efdb4326b334762f4b41c}"
MODULE_URL="${OMNI_HF_MODULE_URL:-https://github.com/omnibenchmark/omni-huggingface}"
MODULE_COMMIT="${OMNI_HF_MODULE_COMMIT:-dd042df5956fff3a022dde98a0fffa3e5517c503}"

# sha256 of the Drive download (87,631,538 bytes), see the header.
SHA_CKPT="aa27109136580b24a1b985f63d0ba31f22490d09e71b13d1793cbff38619dfe3"

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "scbert_model_hf/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
manifest="$(cd "$output_dir" && pwd)/scbert_hf.json"   # absolute: run.py resolves relative paths against its own --output_dir
py=$(command -v python3 || command -v python)

"$py" -c 'import huggingface_hub' 2>/dev/null || {
  echo "scbert_model_hf/run.sh: huggingface_hub not importable — run via 'pixi run -e hf fetch-scbert-hf'." >&2; exit 3; }

module="${OMNI_HF_MODULE:-}"
if [[ -z $module ]]; then
  command -v git >/dev/null || { echo "scbert_model_hf/run.sh: git needed to fetch $MODULE_URL (or set OMNI_HF_MODULE)" >&2; exit 3; }
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  git clone --quiet "$MODULE_URL" "$tmp/omni-huggingface"   # full clone: --depth 1 can only land on a branch tip
  git -C "$tmp/omni-huggingface" checkout --quiet "$MODULE_COMMIT"
  module="$tmp/omni-huggingface"
fi
[[ -f "$module/run.py" ]] || { echo "scbert_model_hf/run.sh: no run.py in '$module'" >&2; exit 3; }

"$py" "$module/run.py" --repo "$REPO_HF" --repo_type model --revision "$REV" \
  --files "panglao_pretrain.pth" --output "manifest=$manifest"

snap=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$manifest")
got=$(sha256sum "$snap/panglao_pretrain.pth" | cut -d' ' -f1)
[[ $got == "$SHA_CKPT" ]] || {
  echo "scbert_model_hf: panglao_pretrain.pth sha256 $got != $SHA_CKPT — not the scBERT release; refusing." >&2; exit 1; }
echo "scbert_model_hf: $REPO_HF@${REV:0:7} verified -> $manifest (cache: $snap)"
