#!/usr/bin/env bash
# Side-load fetcher: the scGPT whole-human checkpoint, from the HUGGING FACE HUB.
#
# Sibling of ../scgpt_model (which gdowns the Google-Drive release, now deprecated). Same
# weights, reproducible transport: this delegates to the omni-huggingface OB module
# (github.com/omnibenchmark/omni-huggingface), pinned to a repo REVISION.
#
# It writes a MANIFEST, not a copy: the weights land once in the shared HF cache and the
# manifest records `snapshot`, the cache path, which ../../methods/scgpt reads. Clearing
# the cache invalidates the manifest — re-run this.
#
# Not an OB stage: ob 0.6.0's APIVersion enum stops at 0.5.0 (model/benchmark.py:124-135),
# so an api-0.7 stage cannot be declared yet. Same reason as ../gene2go, [[ob-no-global-input]].
#
# Usage:
#   pixi run -e hf fetch-scgpt-model-hf                    # -> data/scgpt/scgpt_human_hf.json
#   OMNI_HF_MODULE=/path/to/omni-huggingface pixi run -e hf fetch-scgpt-model-hf
# Output: <output_dir>/scgpt_human_hf.json
set -euo pipefail

REPO_HF="${OMNI_HF_REPO:-perturblab/scgpt-human}"
REV="${OMNI_HF_REVISION:-571a0445d68fa48381f863ff75dd4f6d0eae3dfc}"
MODULE_URL="${OMNI_HF_MODULE_URL:-https://github.com/omnibenchmark/omni-huggingface}"
# Pinned like any other module in benchmark.yaml — `main` moves.
MODULE_COMMIT="${OMNI_HF_MODULE_COMMIT:-dd042df5956fff3a022dde98a0fffa3e5517c503}"

# sha256 of the canonical Drive release (../scgpt_model's pins): this is what makes the HF
# mirror trustworthy. args.json legitimately differs (the HF copy carries extra keys), so
# it is checked on the keys run_scgpt.py:190-194 reads, not on bytes.
SHA_BEST="6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922"
SHA_VOCAB="acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "scgpt_model_hf/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
manifest="$(cd "$output_dir" && pwd)/scgpt_human_hf.json"   # absolute: run.py resolves relative paths against its own --output_dir
py=$(command -v python3 || command -v python)

"$py" -c 'import huggingface_hub' 2>/dev/null || {
  echo "scgpt_model_hf/run.sh: huggingface_hub not importable — run via 'pixi run -e hf fetch-scgpt-model-hf'." >&2; exit 3; }

module="${OMNI_HF_MODULE:-}"
if [[ -z $module ]]; then
  command -v git >/dev/null || { echo "scgpt_model_hf/run.sh: git needed to fetch $MODULE_URL (or set OMNI_HF_MODULE)" >&2; exit 3; }
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  git clone --quiet "$MODULE_URL" "$tmp/omni-huggingface"   # full clone: --depth 1 can only land on a branch tip
  git -C "$tmp/omni-huggingface" checkout --quiet "$MODULE_COMMIT"
  module="$tmp/omni-huggingface"
fi
[[ -f "$module/run.py" ]] || { echo "scgpt_model_hf/run.sh: no run.py in '$module'" >&2; exit 3; }

# A .json output makes the module write its manifest and copy nothing. --output is plain
# argparse there, not gated on any OB api version.
"$py" "$module/run.py" --repo "$REPO_HF" --repo_type model --revision "$REV" \
  --files "args.json,best_model.pt,vocab.json" --output "manifest=$manifest"

# Verify what the manifest points at against the canonical release.
snap=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$manifest")
for f in "best_model.pt:$SHA_BEST" "vocab.json:$SHA_VOCAB"; do
  got=$(sha256sum "$snap/${f%%:*}" | cut -d' ' -f1)
  [[ $got == "${f##*:}" ]] || {
    echo "scgpt_model_hf: ${f%%:*} sha256 $got != ${f##*:} — the HF copy is NOT the release." >&2
    echo "  modules/methods/scgpt pins its faithfulness on these bytes; refusing." >&2; exit 1; }
done
"$py" - "$snap/args.json" <<'PY'
import json, sys
# The five keys run_scgpt.py:190-194 reads off args.json (the HF copy carries more).
want = {"embsize": 512, "nheads": 8, "d_hid": 512, "nlayers": 12, "n_layers_cls": 3}
cfg = json.load(open(sys.argv[1]))
got = {k: cfg.get(k) for k in want}
if got != want:
    sys.exit(f"scgpt_model_hf: args.json config drift: {got} != {want}")
PY
echo "scgpt_model_hf: $REPO_HF@${REV:0:7} verified -> $manifest (cache: $snap)"
