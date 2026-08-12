#!/usr/bin/env bash
# Side-load fetcher: scGPT whole-human pretrained checkpoint (the scGPT_human DIR =
# best_model.pt ~196MB + vocab.json + args.json).
#
# DEPRECATED — use ../scgpt_model_hf (`pixi run -e hf fetch-scgpt-model-hf`), which pulls
# the SAME bytes off the Hugging Face Hub: revision-pinned, no gdown, no Drive rate limits.
# Verified 2026-08-12 with `cmp`: best_model.pt and vocab.json are byte-identical.
# Kept because the sha256 pins below are what that fetcher verifies against — this file is
# the provenance record — and as a fallback if the HF repo disappears. The scgpt method
# still accepts the directory this writes.
#
# Unlike scFoundation's single .ckpt, scGPT ships a DIRECTORY (run_scgpt.py reads
# best_model.pt / vocab.json / args.json from it). Provenance is the PUBLIC scGPT release
# (bowang-lab/scGPT, a Google-Drive folder linked from the repo README) — far more
# reproducible than scFoundation's auth-walled SharePoint. We pin the sha256 of each file.
#
# It is NOT an OB DAG node — the checkpoint is an external/global resource and OB 0.5.1
# would silently drop it as a method input (see modules/godata/gene2go, [[ob-no-global-input]]).
#
# Usage:
#   OMNI_SCGPT_URL='file:///abs/path/scGPT_human' pixi run fetch-scgpt-model   # copy a local mirror dir (primary)
#   OMNI_SCGPT_URL='gdrive://<folder-id>'          pixi run fetch-scgpt-model   # gdown a Drive folder (needs gdown)
# Output: <output_dir>/scGPT_human/{best_model.pt,vocab.json,args.json}
set -euo pipefail

# Canonical public source: the scGPT_human Google-Drive folder (from the scGPT README).
# gdrive folder id 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y (whole-human; verify against the repo).
DEFAULT_URL="${OMNI_SCGPT_URL:-gdrive://1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y}"
url="$DEFAULT_URL"
output_dir=""
# sha256 of the on-disk reference checkpoint (best_model.pt is the load-bearing one).
SHA_BEST="${OMNI_SCGPT_SHA_BEST:-6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922}"
SHA_VOCAB="${OMNI_SCGPT_SHA_VOCAB:-acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f}"
SHA_ARGS="${OMNI_SCGPT_SHA_ARGS:-c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --url)        url="$2";        shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "scgpt_model/run.sh: need --output_dir" >&2; exit 2; }
dest="$output_dir/scGPT_human"
mkdir -p "$dest"

case "$url" in
  file://*)
    src="${url#file://}"
    for f in best_model.pt vocab.json args.json; do
      [[ -f "$src/$f" ]] || { echo "scgpt_model/run.sh: $src/$f missing in local mirror" >&2; exit 1; }
      cp "$src/$f" "$dest/$f"
    done
    echo "scgpt_model: copied local mirror $src -> $dest"
    ;;
  gdrive://*)
    command -v gdown >/dev/null || { echo "scgpt_model/run.sh: gdown not on PATH (pip install gdown) to fetch the Drive folder" >&2; exit 3; }
    fid="${url#gdrive://}"
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    gdown --folder "https://drive.google.com/drive/folders/$fid" -O "$tmp" --quiet
    found=$(find "$tmp" -name best_model.pt -printf '%h\n' | head -1)
    [[ -n $found ]] || { echo "scgpt_model/run.sh: best_model.pt not found in the downloaded folder" >&2; exit 1; }
    for f in best_model.pt vocab.json args.json; do cp "$found/$f" "$dest/$f"; done
    echo "scgpt_model: downloaded Drive folder $fid -> $dest"
    ;;
  *) echo "scgpt_model/run.sh: unsupported URL scheme '$url' (use file:// or gdrive://)" >&2; exit 2 ;;
esac

# Verify the pinned hashes (best_model.pt is load-bearing; vocab/args also pinned).
verify() {  # <file> <expected-sha256>
  local got; got=$(sha256sum "$dest/$1" | cut -d' ' -f1)
  [[ "$got" == "$2" ]] || { echo "scgpt_model/run.sh: sha256 mismatch on $1: got $got want $2" >&2; exit 5; }
}
verify best_model.pt "$SHA_BEST"
verify vocab.json     "$SHA_VOCAB"
verify args.json      "$SHA_ARGS"
echo "scgpt_model: checkpoint hash-verified -> $dest"
