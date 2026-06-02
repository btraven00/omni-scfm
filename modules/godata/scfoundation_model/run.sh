#!/usr/bin/env bash
# Side-load fetcher: scFoundation pretrained checkpoint (models.ckpt, ~1.43GB).
#
# Uses hapiq the same way the omni-data module does — `hapiq download url <uri> --hash`
# — so the checkpoint is fetched + provenance-verified with the project's standard
# downloader (NOT a bespoke urllib path). The BioMap weights live in a SharePoint
# *folder* share-link; hapiq's SharePoint support (HEAD of github.com/btraven00/hapiq)
# resolves a file inside it via the `#<file>` fragment trick — hence the `#models.ckpt`
# appended to the folder URL below.
#
# REQUIREMENTS:
#   - hapiq with SharePoint support (HEAD; the conda/omnidata hapiq predates it). Until
#     it's released to the channel, build HEAD (`go build`) and put it on PATH.
#   - It is NOT an OB DAG node — the checkpoint is an external/global resource and OB
#     0.5.1 would silently drop it as a method input (see modules/godata/gene2go).
#
# Usage:
#   pixi run fetch-scfoundation-model                         # default SharePoint URL+hash
#   OMNI_SCFOUNDATION_URL='https://…#models.ckpt' pixi run fetch-scfoundation-model
#   OMNI_SCFOUNDATION_URL='file:///abs/models.ckpt' …         # a local mirror
# Output: <output_dir>/models.ckpt
set -euo pipefail

# Canonical source: BioMap PublicSharedfiles SharePoint folder + the file fragment.
DEFAULT_URL='https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk#models.ckpt'
url="${OMNI_SCFOUNDATION_URL:-$DEFAULT_URL}"
hash="${OMNI_SCFOUNDATION_HASH:-sha256:9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9}"
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --url)        url="$2";        shift 2 ;;
    --hash)       hash="$2";       shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "scfoundation_model/run.sh: need --output_dir" >&2; exit 2; }
command -v hapiq >/dev/null || { echo "scfoundation_model/run.sh: hapiq not on PATH (need HEAD with SharePoint support)" >&2; exit 3; }
mkdir -p "$output_dir"

# file:// — hapiq's url downloader is for http(s); a local mirror just gets copied.
case "$url" in
  file://*) cp "${url#file://}" "$output_dir/models.ckpt"; echo "copied local mirror -> $output_dir/models.ckpt"; exit 0 ;;
esac

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# omni-data's invocation: hapiq download url <uri> --out <dir> --hash <algo:hex> -y
hapiq download url "$url" --out "$tmp" --hash "$hash" -y
ckpt=$(find "$tmp" -type f -name '*.ckpt' -o -type f -name 'models.ckpt' 2>/dev/null | head -1)
[[ -z $ckpt ]] && ckpt=$(find "$tmp" -type f -not -name 'hapiq.json' | head -1)
[[ -n $ckpt ]] || { echo "scfoundation_model/run.sh: no file downloaded" >&2; exit 1; }
mv "$ckpt" "$output_dir/models.ckpt"
echo "scfoundation: fetched $output_dir/models.ckpt (hapiq, hash-verified)"
