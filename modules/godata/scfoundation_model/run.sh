#!/usr/bin/env bash
# Side-load fetcher: scFoundation pretrained checkpoint (models.ckpt, ~GB).
#
# Like the gene2go fetcher, this is NOT an OB stage — the checkpoint is an external
# global resource and OB 0.5.1 can't wire one into the per-dataset lineage (it'd be
# silently dropped; see modules/godata/gene2go/run.sh). So it's side-loaded once into
# data/scfoundation/ and the scfoundation module reads it (OMNI_SCFOUNDATION_CKPT
# defaults there).
#
# Source is a URL you provide (the BioMap weights live behind SharePoint, not
# script-friendly): http(s) OR a local file:// — both handled by urllib.
#
# TODO(reproducibility): the canonical source is a BioMap SharePoint PublicSharedfiles
# folder (auth-walled, not scriptable today). When hapiq / omni-data gain SharePoint
# support, make this a proper omni-data download module with:
#   url    https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk#models.ckpt
#   sha256 9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9  (0.1B maeautobin, 1432587886 bytes)
# Until then: download manually + point OMNI_SCFOUNDATION_URL at a local file:// or a
# mirror (Figshare/Zenodo/S3) with OMNI_SCFOUNDATION_MD5. NB: weights are under
# scFoundation's separate MODEL_LICENSE — check redistribution terms before mirroring.
#
# Usage:
#   OMNI_SCFOUNDATION_URL='https://…/models.ckpt'      pixi run fetch-scfoundation-model
#   OMNI_SCFOUNDATION_URL='file:///abs/path/models.ckpt' pixi run fetch-scfoundation-model
#   [OMNI_SCFOUNDATION_MD5=<md5>] to verify.
# Output: <output_dir>/models.ckpt
set -euo pipefail

url="${OMNI_SCFOUNDATION_URL:-}"
md5="${OMNI_SCFOUNDATION_MD5:-}"
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --url)        url="$2";        shift 2 ;;
    --md5)        md5="$2";        shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "scfoundation_model/run.sh: need --output_dir" >&2; exit 2; }
[[ -n $url ]] || { echo "scfoundation_model/run.sh: set OMNI_SCFOUNDATION_URL (http(s):// or file://)" >&2; exit 2; }
mkdir -p "$output_dir"
out="$output_dir/models.ckpt"

python - "$url" "$out" "$md5" <<'PY'
import hashlib, shutil, sys, urllib.request
url, out, want = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request(url, headers={"User-Agent": "omni-scfm scfoundation fetch"})
with urllib.request.urlopen(req) as r, open(out, "wb") as f:
    shutil.copyfileobj(r, f)
if want:
    got = hashlib.md5(open(out, "rb").read()).hexdigest()
    if got != want:
        sys.exit(f"scfoundation ckpt md5 mismatch: got {got}, want {want}")
    print(f"scfoundation: fetched {out} (md5 {got})")
else:
    print(f"scfoundation: fetched {out} (no md5 check)")
PY
