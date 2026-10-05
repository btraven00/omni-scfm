#!/usr/bin/env bash
# Side-load fetcher: GEARS precomputed GO-essential graph (go_essential_all.csv, 338MB).
#
# This is the file scFoundation's forked GEARS 0.0.2 needs for the GO perturbation graph.
# The fork can only BUILD it via a single-threaded ~99M-pair Jaccard loop (HOURS); STOCK
# cell-gears 0.1.2 instead DOWNLOADS this precomputed copy. So we fetch the same artifact
# the stock GEARS does — Harvard Dataverse datafile 6934319 (a 57MB tar that extracts to
# go_essential_all/go_essential_all.csv) — and the scfoundation module stages it as the
# dataset's go.csv so get_go_auto just reads it (seconds, not hours). md5-verified.
#
# NOT an OB DAG node (external/global reference; see modules/godata/gene2go).
#
# Usage: pixi run fetch-go-essential
# Output: <output_dir>/go_essential_all.csv
set -euo pipefail

URL="${OMNI_GO_ESSENTIAL_URL:-https://dataverse.harvard.edu/api/access/datafile/6934319}"
MD5="${OMNI_GO_ESSENTIAL_MD5:-3d7b3e13a07420b3b95445b8c16d7eb1}"
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --url)        URL="$2";        shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "go_essential/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
out="$output_dir/go_essential_all.csv"

# file:// — a local mirror of the .csv itself is just copied.
case "$URL" in
  file://*.csv) cp "${URL#file://}" "$out"; echo "copied $out"; exit 0 ;;
esac

# The archive comes through hapiq's cache (../hapiq_get.sh). Its pin is on the extracted
# CSV, not on the tar, so the download is unpinned and the CSV is checked below.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
bash "$(dirname "${BASH_SOURCE[0]}")/../hapiq_get.sh" "$URL" "$tmp/go.tar"

python - "$tmp" "$out" "$MD5" <<'PY'
import glob, hashlib, os, shutil, sys, tarfile
tmp, out, want = sys.argv[1], sys.argv[2], sys.argv[3]
with tarfile.open(os.path.join(tmp, "go.tar")) as t:
    t.extractall(tmp)
csv = (glob.glob(os.path.join(tmp, "**", "go_essential_all.csv"), recursive=True)
       or glob.glob(os.path.join(tmp, "go_essential_all.csv")))
if not csv:
    sys.exit("go_essential: no go_essential_all.csv in the archive")
got = hashlib.md5(open(csv[0], "rb").read()).hexdigest()
if got != want:
    sys.exit(f"go_essential md5 mismatch: got {got}, want {want}")
shutil.move(csv[0], out)
print(f"go_essential: fetched {out} (md5 {got})")
PY
