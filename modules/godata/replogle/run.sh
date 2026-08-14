#!/usr/bin/env bash
# Side-load fetcher: the Replogle REFERENCE dataset for the `transfer` method.
#
# run_transfer_perturbation_prediction.R predicts dataset A's perturbation effects
# from the SAME perturbations measured in a second dataset (--reference_data). The
# paper uses Replogle K562/RPE1 essential, which GEARS' PertData.load downloads from
# Harvard Dataverse (gears/pertdata.py:160-165). We fetch the same archive once and
# stage the h5ad; modules/methods/transfer/run.sh symlinks it into the run sandbox.
#
# NOT an OB DAG node: it's a second dataset feeding every target dataset's lineage,
# exactly the global-input shape OB 0.5.1 can't wire (see modules/godata/gene2go).
#
# The h5ad is re-written through anndata (like modules/preprocess) for two reasons:
#   * the shipped GEARS h5ad predates the root `encoding-type` attr picklerick needs
#   * sparse indices must be sorted or R's Matrix rejects X on read
#
# Usage: pixi run fetch-replogle              (K562; the paper's default reference)
#        OMNI_REPLOGLE_NAME=replogle_rpe1_essential \
#        OMNI_REPLOGLE_URL=https://dataverse.harvard.edu/api/access/datafile/7458694 \
#          pixi run fetch-replogle            (RPE1)
# Output: <output_dir>/<name>.h5ad
set -euo pipefail

NAME="${OMNI_REPLOGLE_NAME:-replogle_k562_essential}"
URL="${OMNI_REPLOGLE_URL:-https://dataverse.harvard.edu/api/access/datafile/7458695}"
# Pinned 2026-08-14 from a browser download (Dataverse WAF-challenges scripted clients —
# HTTP 202 + a JS proof-of-work; see AGENTS.md "Mirroring"). Mirror the archive to
# btraven/omni-scfm-data and point URL at the HF resolve URL: this md5 is what makes the
# mirror trustworthy, so keep it pinned.
MD5="${OMNI_REPLOGLE_MD5:-84eed779531de88f83d3eb16a773a261}"

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --url)        URL="$2";        shift 2 ;;
    --name)       NAME="$2";       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "replogle/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
out="$output_dir/$NAME.h5ad"

python - "$URL" "$out" "$MD5" "$NAME" <<'PY'
import hashlib, os, shutil, sys, tempfile, urllib.request, zipfile
from pathlib import Path

import anndata as ad
from scipy.sparse import issparse

url, out, want, name = sys.argv[1:5]
tmp = Path(tempfile.mkdtemp())

# file:// — a local mirror (zip or h5ad) is used as-is; the archive is ~1GB.
if url.startswith("file://"):
    src = Path(url[len("file://"):])
else:
    src = tmp / "download"
    # Dataverse 403s the default python-urllib User-Agent (see godata/gene2go).
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (omni-scfm replogle fetch)"})
    with urllib.request.urlopen(req) as r, open(src, "wb") as f:
        shutil.copyfileobj(r, f)
    if os.path.getsize(src) == 0:
        sys.exit(f"replogle: empty download from {url} (Dataverse bot challenge? retry, "
                 f"or fetch by hand and pass OMNI_REPLOGLE_URL=file:///path/to/archive.zip)")

got = hashlib.md5(src.read_bytes()).hexdigest()
if want and got != want:
    sys.exit(f"replogle md5 mismatch: got {got}, want {want}")

if zipfile.is_zipfile(src):
    with zipfile.ZipFile(src) as zf:
        members = [m for m in zf.namelist() if m.endswith("perturb_processed.h5ad")]
        if not members:
            sys.exit(f"replogle: no perturb_processed.h5ad inside {url}")
        h5ad = tmp / "raw.h5ad"
        with zf.open(members[0]) as s, open(h5ad, "wb") as d:
            shutil.copyfileobj(s, d)
else:
    h5ad = src

adata = ad.read_h5ad(h5ad)
for mat in (adata.X, *adata.layers.values()):   # R's Matrix needs sorted indices
    if issparse(mat) and hasattr(mat, "sort_indices") and not mat.has_sorted_indices:
        mat.sort_indices()
adata.write_h5ad(out)                            # + modern encoding-type for picklerick
shutil.rmtree(tmp, ignore_errors=True)
print(f"replogle: wrote {out} ({adata.shape[0]} cells x {adata.shape[1]} genes), "
      f"archive md5 {got}{'' if want else '  <- pin this in modules/godata/replogle/run.sh'}")
PY
