#!/usr/bin/env bash
# Side-load fetcher: GEARS gene2go vocabulary (global reference data).
#
# Fetches gene2go_all.pkl ONCE from Harvard Dataverse (datafile 6153417) and
# md5-verifies it — the universal GO mapping GEARS' PertData.load needs to GO-filter
# perturbations. It is NOT dataset-specific, and GEARS otherwise re-downloads it on
# every split/method/seed run.
#
# This is deliberately NOT an OmniBenchmark stage. OB 0.5.1 cannot wire a single
# global artifact into multiple per-dataset lineages: a parallel-root output listed
# in a downstream `inputs:` is silently dropped from argv (not an ancestor of the
# dataset lineage), and making it the sole root breaks the {dataset} label mechanism.
# So we SIDE-LOAD: fetch once into data/godata/ (run `pixi run fetch-godata`), and the
# split/method run.sh scripts default to $REPO/data/godata/gene2go_all.pkl. (A proper
# fix — a first-class shared/global input — is planned in OB itself.)
#
# (essential_all_data_pert_genes.pkl is GEARS-GENERATED per dataset, not a download.)
#
# Usage: bash modules/godata/gene2go/run.sh --output_dir data/godata
# Output:
#   <output_dir>/gene2go_all.pkl   the GO vocabulary (md5 77c9af0c61c30ea4d7a85680f4d122dc)
set -euo pipefail

URL="https://dataverse.harvard.edu/api/access/datafile/6153417"
MD5="77c9af0c61c30ea4d7a85680f4d122dc"

output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "gene2go/run.sh: need --output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
out="$output_dir/gene2go_all.pkl"

# urllib follows the Dataverse redirect; hashlib verifies. (python from the base env;
# avoids depending on curl/wget being present in the conda env.)
python - "$URL" "$out" "$MD5" <<'PY'
import hashlib, shutil, sys, urllib.request
url, out, want = sys.argv[1], sys.argv[2], sys.argv[3]
# Dataverse 403s the default python-urllib User-Agent; send a browser-like one
# (GEARS' own downloader uses requests, which sends a UA). urlopen follows redirects.
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (omni-scfm gene2go fetch)"})
with urllib.request.urlopen(req) as r, open(out, "wb") as f:
    shutil.copyfileobj(r, f)
got = hashlib.md5(open(out, "rb").read()).hexdigest()
if got != want:
    sys.exit(f"gene2go md5 mismatch: got {got}, want {want}")
print(f"gene2go: fetched {out} (md5 {got})")
PY
