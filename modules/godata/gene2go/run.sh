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

# The md5 is the identity, the host is a parameter: point OMNI_GENE2GO_URL at any mirror
# (or file:///…) and the check still proves you got the canonical bytes. Dataverse
# WAF-challenges some networks (HTTP 202, empty body) — see [[transfer-method]]; a hapiq
# cache server (OMNI_HAPIQ_PEERS) that already holds the bytes avoids Dataverse entirely.
URL="${OMNI_GENE2GO_URL:-https://dataverse.harvard.edu/api/access/datafile/6153417}"
MD5="${OMNI_GENE2GO_MD5:-77c9af0c61c30ea4d7a85680f4d122dc}"

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

# Through hapiq's content-addressed cache (and any OMNI_HAPIQ_PEERS cache servers), md5-
# checked — see ../hapiq_get.sh. A file:// URL is copied as-is.
bash "$(dirname "${BASH_SOURCE[0]}")/../hapiq_get.sh" "$URL" "$out" "$MD5"
