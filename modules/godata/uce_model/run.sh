#!/usr/bin/env bash
# Side-load fetcher: UCE model files (figshare article 24320806, md5-pinned).
#
# What UCE's eval_single_anndata.py would otherwise download on first use into
# ./model_files/ (species_chrom.csv, species_offsets.pkl, all_tokens.torch,
# protein_embeddings/), plus the two pretrained models run_uce.py points at:
#   4layer_model.torch        (3.4GB) -> method `uce`   (--model_type 4layers)
#   33l_8ep_1024t_1280.torch  (5.7GB) -> method `uce33` (--model_type 33layers)
# Shared files ~5.7GB (all_tokens 3.0GB + protein_embeddings tar 2.7GB). Idempotent:
# a file already present with the right md5 is not re-downloaded.
#
# Downloads go through hapiq (`hapiq fetch --hash`, as omni-data does), so they land in
# hapiq's blob cache and are md5-verified. The hash is the identity, the host is a
# parameter: OMNI_UCE_MIRROR=<base URL> fetches <base>/<filename> instead of figshare.
#
# Usage:  pixi run -e omnidata fetch-uce-model          # both models
#         OMNI_UCE_MODELS=4layers pixi run fetch-uce-model
# Output: <output_dir>/model_files/{species_chrom.csv,species_offsets.pkl,all_tokens.torch,
#         protein_embeddings/,4layer_model.torch,33l_8ep_1024t_1280.torch}
set -euo pipefail

MODELS="${OMNI_UCE_MODELS:-4layers 33layers}"
MIRROR="${OMNI_UCE_MIRROR:-}"
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) output_dir="$2"; shift 2 ;;
    --name)       shift 2 ;;
    *)            shift ;;
  esac
done
[[ -n $output_dir ]] || { echo "uce_model/run.sh: need --output_dir" >&2; exit 2; }
command -v hapiq >/dev/null || { echo "uce_model/run.sh: hapiq not on PATH — run via 'pixi run -e omnidata fetch-uce-model'" >&2; exit 3; }
mkdir -p "$output_dir"; mf="$(cd "$output_dir" && pwd)/model_files"; mkdir -p "$mf"   # absolute: hapiq --out
# Force hapiq's blob cache on (the system /etc/hapiq/config.toml `mode = "on"` is not
# honoured by the conda hapiq; an explicit --config is). Re-fetches then hit the cache.
hq_cfg=$(mktemp); trap 'rm -f "$hq_cfg"' EXIT
printf '[cache]\nmode = "on"\ndir = "%s"\n' "${HAPIQ_CACHE_DIR:-$HOME/.cache/hapiq}" > "$hq_cfg"

# figshare file id | filename | md5 (figshare's computed_md5)
files=(
  "42706558|species_chrom.csv|42a9b5871b1ed955fd05414b3ab89081"
  "42706555|species_offsets.pkl|ddf9143508857e62607f323b338ff383"
  "42706585|all_tokens.torch|3523ea431c403f95701b6eeb0c30b997"
  "42715213|protein_embeddings.tar.gz|6756d7fa8348e0f70b5e5b11e324f023"
)
[[ $MODELS == *4layers* ]]  && files+=("42706576|4layer_model.torch|3e2f59d6da6eaa5396aa297edfcec08f")
[[ $MODELS == *33layers* ]] && files+=("43423236|33l_8ep_1024t_1280.torch|eed62d4fe51873c87bcf9660faa07707")

for f in "${files[@]}"; do
  IFS='|' read -r id name md5 <<<"$f"
  out="$mf/$name"
  # the tarball is deleted after extraction; the extracted dir stands in for it
  [[ $name == protein_embeddings.tar.gz && -f "$mf/protein_embeddings/.md5-$md5" ]] && { echo "ok (cached) $name"; continue; }
  if [[ -f $out && $(md5sum "$out" | cut -d' ' -f1) == "$md5" ]]; then echo "ok (cached) $name"; continue; fi
  url="https://ndownloader.figshare.com/files/$id"   # the API host serves directly (see hapiq-figshare notes)
  [[ -n $MIRROR ]] && url="${MIRROR%/}/$name"
  echo "fetching $name <- $url"
  tmp="$mf/.fetch-$id"; rm -rf "$tmp"; mkdir -p "$tmp"
  hapiq --config "$hq_cfg" fetch "$url" --out "$tmp" --hash "md5:$md5" --timeout 7200 -y
  got_file=$(find "$tmp" -type f -not -name 'hapiq.json' | head -1)
  # belt and braces: hapiq --hash had a path bug once (hapiq-figshare notes); re-check here
  [[ -n $got_file && $(md5sum "$got_file" | cut -d' ' -f1) == "$md5" ]] || {
    echo "uce_model: $name failed md5 $md5; refusing." >&2; exit 1; }
  mv "$got_file" "$out"; rm -rf "$tmp"
  if [[ $name == protein_embeddings.tar.gz ]]; then
    tar -xzf "$out" -C "$mf" && rm "$out"
    [[ -d "$mf/protein_embeddings" ]] || { echo "uce_model: tar did not yield protein_embeddings/" >&2; exit 1; }
    touch "$mf/protein_embeddings/.md5-$md5"
  fi
done
echo "uce_model: $MODELS -> $mf"
