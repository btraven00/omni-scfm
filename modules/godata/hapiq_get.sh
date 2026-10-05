#!/usr/bin/env bash
# Shared side-load download helper: one URL -> one file, through hapiq's cache.
#
#   hapiq_get.sh <url> <out_file> [md5]
#
# Why hapiq: its cache is content-addressed (sha256 blobs + a URL->blob index), so a URL
# fetched once is served from the cache with no network, a cache dir can be copied to
# another machine, and `hapiq cache serve` shares it over HTTP — peers stream blobs by
# hash and the client verifies while streaming, so a peer needs no trust. That is how
# we mirror Dataverse (which WAF-challenges some networks) across our machines.
#
#   HAPIQ_CACHE_DIR    cache dir (default ~/.cache/hapiq)
#   OMNI_HAPIQ_PEERS   comma-separated cache servers asked BEFORE the origin,
#                      e.g. http://roland:7777 (see AGENTS.md "Mirroring")
#   OMNI_HAPIQ_TOKEN   bearer token for those servers, if they set one
#
# The md5 (when given) is checked here as well, not only via hapiq's --hash: the hash is
# the identity, the host is a parameter. hapiq's own witness is kept as <out>.hapiq.json.
# file:// URLs are copied as-is (pre-downloaded or browser-fetched archives).
# Needs hapiq >= 0.1.0 (`download url`; `fetch` was removed): OMNI_HAPIQ, else the
# omnidata pixi env (`pixi install -e omnidata`, pinned), else PATH; the version is checked.
set -euo pipefail

url="$1"; out="$2"; md5="${3:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $url == file://* ]]; then
  cp "${url#file://}" "$out"
else
  # The repo's pinned hapiq first (omnidata env), PATH only as a fallback: a machine-wide
  # hapiq can be an old dev build whose `download url` exits 0 without writing a file.
  hq="${OMNI_HAPIQ:-}"
  [[ -z $hq && -x "$REPO/.pixi/envs/omnidata/bin/hapiq" ]] && hq="$REPO/.pixi/envs/omnidata/bin/hapiq"
  [[ -z $hq ]] && hq="$(command -v hapiq || true)"
  [[ -n $hq ]] || { echo "hapiq_get: no hapiq (pixi install -e omnidata, or set OMNI_HAPIQ)" >&2; exit 3; }
  ver="$("$hq" version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
  [[ -n $ver && "$(printf '%s\n0.1.0\n' "$ver" | sort -V | head -1)" == 0.1.0 ]] || {
    echo "hapiq_get: $hq is not hapiq >= 0.1.0 ('$("$hq" version 2>&1 | tail -1)')" >&2; exit 3; }

  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  {
    printf '[cache]\nmode = "on"\ndir = "%s"\n' "${HAPIQ_CACHE_DIR:-$HOME/.cache/hapiq}"
    if [[ -n ${OMNI_HAPIQ_PEERS:-} ]]; then
      printf '[cache.server]\npeers = [%s]\n' "$(sed 's/[^,][^,]*/"&"/g' <<<"$OMNI_HAPIQ_PEERS")"
      [[ -n ${OMNI_HAPIQ_TOKEN:-} ]] && printf 'token = "%s"\n' "$OMNI_HAPIQ_TOKEN"
    fi
  } > "$tmp/hapiq.toml"

  "$hq" --config "$tmp/hapiq.toml" download url "$url" --out "$tmp/out" \
    ${md5:+--hash "md5:$md5"} --timeout 7200 -y
  # hapiq names the file itself (Content-Disposition or URL basename, version-dependent):
  # take the one payload file, whatever it is called.
  mapfile -t got < <(find "$tmp/out" -type f ! -name hapiq.json)
  [[ ${#got[@]} -eq 1 ]] || { echo "hapiq_get: expected 1 file from $url, got ${#got[@]}" >&2; exit 1; }
  mv "${got[0]}" "$out"
  [[ -f "$tmp/out/hapiq.json" ]] && mv "$tmp/out/hapiq.json" "$out.hapiq.json"
fi

if [[ -n $md5 ]]; then
  have="$(md5sum "$out" | cut -d' ' -f1)"
  [[ $have == "$md5" ]] || { rm -f "$out"; echo "hapiq_get: md5 $have != pinned $md5 for $url" >&2; exit 1; }
fi
echo "hapiq_get: $url -> $out${md5:+ (md5 ok)}"
