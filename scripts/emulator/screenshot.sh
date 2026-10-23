#!/usr/bin/env bash
# Save the emulator screen as a PNG:  scripts/emulator/screenshot.sh out.png
set -euo pipefail
# shellcheck source=scripts/emulator/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

(($# == 1)) || gv_die "usage: $0 <out.png>"
out=$1
mkdir -p "$(dirname "$out")"
gv_adb exec-out screencap -p >"$out.tmp"
# A PNG starts with \x89PNG; anything else is an adb error message.
[[ $(head -c 4 "$out.tmp" | od -An -c | tr -d ' ') == 211PNG ]] || {
    cat "$out.tmp" >&2
    rm -f "$out.tmp"
    gv_die "screencap failed"
}
mv "$out.tmp" "$out"
printf '%s\n' "$out"
