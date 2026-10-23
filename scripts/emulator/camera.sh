#!/usr/bin/env bash
# Control what the emulator's back camera (virtual scene) sees. Needs start.sh --camera
# virtualscene (the default).
#
#   scripts/emulator/camera.sh poster IMG [wall|table]   put IMG (png/jpeg) on the wall/table poster
#   scripts/emulator/camera.sh poster --default [wall|table]
#   scripts/emulator/camera.sh look-at-poster            walk the camera to face the wall poster
#   scripts/emulator/camera.sh reset                     back to the boot position (living room)
#
# The virtual scene is a 3D room rendered by the emulator; the camera starts in the living room
# facing a TV. The wall poster hangs in the next room: `look-at-poster` replays the emulator's
# bundled "Walk_to_image_room" macro (~10 s), after which the poster fills the middle of the
# frame (seen in perspective, not as a flat full-frame image).
set -euo pipefail
# shellcheck source=scripts/emulator/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

macros=/opt/android-sdk/emulator/resources/macros

gv_container_running || gv_die "emulator not running; run scripts/emulator/start.sh first"

play_macro() {
    local reply
    reply=$(gv_console "automation play $macros/$1")
    [[ -z $reply ]] || gv_die "console: $reply"
}

case ${1:-} in
poster)
    (($# >= 2)) || gv_die "usage: $0 poster IMG|--default [wall|table]"
    slot=${3:-wall}
    [[ $slot == wall || $slot == table ]] || gv_die "slot must be wall or table"
    if [[ $2 == --default ]]; then
        reply=$(gv_console "virtualscene-image $slot")
    else
        [[ -f $2 ]] || gv_die "image not found: $2"
        ext=${2##*.}
        ext=${ext,,}
        case $ext in png | jpg | jpeg) ;; *) gv_die "image must be .png/.jpg/.jpeg" ;; esac
        # Unique name per call: the emulator caches textures by path.
        dest=/tmp/poster-$slot-$(date +%s%N).$ext
        docker cp "$2" "$GV_EMU_CONTAINER:$dest" >/dev/null
        reply=$(gv_console "virtualscene-image $slot $dest")
    fi
    [[ -z $reply ]] || gv_die "console: $reply"
    ;;
look-at-poster)
    play_macro Walk_to_image_room
    sleep "${GV_MACRO_WAIT:-12}"
    ;;
reset)
    play_macro Reset_position
    sleep 2
    ;;
*)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
