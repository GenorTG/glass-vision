#!/usr/bin/env bash
# Start the headless test emulator in Docker and connect the host's adb to it.
# Prints the adb serial on stdout (everything else goes to stderr).
#
#   scripts/emulator/start.sh [--camera virtualscene|emulated|none] [--poster IMG]
#                             [--table-poster IMG] [--memory MB] [--cores N]
#                             [--timeout SEC] [--rebuild]
#
# --poster puts IMG (png/jpeg) on the wall the virtual-scene back camera looks at on boot,
# so the app's camera sees that picture. Posters are fixed for the container's lifetime.
set -euo pipefail
# shellcheck source=scripts/emulator/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

camera=virtualscene
poster=
table_poster=
memory=4096
cores=4
timeout=300
rebuild=0

while (($#)); do
    case $1 in
    --camera) camera=$2; shift 2 ;;
    --poster) poster=$2; shift 2 ;;
    --table-poster) table_poster=$2; shift 2 ;;
    --memory) memory=$2; shift 2 ;;
    --cores) cores=$2; shift 2 ;;
    --timeout) timeout=$2; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    -h | --help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) gv_die "unknown argument: $1" ;;
    esac
done

case $camera in virtualscene | emulated | none) ;; *) gv_die "--camera must be virtualscene, emulated or none" ;; esac
[[ -x $ADB ]] || gv_die "adb not found at $ADB (set ANDROID_HOME or ADB)"
[[ -e /dev/kvm ]] || gv_die "/dev/kvm missing: no hardware virtualization"
command -v docker >/dev/null || gv_die "docker not installed"

if gv_container_running; then
    gv_log "container $GV_EMU_CONTAINER already running; reconnecting"
    gv_connect 60 || gv_die "running container but adb cannot connect to $GV_EMU_SERIAL"
    printf '%s\n' "$GV_EMU_SERIAL"
    exit 0
fi
docker rm -f -v "$GV_EMU_CONTAINER" >/dev/null 2>&1 || true

if ((rebuild)) || ! docker image inspect "$GV_EMU_IMAGE" >/dev/null 2>&1; then
    gv_log "building $GV_EMU_IMAGE (downloads ~1.9 GB from dl.google.com)"
    docker build -t "$GV_EMU_IMAGE" "$GV_EMU_DIR" >&2
fi

# The host's adb key is copied into the guest at boot, so adb is authorized without a prompt.
# adb creates the key pair the first time its server starts.
"$ADB" start-server >/dev/null
adbkey_pub=${ANDROID_USER_HOME:-$HOME/.android}/adbkey.pub
[[ -f $adbkey_pub ]] || gv_die "no adb public key at $adbkey_pub"

run_args=(
    -d --name "$GV_EMU_CONTAINER"
    --device /dev/kvm
    -p "127.0.0.1:$GV_EMU_ADB_PORT:5555"
    -p "127.0.0.1:$GV_EMU_CONSOLE_PORT:5554"
    -v "$adbkey_pub:/root/.android/adbkey.pub:ro"
    -e "GV_CAMERA_BACK=$camera" -e "GV_EMU_MEMORY=$memory" -e "GV_EMU_CORES=$cores"
    -e "GV_EMU_ARGS=${GV_EMU_ARGS:-}"
)
add_poster() { # slot path
    local path ext
    [[ -f $2 ]] || gv_die "poster not found: $2"
    path=$(realpath "$2")
    ext=${path##*.}
    ext=${ext,,}
    case $ext in png | jpg | jpeg) ;; *) gv_die "poster must be .png/.jpg/.jpeg: $2" ;; esac
    run_args+=(-v "$path:/posters/$1.$ext:ro")
}
[[ -n $poster ]] && add_poster wall "$poster"
[[ -n $table_poster ]] && add_poster table "$table_poster"

started=$SECONDS
docker run "${run_args[@]}" "$GV_EMU_IMAGE" >/dev/null
gv_log "container started; waiting for boot (timeout ${timeout}s)"

deadline=$((started + timeout))
gv_connect 120 || { docker logs --tail 40 "$GV_EMU_CONTAINER" >&2; gv_die "adb could not connect"; }
until [[ $(gv_adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r') == 1 ]]; do
    gv_container_running || { docker logs --tail 40 "$GV_EMU_CONTAINER" >&2; gv_die "emulator exited"; }
    ((SECONDS < deadline)) || gv_die "boot timed out after ${timeout}s"
    sleep 2
    # Early in boot adbd restarts; re-establish the transport if it dropped.
    [[ $(gv_adb get-state 2>/dev/null) == device ]] || gv_connect 30 || true
done
until gv_adb shell pm path android 2>/dev/null | grep -q package:; do
    ((SECONDS < deadline)) || gv_die "package manager not ready after ${timeout}s"
    sleep 1
done
gv_log "booted in $((SECONDS - started))s"

# Test-friendly device state: no animations, screen stays on and unlocked.
for key in window_animation_scale transition_animation_scale animator_duration_scale; do
    gv_adb shell settings put global "$key" 0
done
gv_adb shell svc power stayon true
gv_adb shell input keyevent KEYCODE_WAKEUP
gv_adb shell wm dismiss-keyguard >/dev/null 2>&1 || true

printf '%s\n' "$GV_EMU_SERIAL"
