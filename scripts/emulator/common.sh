# Shared settings for scripts/emulator/*.sh (sourced, not executed).
# shellcheck shell=bash
# shellcheck disable=SC2034  # variables are used by the scripts that source this file

GV_EMU_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GV_REPO_ROOT=$(cd "$GV_EMU_DIR/../.." && pwd)

GV_EMU_IMAGE=${GV_EMU_IMAGE:-glass-vision-emulator:api34}
GV_EMU_CONTAINER=${GV_EMU_CONTAINER:-glass-vision-emulator}
# Host ports (127.0.0.1 only). Deliberately outside 5554-5585: the adb server auto-probes that
# range for local emulators, which would show the same device twice (emulator-N + host:port)
# and make connectedAndroidTest run every test twice.
GV_EMU_ADB_PORT=${GV_EMU_ADB_PORT:-15555}
GV_EMU_CONSOLE_PORT=${GV_EMU_CONSOLE_PORT:-15554}
GV_EMU_SERIAL=127.0.0.1:$GV_EMU_ADB_PORT

ANDROID_HOME=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}
ADB=${ADB:-$ANDROID_HOME/platform-tools/adb}

gv_adb() {
    "$ADB" -s "$GV_EMU_SERIAL" "$@"
}

gv_log() {
    printf '[emulator] %s\n' "$*" >&2
}

gv_die() {
    gv_log "ERROR: $*"
    exit 1
}

gv_container_running() {
    [[ $(docker inspect -f '{{.State.Running}}' "$GV_EMU_CONTAINER" 2>/dev/null) == true ]]
}

# Sends one command to the emulator console (e.g. "sensor status") and prints the reply.
gv_console() {
    local reply
    reply=$(printf '%s\nquit\n' "$*" | nc -q 2 127.0.0.1 "$GV_EMU_CONSOLE_PORT" | tr -d '\r') || true
    printf '%s\n' "$reply" | sed -e '/^Android Console:/d' -e '/^OK$/d' -e '/^$/d'
}

# adb connect + wait until the device answers; adb's `connect` reports success even when
# the socket is still dead, so verify with get-state.
gv_connect() {
    local deadline=$((SECONDS + ${1:-60}))
    while ((SECONDS < deadline)); do
        "$ADB" connect "$GV_EMU_SERIAL" >/dev/null 2>&1 || true
        if [[ $(gv_adb get-state 2>/dev/null) == device ]]; then
            return 0
        fi
        "$ADB" disconnect "$GV_EMU_SERIAL" >/dev/null 2>&1 || true
        sleep 2
    done
    return 1
}
