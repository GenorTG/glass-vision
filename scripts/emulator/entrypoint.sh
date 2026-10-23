#!/usr/bin/env bash
# Runs INSIDE the emulator container (see Dockerfile). Creates the AVD on first start, launches
# the emulator headless and re-exposes its loopback-only adb/console ports on the container IP.
#
# Env (set by start.sh):
#   GV_CAMERA_BACK   virtualscene | emulated | none   (default virtualscene)
#   GV_EMU_MEMORY    guest RAM in MB                  (default 4096)
#   GV_EMU_CORES     guest vCPUs                      (default 4)
#   GV_EMU_ARGS      extra emulator arguments, word-split
# Mounts (optional): /root/.android/adbkey.pub (host adb key -> pre-authorized),
#                    /posters/wall.{png,jpg} and /posters/table.{png,jpg} (virtual scene posters).
set -euo pipefail

avd_dir=/root/.android/avd
avd=gv
camera_back=${GV_CAMERA_BACK:-virtualscene}

mkdir -p "$avd_dir/$avd.avd"
if [[ ! -f $avd_dir/$avd.avd/config.ini ]]; then
    cat >"$avd_dir/$avd.ini" <<EOF
avd.ini.encoding=UTF-8
path=$avd_dir/$avd.avd
target=android-$API_LEVEL
EOF
    cat >"$avd_dir/$avd.avd/config.ini" <<EOF
avd.ini.encoding=UTF-8
AvdId=$avd
avd.ini.displayname=glass-vision test
PlayStore.enabled=false
abi.type=x86_64
hw.cpu.arch=x86_64
hw.cpu.ncore=${GV_EMU_CORES:-4}
hw.ramSize=${GV_EMU_MEMORY:-4096}
hw.lcd.width=1080
hw.lcd.height=2400
hw.lcd.density=420
hw.keyboard=yes
hw.gpu.enabled=yes
hw.gpu.mode=swiftshader_indirect
hw.camera.back=$camera_back
hw.camera.front=emulated
hw.audioInput=no
hw.audioOutput=no
disk.dataPartition.size=6G
image.sysdir.1=system-images/android-$API_LEVEL/google_apis/x86_64/
tag.id=google_apis
tag.display=Google APIs
EOF
fi

# Empty token file = console auth disabled. The console is only published on the host's loopback.
: >/root/.emulator_console_auth_token

poster_args=()
for slot in wall table; do
    for ext in png jpg jpeg; do
        if [[ -f /posters/$slot.$ext ]]; then
            poster_args+=(-virtualscene-poster "$slot=/posters/$slot.$ext")
            break
        fi
    done
done

# Emulator console on 5556 / adb on 5557 (loopback); socat publishes them as 5554 / 5555.
socat TCP-LISTEN:5554,fork,reuseaddr TCP:127.0.0.1:5556 &
socat TCP-LISTEN:5555,fork,reuseaddr TCP:127.0.0.1:5557 &

read -r -a extra_args <<<"${GV_EMU_ARGS:-}"
exec "$ANDROID_SDK_ROOT/emulator/emulator" -avd "$avd" -port 5556 \
    -no-window -no-audio -no-boot-anim -no-snapshot -no-metrics \
    -gpu swiftshader_indirect -accel on \
    -camera-back "$camera_back" -camera-front emulated \
    "${poster_args[@]}" "${extra_args[@]}"
