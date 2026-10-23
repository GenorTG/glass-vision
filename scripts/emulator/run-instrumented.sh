#!/usr/bin/env bash
# Run the app's instrumented tests on the test emulator (start it first with start.sh) and
# collect the reports + logcat under scripts/emulator/out/<timestamp>/.
#
#   scripts/emulator/run-instrumented.sh [extra gradle args...]
#   e.g. ... -Pandroid.testInstrumentationRunnerArguments.class=pl.genor.glassvision.FooTest
#
# GV_ANDROID_DIR overrides the Gradle project (default <repo>/android), e.g. a worktree.
# Exit status = gradle's.
set -euo pipefail
# shellcheck source=scripts/emulator/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

gv_container_running || gv_die "emulator not running; run scripts/emulator/start.sh first"
gv_connect 30 || gv_die "adb cannot reach $GV_EMU_SERIAL"

android_dir=${GV_ANDROID_DIR:-$GV_REPO_ROOT/android}
[[ -x $android_dir/gradlew ]] || gv_die "no gradlew in $android_dir"
out=$GV_EMU_DIR/out/$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
gv_adb logcat -c || true

status=0
# ANDROID_SERIAL limits AGP's device provider to the emulator (other devices may be attached).
(cd "$android_dir" && ANDROID_SERIAL=$GV_EMU_SERIAL ./gradlew connectedDebugAndroidTest "$@") || status=$?

build=$android_dir/app/build
for dir in reports/androidTests/connected outputs/androidTest-results/connected; do
    if [[ -d $build/$dir ]]; then
        mkdir -p "$out/$dir"
        cp -r "$build/$dir/." "$out/$dir/"
    fi
done
gv_adb logcat -d >"$out/logcat.txt" 2>/dev/null || true
"$GV_EMU_DIR/screenshot.sh" "$out/final-screen.png" >/dev/null 2>&1 || true

gv_log "gradle exit $status; reports in $out"
for html in "$out"/reports/androidTests/connected/{,debug/}index.html; do
    [[ -f $html ]] && gv_log "HTML report: $html"
done
exit "$status"
