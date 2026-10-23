#!/usr/bin/env bash
# Stop and remove the test emulator container (its AVD/userdata lives in the container layer,
# so this wipes the device too).
#
#   scripts/emulator/stop.sh [--purge]
#
# --purge also deletes the image (~5.7 GB) and its build cache; the next start.sh rebuilds it
# (~1.5 min on a fast link).
set -euo pipefail
# shellcheck source=scripts/emulator/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

purge=0
case ${1:-} in
--purge) purge=1 ;;
'') ;;
*) gv_die "usage: $0 [--purge]" ;;
esac

"$ADB" disconnect "$GV_EMU_SERIAL" >/dev/null 2>&1 || true
if docker inspect "$GV_EMU_CONTAINER" >/dev/null 2>&1; then
    docker rm -f -v "$GV_EMU_CONTAINER" >/dev/null
    gv_log "removed container $GV_EMU_CONTAINER"
fi

if ((purge)) && docker image inspect "$GV_EMU_IMAGE" >/dev/null 2>&1; then
    docker rmi "$GV_EMU_IMAGE" >/dev/null
    # BuildKit keeps every layer (incl. the 5 GB download) as build cache after rmi. Drop only
    # the entries this Dockerfile created, recognised by their step text. A parent entry can't be
    # pruned while its child exists, so repeat until none are left.
    for _ in 1 2 3 4; do
        mapfile -t ids < <(docker buildx du --verbose 2>/dev/null |
            awk '/^ID:/{id=$2} /^Description:/ && /ANDROID_SDK_ROOT\/platforms|gv-emulator-entrypoint|libxkbfile1/{print id}')
        ((${#ids[@]})) || break
        for id in "${ids[@]}"; do
            docker builder prune -f --filter "id=$id" >/dev/null 2>&1 || true
        done
    done
    gv_log "removed image $GV_EMU_IMAGE and its build cache"
fi
