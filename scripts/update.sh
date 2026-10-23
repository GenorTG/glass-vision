#!/usr/bin/env bash
# Update a venv install: pull the latest code, reinstall the pinned dependencies, check/download models, restart.
#   scripts/update.sh [--service-name NAME] [any scripts/install.sh option]
# Settings in server/config.env are kept (install.sh reuses them in --non-interactive mode).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="glass-vision"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --service-name) SERVICE="$2"; args+=("$1" "$2"); shift ;;
    -h|--help) sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) args+=("$1") ;;
  esac
  shift
done

cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "error: local changes in $ROOT; commit or stash them first (git status)" >&2
  exit 1
fi
before="$(git rev-parse HEAD)"
git pull --ff-only
echo "==> $(git log --oneline -1) (was ${before:0:7})"

# Same steps as a fresh install, reusing the venv and config; --no-systemd: the unit is restarted below instead.
"$ROOT/scripts/install.sh" --non-interactive --no-systemd "${args[@]}"

if command -v systemctl >/dev/null && systemctl --user cat "$SERVICE" >/dev/null 2>&1; then
  unit="$HOME/.config/systemd/user/$SERVICE.service"
  if [[ -f "$unit" && ! -L "$unit" ]]; then  # refresh it from the template (paths may have changed)
    sed "s|@SERVER_DIR@|$ROOT/server|g" "$ROOT/deploy/glass-vision.service.in" >"$unit"
    systemctl --user daemon-reload
  fi
  echo "==> restarting $SERVICE"
  systemctl --user restart "$SERVICE"
  systemctl --user --no-pager status "$SERVICE" | head -n 5
else
  echo "==> no systemd unit $SERVICE: restart your server (server/run -m glass_vision) to use the new version"
fi
