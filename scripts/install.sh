#!/usr/bin/env bash
# glass-vision server installer (Linux; macOS best-effort, CPU only). Idempotent: rerun it any time.
#
# Creates a self-contained venv in server/.venv, installs PyTorch for your GPU (or CPU), the pinned dependencies in
# the one order that works, the models, and server/config.env. Run with --help for every option.
set -euo pipefail

# Where release APKs are published (GitHub "owner/name"). Replaced when the project is published.
GV_REPO="${GV_REPO:-GenorTG/glass-vision}"

TORCH_VERSION="2.12.0"
TORCHVISION_VERSION="0.27.0"
TORCH_INDEX="https://download.pytorch.org/whl"
MIN_FREE_GB=10

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
VENV="$SERVER/.venv"
CONFIG="$SERVER/config.env"

usage() {
  cat <<EOF
Usage: scripts/install.sh [options]

Interactive by default: asks for the listen address, port, token and optional steps. With --non-interactive every
question takes its flag, else the value already in server/config.env, else the default shown in [brackets].

Server settings (written to server/config.env, chmod 600):
  --host ADDR            listen address [127.0.0.1]. Use your LAN or Tailscale IP so the phone can reach it.
                         0.0.0.0 = every interface (only with a token, never on a network you don't trust)
  --port N               listen port [8770]
  --token SECRET         require this token (the app sends it)
  --generate-token       generate a random token
  --no-token             no token (anyone who can reach the port can use the server)
  --device DEV           auto | cpu | cuda:N [auto]
  --models-dir DIR       where models go [server/models]
  --data-dir DIR         face gallery, captures, distributed APK [server/data]

Install:
  --cuda VARIANT         PyTorch build: auto | cu130 | cu126 | cpu [auto: from the NVIDIA driver version]
  --python PATH          Python 3.11 or 3.12 to build the venv with [first found: python3.12, python3.11, python3]
  --recreate-venv        move an existing server/.venv aside (server/.venv.old-<time>) and build a fresh one
  --skip-models          don't download models (the server fetches some lazily; run scripts/download_models.py later)
  --systemd / --no-systemd   install + start a systemd --user service [ask; non-interactive: no]
  --service-name NAME    systemd unit name [glass-vision]
  --apk / --no-apk       download the latest app release (APK) from github.com/$GV_REPO so this server can
                         hand it out to phones (/app.apk, in-app updates) [ask; non-interactive: no]
  --non-interactive      never prompt
  -h, --help             this help
EOF
}

# --- arguments -------------------------------------------------------------------------------------------------
INTERACTIVE=1
OPT_HOST="" OPT_PORT="" OPT_TOKEN="" TOKEN_MODE="" OPT_DEVICE="" OPT_MODELS="" OPT_DATA=""
CUDA="auto" PYTHON="" RECREATE=0 SKIP_MODELS=0 SYSTEMD="" SERVICE="glass-vision" APK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) OPT_HOST="$2"; shift ;;
    --port) OPT_PORT="$2"; shift ;;
    --token) OPT_TOKEN="$2"; TOKEN_MODE="set"; shift ;;
    --generate-token) TOKEN_MODE="generate" ;;
    --no-token) TOKEN_MODE="none" ;;
    --device) OPT_DEVICE="$2"; shift ;;
    --models-dir) OPT_MODELS="$2"; shift ;;
    --data-dir) OPT_DATA="$2"; shift ;;
    --cuda) CUDA="$2"; shift ;;
    --python) PYTHON="$2"; shift ;;
    --recreate-venv) RECREATE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --systemd) SYSTEMD=1 ;;
    --no-systemd) SYSTEMD=0 ;;
    --service-name) SERVICE="$2"; shift ;;
    --apk) APK=1 ;;
    --no-apk) APK=0 ;;
    --non-interactive) INTERACTIVE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[[ -t 0 ]] || INTERACTIVE=0

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
ask() { # ask "question" default → echo answer
  local answer
  if [[ $INTERACTIVE -eq 0 ]]; then echo "$2"; return; fi
  read -r -p "$1 [$2]: " answer </dev/tty
  echo "${answer:-$2}"
}
yes_no() { # yes_no "question" default(y/n) → exit status
  local answer
  answer="$(ask "$1 (y/n)" "$2")"
  [[ "$answer" =~ ^[Yy] ]]
}
config_get() { # value of KEY in the existing config.env ("" if none)
  [[ -f "$CONFIG" ]] || return 0
  sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=[[:space:]]*//p" "$CONFIG" | tail -n1 | sed -E "s/^[\"']//; s/[\"']$//"
}
OS="$(uname -s)"
START=$(date +%s)

# --- 1. Python ---------------------------------------------------------------------------------------------------
say "Checking Python"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 12)))'; then
      PYTHON="$(command -v "$candidate")"; break
    fi
  done
  [[ -n "$PYTHON" ]] || die "need Python 3.11 or 3.12 (found: $(python3 -V 2>&1 || echo none)). Debian/Ubuntu: sudo apt install python3.11 python3.11-venv (or use Docker, see README)"
fi
"$PYTHON" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 12)))' \
  || die "$PYTHON is $("$PYTHON" -V 2>&1); need 3.11 or 3.12 (the pinned numpy/scipy/onnxruntime need >= 3.11)"
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null \
  || die "$PYTHON can't create venvs. Debian/Ubuntu: sudo apt install python3-venv (or python$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')-venv)"
echo "    $PYTHON ($("$PYTHON" -V 2>&1))"

free_gb=$(df -Pk "$SERVER" | awk 'NR==2 {print int($4 / 1048576)}')
[[ "$free_gb" -ge "$MIN_FREE_GB" ]] || warn "only ${free_gb} GB free under $SERVER; a GPU install needs ~8 GB (PyTorch+CUDA ~5 GB, models ~1.2 GB)"

# --- 2. GPU → PyTorch build ---------------------------------------------------------------------------------------
# PyTorch 2.12 publishes CPU, CUDA 12.6, 13.0 and 13.2 builds (https://pytorch.org/get-started/locally/, compatibility
# matrix in https://github.com/pytorch/pytorch/blob/main/RELEASE.md). The wheels bundle the CUDA runtime + cuDNN, so
# only the NVIDIA *driver* matters: CUDA 13.x needs driver >= 580, CUDA 12.x >= 525 (minor-version compatibility,
# https://docs.nvidia.com/deploy/cuda-compatibility/). cu130 is the tested one; onnxruntime-gpu must match the CUDA
# major (1.31 → CUDA 13, 1.26 → CUDA 12).
say "Detecting GPU"
if [[ "$CUDA" == "auto" ]]; then
  CUDA="cpu"
  if [[ "$OS" == "Linux" ]] && command -v nvidia-smi >/dev/null; then
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    major="${driver%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]]; then
      if (( major >= 580 )); then CUDA="cu130"; elif (( major >= 525 )); then CUDA="cu126"; else
        warn "NVIDIA driver $driver is too old for PyTorch $TORCH_VERSION (needs >= 525): installing the CPU build"
      fi
      echo "    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1), driver $driver → $CUDA"
    fi
  fi
  [[ "$CUDA" == "cpu" ]] && echo "    no usable NVIDIA GPU → CPU build (works, but recognition is slow)"
fi
case "$CUDA" in
  cu130) ORT_REQ="requirements-cu13.txt" ;;
  cu126) ORT_REQ="requirements-cu12.txt" ;;
  cpu) ORT_REQ="requirements-cpu.txt" ;;
  *) die "--cuda must be auto, cu130, cu126 or cpu (got $CUDA)" ;;
esac

# --- 3. venv -----------------------------------------------------------------------------------------------------
say "Creating venv $VENV"
if [[ -d "$VENV" ]]; then
  reuse=1
  if grep -qi '^include-system-site-packages *= *true' "$VENV/pyvenv.cfg" 2>/dev/null; then
    [[ $RECREATE -eq 1 ]] || die "$VENV uses --system-site-packages (a developer venv); rerun with --recreate-venv to replace it (moved aside, not deleted)"
    reuse=0
  fi
  [[ -x "$VENV/bin/python" ]] || reuse=0
  if [[ $RECREATE -eq 1 || $reuse -eq 0 ]]; then
    aside="$VENV.old-$(date +%Y%m%d-%H%M%S)"
    mv "$VENV" "$aside"
    echo "    moved the old venv to $aside (delete it when the new one works)"
  else
    echo "    reusing it"
  fi
fi
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
pip() { "$PY" -m pip --disable-pip-version-check "$@"; }
pip install --quiet --upgrade pip

# --- 4. packages (order matters: AGENTS.md "Gotchas") ------------------------------------------------------------
cd "$SERVER"
constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" >"$constraints"

say "Installing PyTorch $TORCH_VERSION ($CUDA)"
if [[ "$OS" == "Darwin" ]]; then
  pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"  # PyPI's macOS wheels (CPU/MPS)
else
  pip install --index-url "$TORCH_INDEX/$CUDA" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
fi

say "Installing pinned dependencies"
# -c: nothing below may replace the torch/torchvision build installed above.
pip install -r requirements.txt -c "$constraints"
# --no-deps: their declared deps would pull a PyPI torch, opencv-contrib (shadows cv2) or CPU onnxruntime.
pip install --no-deps -r requirements-nodeps.txt
# onnxruntime last, alone, so no other package's files overwrite it (insightface's CPU onnxruntime used to).
pip uninstall --yes --quiet onnxruntime onnxruntime-gpu 2>/dev/null || true
pip install --no-deps --force-reinstall -r "$ORT_REQ"

# onnxruntime-gpu dlopens unversioned libcudnn*.so and links libcublas/libcudart by soname, but the CUDA libraries
# come from torch's nvidia-* wheels inside the venv (no system CUDA needed). Expose them all in .venv/ort-libs,
# which ./run puts on LD_LIBRARY_PATH.
if [[ "$CUDA" != "cpu" ]]; then
  say "Linking CUDA libraries for onnxruntime"
  site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  shim="$VENV/ort-libs"
  mkdir -p "$shim"
  find "$shim" -maxdepth 1 -type l -delete
  shopt -s nullglob
  linked=0
  for lib in "$site"/nvidia/*/lib/lib*.so*; do
    ln -sf "$lib" "$shim/$(basename "$lib")"; linked=$((linked + 1))
  done
  for lib in "$site"/nvidia/cudnn/lib/libcudnn*.so.[0-9]*; do
    base="$(basename "$lib")"; ln -sf "$lib" "$shim/${base%%.so.*}.so"
  done
  shopt -u nullglob
  [[ $linked -gt 0 ]] || warn "no nvidia/*/lib in $site: onnxruntime (faces) will run on the CPU"
  echo "    $linked libraries → $shim"
fi

say "Checking the install"
"$SERVER/run" - <<'EOF'
import onnxruntime, torch, torchvision
import clip, easyocr, insightface, mediapipe, ultralytics  # noqa: F401  (all import cleanly)
from torchvision.ops import nms

nms(torch.zeros((1, 4)), torch.zeros(1), 0.5)  # the op a mismatched torchvision lacks
cuda = torch.cuda.is_available()
print(f"    torch {torch.__version__} (CUDA {'yes: ' + torch.cuda.get_device_name(0) if cuda else 'no'}), "
      f"torchvision {torchvision.__version__}, onnxruntime {onnxruntime.__version__} {onnxruntime.get_available_providers()}")
EOF

# --- 5. configuration --------------------------------------------------------------------------------------------
say "Configuring"
detect_addresses() {
  local ts=""
  if command -v tailscale >/dev/null; then ts="$(tailscale ip -4 2>/dev/null | head -n1 || true)"; fi
  [[ -n "$ts" ]] && echo "$ts tailscale"
  if [[ "$OS" == "Linux" ]] && command -v ip >/dev/null; then
    ip -4 -o addr show scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1], $2}' | grep -v "^${ts:-none} " || true
  elif [[ "$OS" == "Darwin" ]]; then
    for i in en0 en1; do a="$(ipconfig getifaddr "$i" 2>/dev/null || true)"; [[ -n "$a" ]] && echo "$a $i"; done
  fi
}

host_default="$(config_get GV_HOST)"; host_default="${host_default:-127.0.0.1}"
if [[ -n "$OPT_HOST" ]]; then
  HOST="$OPT_HOST"
elif [[ $INTERACTIVE -eq 1 ]]; then
  echo "Which address should the server listen on? The phone must be able to reach it."
  addrs=()
  while IFS= read -r line; do addrs+=("$line"); done < <(detect_addresses)  # (no mapfile: macOS bash 3.2)
  addrs+=("127.0.0.1 this machine only (testing)" "0.0.0.0 ALL interfaces (only with a token; never on untrusted networks)")
  for i in "${!addrs[@]}"; do printf '  %d) %s\n' "$((i + 1))" "${addrs[$i]}"; done
  choice="$(ask "Number or address" "$host_default")"
  if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#addrs[@]} )); then
    HOST="${addrs[$((choice - 1))]%% *}"
  else
    HOST="$choice"
  fi
else
  HOST="$host_default"
fi

port_default="$(config_get GV_PORT)"; PORT="${OPT_PORT:-$(ask "Port" "${port_default:-8770}")}"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "bad port: $PORT"

gen_token() { "$PY" -c 'import secrets; print(secrets.token_hex(16))'; }
existing_token="$(config_get GV_TOKEN)"
case "$TOKEN_MODE" in
  set) TOKEN="$OPT_TOKEN" ;;
  generate) TOKEN="$(gen_token)" ;;
  none) TOKEN="" ;;
  *)
    if [[ $INTERACTIVE -eq 1 ]]; then
      if [[ -n "$existing_token" ]] && yes_no "Keep the existing token" y; then TOKEN="$existing_token"
      elif yes_no "Protect the server with a token (recommended)" y; then
        TOKEN="$(ask "Token (Enter = generate one)" "")"; TOKEN="${TOKEN:-$(gen_token)}"
      else TOKEN=""; fi
    else
      TOKEN="$existing_token"
    fi ;;
esac
if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
  [[ -n "$TOKEN" ]] || warn "listening on ALL interfaces without a token: anyone who can reach port $PORT can use your server"
  warn "never forward this port to the internet; use your LAN or Tailscale (see README Security)"
fi
DEVICE="${OPT_DEVICE:-$(config_get GV_DEVICE)}"; DEVICE="${DEVICE:-auto}"
MODELS_DIR="${OPT_MODELS:-$(config_get GV_MODELS_DIR)}"
DATA_DIR="${OPT_DATA:-$(config_get GV_DATA_DIR)}"

(
  umask 077
  {
    echo "# glass-vision server settings (scripts/install.sh, $(date -u +%Y-%m-%dT%H:%MZ)). Env vars with the same names win."
    echo "# All keys: server/glass_vision/config.py. Apply changes: restart the server."
    echo "GV_HOST=$HOST"
    echo "GV_PORT=$PORT"
    echo "GV_TOKEN=$TOKEN"
    echo "GV_DEVICE=$DEVICE"
    [[ -n "$MODELS_DIR" ]] && echo "GV_MODELS_DIR=$MODELS_DIR"
    [[ -n "$DATA_DIR" ]] && echo "GV_DATA_DIR=$DATA_DIR"
    # Keep any other GV_* keys the user added by hand.
    if [[ -f "$CONFIG" ]]; then
      grep -E '^[[:space:]]*(export[[:space:]]+)?GV_[A-Z_]+=' "$CONFIG" \
        | grep -v -E 'GV_(HOST|PORT|TOKEN|DEVICE|MODELS_DIR|DATA_DIR)=' || true
    fi
  } >"$CONFIG.new"
  mv "$CONFIG.new" "$CONFIG"
)
chmod 600 "$CONFIG"
echo "    wrote $CONFIG"
eval "$("$PY" - <<'EOF'
import shlex, sys
sys.path.insert(0, ".")
from glass_vision.config import load_settings
s = load_settings()
print(f"MODELS_DIR={shlex.quote(str(s.models_dir))} DATA_DIR={shlex.quote(str(s.data_dir))}")
EOF
)"

# --- 6. models ---------------------------------------------------------------------------------------------------
if [[ $SKIP_MODELS -eq 0 ]]; then
  say "Downloading models into $MODELS_DIR"
  "$PY" "$ROOT/scripts/download_models.py" || die "model download failed: rerun scripts/install.sh (downloads resume)"
fi

# --- 7. app APK (optional) ---------------------------------------------------------------------------------------
if [[ -z "$APK" ]]; then
  APK=0
  [[ $INTERACTIVE -eq 1 && "$GV_REPO" != OWNER/* ]] && yes_no "Download the latest app release so this server can hand it out to phones" y && APK=1
fi
if [[ $APK -eq 1 ]]; then
  if [[ "$GV_REPO" == OWNER/* ]]; then
    warn "no release repository configured (GV_REPO=$GV_REPO): skipping the APK download"
  else
    say "Downloading the latest app release from github.com/$GV_REPO"
    "$PY" "$ROOT/scripts/fetch_release_apk.py" --repo "$GV_REPO" --dest "$DATA_DIR/app" \
      || warn "APK download failed; the server runs fine without it (no /app.apk for phones)"
  fi
fi

# --- 8. systemd (optional) ---------------------------------------------------------------------------------------
UNIT=""
if [[ "$OS" == "Linux" ]] && command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  if [[ -z "$SYSTEMD" ]]; then
    SYSTEMD=0
    [[ $INTERACTIVE -eq 1 ]] && yes_no "Install as a systemd user service (starts now and on login)" y && SYSTEMD=1
  fi
  if [[ "$SYSTEMD" -eq 1 ]]; then
    say "Installing systemd --user unit $SERVICE"
    UNIT="$HOME/.config/systemd/user/$SERVICE.service"
    mkdir -p "$(dirname "$UNIT")"
    [[ -L "$UNIT" ]] && rm "$UNIT"  # older installs symlinked the unit from the repo
    sed "s|@SERVER_DIR@|$SERVER|g" "$ROOT/deploy/glass-vision.service.in" >"$UNIT"
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE" >/dev/null
    systemctl --user restart "$SERVICE"
    echo "    started; logs: journalctl --user -u $SERVICE -f"
    echo "    to run it at boot without logging in: loginctl enable-linger $USER"
  fi
elif [[ "$SYSTEMD" == 1 ]]; then
  warn "systemd --user isn't available here: start the server with server/run -m glass_vision"
fi

# --- done --------------------------------------------------------------------------------------------------------
say "Done in $(( $(date +%s) - START )) s"
echo
if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
  urls=()
  while IFS= read -r a; do urls+=("ws://${a%% *}:$PORT/ws"); done < <(detect_addresses)
  [[ ${#urls[@]} -gt 0 ]] || urls=("ws://127.0.0.1:$PORT/ws")
else
  urls=("ws://$HOST:$PORT/ws")
fi
echo "  In the app (Connect to your server / Settings → Connection):"
for u in "${urls[@]}"; do a="${u#ws://}"; echo "    Server address: ${a%/ws}   (full URL: $u)"; done
if [[ -n "$TOKEN" ]]; then echo "    Access token:   $TOKEN"; else echo "    Access token:   (none)"; fi
echo
if [[ -z "$UNIT" ]]; then
  echo "  Start the server:  $SERVER/run -m glass_vision"
fi
echo "  Check it:          curl http://${urls[0]#ws://}" | sed 's|/ws$|/health|'
echo "  Settings:          $CONFIG"
