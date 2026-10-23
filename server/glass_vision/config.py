"""Server settings: environment variables, then an optional env file (`server/config.env`), then defaults.

Every key is `GV_*`; environment variables win over the file, so a systemd unit / docker `-e` can override a single
value. `GV_ENV_FILE` points at another file (empty string = none). The file is plain `KEY=value` lines (`#` comments,
optional `export ` prefix, optional single/double quotes) — the same file `scripts/install.sh` writes.

| key | default | meaning |
|---|---|---|
| GV_HOST | 127.0.0.1 | listen address (installer sets the real one: LAN / Tailscale IP) |
| GV_PORT | 8770 | listen port |
| GV_TOKEN | (none) | shared secret; when set every request but `/health` needs it (docs/PROTOCOL.md "Auth") |
| GV_DEVICE | auto | `auto` → `cuda:0` if torch can run on it, else `cpu`; or an explicit `cuda:N` / `cpu` |
| GV_MODELS_DIR | server/models | model weights (`scripts/download_models.py` fills it) |
| GV_DATA_DIR | server/data | persistent user data (face gallery, captures, distributed APK) |
| GV_APK_PATH | see `default_apk_path` | APK served at `/app.apk` for in-app updates |
| GV_IDLE_UNLOAD_S | 600 | unload a pipeline nobody used for this many seconds |
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent
DEFAULT_ENV_FILE = SERVER_ROOT / "config.env"
# Debug build output of android/ — what a developer checkout serves when no release APK was installed.
DEV_APK = REPO_ROOT / "android" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
APK_NAME = "glass-vision.apk"


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8770
    token: str | None = None
    device: str = "auto"  # unresolved; see resolve_device()
    models_dir: Path = SERVER_ROOT / "models"
    data_dir: Path = SERVER_ROOT / "data"
    apk_path: Path = SERVER_ROOT / "data" / "app" / APK_NAME
    idle_unload_s: float = 600.0


def parse_env_file(text: str) -> dict[str, str]:
    """`KEY=value` lines → dict. Blank lines and `#` comments are skipped; surrounding quotes are stripped."""
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            raise ValueError(f"env file line {lineno}: expected KEY=value, got {raw!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def default_apk_path(data_dir: Path) -> Path:
    """`<data>/app/glass-vision.apk` (installer / release download); a developer checkout without one serves its
    own debug build instead."""
    installed = data_dir / "app" / APK_NAME
    if not installed.is_file() and DEV_APK.is_file():
        return DEV_APK
    return installed


def load_settings(environ: Mapping[str, str] | None = None, env_file: Path | None = None) -> Settings:
    """Read settings. Raises ValueError (with the offending key) on malformed values."""
    env = dict(os.environ if environ is None else environ)
    if env_file is None:
        override = env.get("GV_ENV_FILE")
        env_file = DEFAULT_ENV_FILE if override is None else (Path(override) if override else None)
    merged: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        merged.update(parse_env_file(env_file.read_text()))
    merged.update({k: v for k, v in env.items() if k.startswith("GV_")})

    def get(key: str) -> str | None:
        value = merged.get(key, "").strip()
        return value or None

    def number(key: str, default: float, *, integer: bool = False) -> float:
        raw = get(key)
        if raw is None:
            return default
        try:
            return int(raw) if integer else float(raw)
        except ValueError:
            raise ValueError(f"{key} must be a number, got {raw!r}") from None

    defaults = Settings()
    models_dir = Path(get("GV_MODELS_DIR") or defaults.models_dir).expanduser()
    data_dir = Path(get("GV_DATA_DIR") or defaults.data_dir).expanduser()
    apk = get("GV_APK_PATH")
    port = int(number("GV_PORT", defaults.port, integer=True))
    if not 0 < port < 65536:
        raise ValueError(f"GV_PORT must be 1..65535, got {port}")
    device = get("GV_DEVICE") or defaults.device
    if device != "auto" and device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"GV_DEVICE must be auto, cpu or cuda[:N], got {device!r}")
    return Settings(
        host=get("GV_HOST") or defaults.host,
        port=port,
        token=get("GV_TOKEN"),
        device=device,
        models_dir=models_dir,
        data_dir=data_dir,
        apk_path=Path(apk).expanduser() if apk else default_apk_path(data_dir),
        idle_unload_s=number("GV_IDLE_UNLOAD_S", defaults.idle_unload_s),
    )


def resolve_device(device: str) -> str:
    """`auto` → `cuda:0` if torch can actually run a kernel there (driver present *and* the wheel has kernels for
    this GPU's architecture), else `cpu`. Explicit values pass through unchanged."""
    if device != "auto":
        return device
    import torch  # heavy: only once the server really starts

    if not torch.cuda.is_available():
        log.info("no usable CUDA GPU: running on CPU (slow but works)")
        return "cpu"
    try:
        (torch.ones(1, device="cuda:0") + 1).item()
    except RuntimeError as exc:  # e.g. "no kernel image is available for execution on the device"
        log.warning("CUDA GPU present but unusable by this torch build (%s): running on CPU", exc)
        return "cpu"
    return "cuda:0"


settings = load_settings()
