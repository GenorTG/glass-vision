"""Run: ./run -m glass_vision. Settings: GV_* environment / server/config.env (see glass_vision/config.py).

`./run` (venv) and the Docker entrypoint set the CUDA/cuDNN environment; the defaults below cover a bare
`python -m glass_vision`.
"""

import logging
import os
import sys

# Before anything imports torch / CUDA:
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:0 = nvidia-smi index 0
# One GPU only; keep older devices hidden so all pipeline kernels use one modern CUDA device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime (it once replaced torch)

import uvicorn

from .config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("glass_vision")

try:
    settings = load_settings()
except ValueError as exc:
    sys.exit(f"glass-vision: bad configuration: {exc}")

if settings.host in ("0.0.0.0", "::") and not settings.token:
    log.warning("listening on ALL interfaces without GV_TOKEN: anyone who can reach this port sees your camera stream")

uvicorn.run(
    "glass_vision.app:app",
    host=settings.host,
    port=settings.port,
    ws_max_size=8 * 1024 * 1024,
    log_config=None,  # keep the basicConfig format (and the token-redaction filters in app.py)
)
