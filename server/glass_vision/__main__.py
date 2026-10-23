"""Run: ./run -m glass_vision. Settings: GV_* environment / server/config.env (see glass_vision/config.py).

`./run` (venv) and the Docker entrypoint set the CUDA/cuDNN environment; the defaults below cover a bare
`python -m glass_vision`.
"""

import logging
import os
import sys

# Before anything imports torch / CUDA:
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # cuda:0 = nvidia-smi index 0
# One GPU only: torch disables cuDNN process-wide while any visible GPU is below SM 7.5 (EasyOCR then crashes).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime (it once replaced torch)
# torch 2.12 + CUDA 13: cuBLASLt's fused-bias addmm hit "illegal memory access" in EasyOCR's recognizer head (server/run).
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

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
