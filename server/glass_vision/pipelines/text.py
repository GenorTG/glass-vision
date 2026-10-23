"""Text (OCR): EasyOCR (CRAFT detector + CRNN recognizer) reading signs, labels and screens.

Slow pipeline: the framework runs it in the background on the newest frame (`interval_ms`). One `Reader` per
language set is cached, so switching back to a set already used is free; the default set is warmed in `load`.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, RunContext

log = logging.getLogger(__name__)

EASYOCR_DIR = MODELS_DIR / "easyocr"
LANGUAGE_SETS: dict[str, list[str]] = {"en": ["en"], "en+pl": ["en", "pl"]}  # en → english_g2, +pl → latin_g2
DEFAULT_LANGUAGES = "en+pl"
MAX_SIDE = 1280  # downscale before OCR; boxes are scaled back to the original pixels
# readtext tuning (measured on the 3090): batch_size=16 recognizes all crops in one pass instead of one call per
# word (960x720 sign 67 → 55 ms, 810x1080 street photo 130 → 92 ms). canvas_size only caps the detector input,
# which MAX_SIDE already does; cudnn.benchmark was not a clear win and is process-global, so it stays off.
READTEXT_KWARGS: dict[str, Any] = {"batch_size": 16, "canvas_size": MAX_SIDE, "paragraph": False}


def _check_cudnn_usable() -> None:
    """torch's Python cuDNN init refuses (process-wide) when *any* visible GPU is below SM 7.5, which breaks the
    recognizer's LSTM (`flatten_parameters`). Fail with a fix instead of a cryptic cuDNN error deep in EasyOCR."""
    old = [
        f"cuda:{i} {torch.cuda.get_device_name(i)}"
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(i) < (7, 5)
    ]
    if old:
        raise RuntimeError(
            f"OCR needs cuDNN, which torch disables while a GPU below SM 7.5 is visible ({', '.join(old)}); "
            "hide it with CUDA_VISIBLE_DEVICES (set before CUDA initializes)"
        )


class TextPipeline(Pipeline):
    info = PipelineInfo(
        id="text",
        name="Text (OCR)",
        description="Reads text on signs, labels and screens (English + Polish).",
        default_enabled=False,
        uses_global_conf=False,
        interval_ms=400,
        options=[
            OptionSpec("languages", "Languages", "choice", DEFAULT_LANGUAGES, choices=list(LANGUAGE_SETS)),
            OptionSpec("min_conf", "Minimum confidence", "number", 0.3, min=0.1, max=0.9, step=0.05),
        ],
    )

    def __init__(self) -> None:
        self._device: str | bool = False
        self._readers: dict[str, Any] = {}  # language-set key → easyocr.Reader

    def load(self, device: str) -> None:
        if device.startswith("cuda"):
            try:
                _check_cudnn_usable()
            except RuntimeError as exc:
                log.warning("text: %s — running OCR on the CPU instead (slow)", exc)
                device = "cpu"
        self._device = device if device.startswith("cuda") else False
        reader = self._reader(DEFAULT_LANGUAGES)
        reader.readtext(np.full((64, 256, 3), 255, np.uint8), **READTEXT_KWARGS)  # warm up CUDA kernels

    def unload(self) -> None:
        self._readers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reader(self, languages: str) -> Any:
        reader = self._readers.get(languages)
        if reader is None:
            import easyocr  # heavy import (torch, scipy…): only when the pipeline is actually used

            EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
            log.info("text: building EasyOCR reader %s on %s", languages, self._device or "cpu")
            reader = easyocr.Reader(
                LANGUAGE_SETS[languages],
                gpu=self._device,
                model_storage_directory=str(EASYOCR_DIR),
                user_network_directory=str(EASYOCR_DIR / "user_network"),
                download_enabled=True,
                verbose=False,
            )
            # EasyOCR wraps both nets in DataParallel, which scatters batches over *every* visible GPU; unwrap so
            # they stay on the requested device.
            for attr in ("detector", "recognizer"):
                net = getattr(reader, attr)
                if isinstance(net, torch.nn.DataParallel):
                    setattr(reader, attr, net.module)
            self._readers[languages] = reader
        return reader

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        reader = self._reader(ctx.options["languages"])
        min_conf = float(ctx.options["min_conf"])
        h, w = image.shape[:2]

        scale = min(1.0, MAX_SIDE / max(h, w))
        small = (
            image
            if scale == 1.0
            else cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # EasyOCR treats ndarrays as RGB

        out: list[Annotation] = []
        for quad, text, conf in reader.readtext(rgb, **READTEXT_KWARGS):
            label = text.strip()
            if not label or conf < min_conf:
                continue
            pts = np.asarray(quad, dtype=np.float64) / scale
            x1, y1 = np.clip(pts.min(axis=0), 0, (w, h))
            x2, y2 = np.clip(pts.max(axis=0), 0, (w, h))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Annotation(box=(float(x1), float(y1), float(x2), float(y2)), conf=float(conf), label=label))
        return out
