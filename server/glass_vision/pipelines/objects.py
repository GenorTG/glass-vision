"""80 COCO classes with YOLO11s: the always-on default."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ._ultralytics import WARMUP_IMAGE, predict_kwargs, to_annotations
from .base import MODELS_DIR, Annotation, Pipeline, PipelineInfo, RunContext


class ObjectsPipeline(Pipeline):
    info = PipelineInfo(
        id="objects",
        name="Objects (80 everyday classes)",
        description="People, vehicles, animals, furniture, food, electronics… (YOLO11s, COCO).",
        default_enabled=True,
        uses_global_conf=True,
    )

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        self._device = device
        self._model = YOLO(str(MODELS_DIR / "yolo11s.pt"))
        self._model.predict(WARMUP_IMAGE, device=device, verbose=False)

    def unload(self) -> None:
        self._model = None

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        assert self._model is not None, "not loaded"
        return to_annotations(self._model.predict(image, **predict_kwargs(ctx, self._device))[0])
