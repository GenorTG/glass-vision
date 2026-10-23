"""Shared plumbing for ultralytics-based pipelines (objects, pose, open_vocab)."""

from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Results

from ..tracking import LOW_CONF
from .base import Annotation, RunContext

WARMUP_IMAGE = np.zeros((480, 640, 3), np.uint8)


def predict_kwargs(ctx: RunContext, device: str) -> dict[str, object]:
    # Detect down to the tracker floor; the client's conf only gates new tracks (see tracking.py).
    return {"conf": LOW_CONF, "imgsz": ctx.imgsz, "device": device, "verbose": False}


def to_annotations(result: Results) -> list[Annotation]:
    boxes = result.boxes.cpu().numpy()
    names = result.names
    keypoints = result.keypoints.cpu().numpy() if result.keypoints is not None else None
    out: list[Annotation] = []
    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True)):
        points = None
        if keypoints is not None:
            xy = keypoints.xy[i]
            vis = keypoints.conf[i] if keypoints.conf is not None else np.ones(len(xy))
            points = [(float(x), float(y), float(v)) for (x, y), v in zip(xy, vis, strict=True)]
        out.append(Annotation((float(x1), float(y1), float(x2), float(y2)), float(conf), names[int(cls)], None, points))
    return out
