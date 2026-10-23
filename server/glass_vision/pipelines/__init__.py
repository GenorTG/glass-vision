"""Pipeline registry. Entries are imported lazily: a pipeline whose module or dependencies are missing is
skipped with a warning instead of taking the whole server down."""

from __future__ import annotations

import importlib
import logging

from .base import Pipeline

log = logging.getLogger(__name__)

# Order = order in the app's settings.
ALL_PIPELINES: dict[str, str] = {
    "objects": "glass_vision.pipelines.objects:ObjectsPipeline",
    "open_vocab": "glass_vision.pipelines.open_vocab:OpenVocabPipeline",
    "faces": "glass_vision.pipelines.faces:FacesPipeline",
    "text": "glass_vision.pipelines.text:TextPipeline",
    "gestures": "glass_vision.pipelines.gestures:GesturesPipeline",
    "pose": "glass_vision.pipelines.pose:PosePipeline",
}


def available_pipelines() -> dict[str, type[Pipeline]]:
    found: dict[str, type[Pipeline]] = {}
    for pipeline_id, target in ALL_PIPELINES.items():
        module_name, class_name = target.split(":")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            log.warning("pipeline %s unavailable", pipeline_id, exc_info=True)
            continue
        if cls.info.id != pipeline_id:
            raise RuntimeError(f"{target}: info.id {cls.info.id!r} != registry key {pipeline_id!r}")
        found[pipeline_id] = cls
    return found
