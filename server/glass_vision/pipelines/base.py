"""Pipeline plug-in contract. One subclass per recognizer; see docs/PROTOCOL.md (v3) for the wire side.

A pipeline is a *shared* model (one instance per server process) with per-connection options.
The framework (`glass_vision.session`) owns everything around it: lazy load/unload, a per-pipeline lock
(so `run` is never called concurrently on one instance), background scheduling for slow pipelines,
tracking/smoothing, normalization of coordinates, and the wire format. A pipeline only turns one BGR
image into a list of `Annotation`s in *pixel* coordinates.

Writing one:
1. Subclass `Pipeline`, fill `info` (a class attribute), implement `load`, `run` (and `unload`, `command` if needed).
2. Download weights in `load` into `MODELS_DIR` (never the cwd); keep persistent user data in `DATA_DIR / <id>`.
3. Register it in `glass_vision/pipelines/__init__.py` (`ALL_PIPELINES`).
4. Add `server/tests/test_pipeline_<id>.py` that loads the real model and runs it on a real image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ..config import settings

MODELS_DIR = settings.models_dir  # weights: GV_MODELS_DIR, default server/models (gitignored)
DATA_DIR = settings.data_dir  # persistent user data (face gallery…): GV_DATA_DIR, default server/data (gitignored)

OptionType = Literal["bool", "text", "number", "choice"]
Point = tuple[float, float, float]  # x, y (pixels), visibility/confidence 0..1


@dataclass(frozen=True)
class OptionSpec:
    key: str
    label: str
    type: OptionType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None

    def coerce(self, value: Any) -> Any:
        """Validate a client-sent value; raises ValueError with a user-readable message."""
        if self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.key} must be true/false")
            return value
        if self.type == "text":
            if not isinstance(value, str) or len(value) > 500:
                raise ValueError(f"{self.key} must be text (max 500 chars)")
            return value
        if self.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.key} must be a number")
            if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
                raise ValueError(f"{self.key} must be in [{self.min}, {self.max}]")
            return float(value)
        if self.type == "choice":
            if value not in (self.choices or []):
                raise ValueError(f"{self.key} must be one of {self.choices}")
            return value
        raise AssertionError(self.type)

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ActionSpec:
    id: str
    label: str
    args: list[OptionSpec] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "args": [a.to_json() for a in self.args]}


@dataclass(frozen=True)
class PipelineInfo:
    id: str  # [a-z_]+, stable: used in configs, track ids and DATA_DIR
    name: str
    description: str
    default_enabled: bool = False
    # True: detect down to LOW_CONF and let the client's global `conf` gate new tracks (YOLO family).
    # False: `run` returns only what it considers real; the tracker accepts everything it gets.
    uses_global_conf: bool = False
    # 0 = run on every frame, inline (must be fast, ~<20 ms). >0 = slow: run in the background on the newest
    # frame at most every `interval_ms`; results are merged into replies as they arrive.
    interval_ms: int = 0
    options: list[OptionSpec] = field(default_factory=list)
    actions: list[ActionSpec] = field(default_factory=list)
    skeleton: list[tuple[int, int]] | None = None  # edges between `Annotation.points` indices

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default_enabled": self.default_enabled,
            "uses_global_conf": self.uses_global_conf,
            "interval_ms": self.interval_ms,
            "options": [o.to_json() for o in self.options],
            "actions": [a.to_json() for a in self.actions],
            "skeleton": [list(e) for e in self.skeleton] if self.skeleton else None,
        }

    def default_options(self) -> dict[str, Any]:
        return {o.key: o.default for o in self.options}


@dataclass
class Annotation:
    """One thing found in one frame, in pixel coordinates of that frame."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels
    conf: float  # 0..1
    label: str  # what is shown and voted on across frames: class, person name, gesture, OCR text
    detail: str | None = None  # optional second line (score, …)
    points: list[Point] | None = None  # keypoints in pixels, indexed by `PipelineInfo.skeleton`


@dataclass(frozen=True)
class RunContext:
    """Per-call inputs besides the image."""

    options: dict[str, Any]  # this connection's options for this pipeline (defaults filled in)
    conf: float  # client's global confidence threshold (meaningful when uses_global_conf)
    imgsz: int  # client's model input size


@dataclass(frozen=True)
class CommandContext:
    options: dict[str, Any]
    frame: np.ndarray | None  # newest BGR frame of this connection (None if nothing streamed yet)


@dataclass
class CommandResult:
    ok: bool
    message: str
    items: list[dict[str, str]] | None = None  # [{"key": ..., "label": ...}]
    item_action: dict[str, Any] | None = None  # {"id": "delete", "label": "Forget", "confirm": True}
    files: list[dict[str, str]] | None = None  # [{"name": ..., "kind": ..., "url": "/captures/<id>/<name>"}]


class Pipeline(ABC):
    info: ClassVar[PipelineInfo]

    @abstractmethod
    def load(self, device: str) -> None:
        """Load weights (downloading into MODELS_DIR if missing) and warm up. Runs in a worker thread."""

    def unload(self) -> None:
        """Drop references so VRAM can be freed; `load` may be called again later."""

    @abstractmethod
    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        """BGR uint8 HxWx3 → annotations. Called from a worker thread, never concurrently on one instance."""

    def command(self, action: str, args: dict[str, Any], ctx: CommandContext) -> CommandResult:
        return CommandResult(ok=False, message=f"{self.info.id}: unknown action {action!r}")
