"""Temporal smoothing for one pipeline on one connection: ByteTrack plus a display layer, so boxes don't
flicker frame to frame.

Three mechanisms, each aimed at one kind of flicker:
1. ByteTrack two-stage matching. YOLO-family pipelines detect down to LOW_CONF; detections below the
   user's threshold can't start a track but can keep an existing one alive (most "missed" frames are a
   dip from 0.40 to 0.30). A new track must be seen in two consecutive updates, which filters one-frame ghosts.
2. Hold: a track missed entirely stays visible for `hold_ms`, at its Kalman-predicted position.
3. Label and confidence are aggregated: per-label score votes that decay each update (no cup/bowl/cup or
   "unknown"/"Alex" flipping, OCR text settles on the most-read string) and an EMA of the confidence.

Pipeline-agnostic: it consumes `Annotation`s (pixel coords) and carries `detail`/`points` of the newest
matching detection through to the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from .pipelines.base import Annotation

LOW_CONF = 0.1  # detector floor for uses_global_conf pipelines; ByteTrack's second-stage threshold
TRACK_BUFFER_FRAMES = 30  # how long ByteTrack remembers a lost track for re-association (same id)
MATCH_THRESH = 0.8
LABEL_DECAY = 0.8  # per-update decay of label votes
CONF_ALPHA = 0.3  # EMA weight of the newest confidence

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackView:
    id: str  # "<pipeline>:<n>"
    pipeline: str
    label: str
    conf: float
    box: Box  # normalized xyxy
    seen_ms_ago: int  # 0 = detected in the newest update; >0 = held/predicted
    detail: str | None = None
    points: list[tuple[float, float, float]] | None = None  # normalized x, y, visibility

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "pipeline": self.pipeline,
            "label": self.label,
            "conf": round(self.conf, 3),
            "box": [round(v, 4) for v in self.box],
            "seen_ms_ago": self.seen_ms_ago,
        }
        if self.detail is not None:
            out["detail"] = self.detail
        if self.points is not None:
            out["points"] = [[round(x, 4), round(y, 4), round(v, 2)] for x, y, v in self.points]
        return out


@dataclass
class _TrackState:
    conf: float
    box: Box
    last_seen_ms: float
    detail: str | None
    points: list[tuple[float, float, float]] | None
    label_votes: dict[str, float] = field(default_factory=dict)
    predicted: Box | None = None  # Kalman prediction while lost

    def vote(self, label: str, score: float) -> None:
        for key in self.label_votes:
            self.label_votes[key] *= LABEL_DECAY
        self.label_votes[label] = self.label_votes.get(label, 0.0) + score
        # Drop labels that have decayed to nothing (OCR produces many one-off strings).
        self.label_votes = {k: v for k, v in self.label_votes.items() if v > 1e-3}

    @property
    def label(self) -> str:
        return max(self.label_votes, key=self.label_votes.__getitem__)


class TrackSmoother:
    def __init__(self, pipeline_id: str) -> None:
        self._pipeline_id = pipeline_id
        self._args = SimpleNamespace(
            track_high_thresh=0.35,
            track_low_thresh=LOW_CONF,
            new_track_thresh=0.35,
            track_buffer=TRACK_BUFFER_FRAMES,
            match_thresh=MATCH_THRESH,
            fuse_score=True,
        )
        self._tracker = BYTETracker(self._args)
        self._states: dict[int, _TrackState] = {}

    def update(
        self, annotations: list[Annotation], width: int, height: int, new_track_conf: float | None, now_ms: float
    ) -> None:
        """Feed one frame's annotations (pixel coords of a `width`×`height` image).

        `new_track_conf`: the client's threshold for uses_global_conf pipelines; None = the pipeline already
        filtered, accept everything (both ByteTrack stages then see every detection as high-confidence).
        """
        high = new_track_conf if new_track_conf is not None else 0.0
        self._args.track_high_thresh = self._args.new_track_thresh = high
        self._args.track_low_thresh = LOW_CONF if new_track_conf is not None else 0.0
        scale = np.array([width, height, width, height], dtype=np.float32)

        rows = [[*a.box, a.conf, 0.0] for a in annotations]
        boxes = Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (height, width))

        seen: set[int] = set()
        for row in self._tracker.update(boxes):
            x1, y1, x2, y2, track_id, score = row[:6]
            track_id, score, ann = int(track_id), float(score), annotations[int(row[7])]
            box = _clip(np.array([x1, y1, x2, y2]) / scale)
            points = _norm_points(ann.points, width, height)
            state = self._states.get(track_id)
            if state is None:
                state = _TrackState(score, box, now_ms, ann.detail, points)
                self._states[track_id] = state
            else:
                state.conf += CONF_ALPHA * (score - state.conf)
                state.box, state.last_seen_ms, state.detail, state.points = box, now_ms, ann.detail, points
                state.predicted = None
            state.vote(ann.label, score)
            seen.add(track_id)

        lost = {t.track_id: t for t in self._tracker.lost_stracks}
        for track_id, state in list(self._states.items()):
            if track_id in seen:
                continue
            if track_id in lost:
                state.predicted = _clip(lost[track_id].xyxy / scale)
            else:  # ByteTrack dropped it for good
                del self._states[track_id]

    def views(self, hold_ms: float, now_ms: float) -> list[TrackView]:
        """Tracks to display now; forgets those not seen for longer than `hold_ms`."""
        out: list[TrackView] = []
        for track_id, state in list(self._states.items()):
            age_ms = now_ms - state.last_seen_ms
            if age_ms > hold_ms and age_ms > 0:
                del self._states[track_id]
                continue
            held = state.predicted is not None
            out.append(
                TrackView(
                    id=f"{self._pipeline_id}:{track_id}",
                    pipeline=self._pipeline_id,
                    label=state.label,
                    conf=state.conf,
                    box=state.predicted if held else state.box,
                    seen_ms_ago=round(age_ms) if held else 0,
                    detail=state.detail,
                    points=state.points,
                )
            )
        return out


def normalize(ann: Annotation, width: int, height: int, pipeline_id: str) -> dict[str, Any]:
    """Raw (untracked) annotation → wire JSON (`dets`)."""
    x1, y1, x2, y2 = _clip(np.array(ann.box, dtype=np.float32) / np.array([width, height, width, height]))
    out: dict[str, Any] = {
        "pipeline": pipeline_id,
        "label": ann.label,
        "conf": round(ann.conf, 3),
        "box": [round(v, 4) for v in (x1, y1, x2, y2)],
    }
    if ann.detail is not None:
        out["detail"] = ann.detail
    points = _norm_points(ann.points, width, height)
    if points is not None:
        out["points"] = [[round(x, 4), round(y, 4), round(v, 2)] for x, y, v in points]
    return out


def _clip(box: np.ndarray) -> Box:
    x1, y1, x2, y2 = np.clip(box, 0.0, 1.0).tolist()
    return (x1, y1, x2, y2)


def _norm_points(
    points: list[tuple[float, float, float]] | None, width: int, height: int
) -> list[tuple[float, float, float]] | None:
    if points is None:
        return None
    return [(min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0), v) for x, y, v in points]
