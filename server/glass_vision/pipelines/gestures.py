"""Hand landmarks + 7 canned gestures via MediaPipe Tasks `GestureRecognizer` (float16).

GPU delegate (EGL/OpenGL ES on the 3090, ~6 ms/frame) with CPU fallback (XNNPACK, ~21 ms). On GPU, MediaPipe prints
`tensor.cc] Tensors are designed for single writes ... undefined behavior` once while building its graph. It is
internal to MediaPipe and harmless here: printed once (not per frame), and CPU vs GPU results match (same labels,
landmarks within 1.3% of the frame, scores within 0.01) on the test images, also when called from many threads.

Running mode IMAGE: one instance serves every connection, so VIDEO mode's per-stream monotonic timestamps
don't fit. `num_hands` and the confidences are creation-time options, so a change recreates the recognizer
(~tens of ms); connections with different values would thrash it, which is acceptable for this use.
"""

from __future__ import annotations

import logging
import urllib.request
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, Point, RunContext

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)
MODEL_PATH = MODELS_DIR / "mediapipe" / "gesture_recognizer.task"

# MediaPipe HAND_CONNECTIONS (palm, then thumb → pinky), hard-coded so mediapipe.solutions isn't needed.
HAND_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]  # fmt: skip

BOX_PAD = 0.10  # fraction of the landmark bounds added on each side


_UNMIRROR = {"Left": "Right", "Right": "Left"}


def _download_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    log.info("downloading %s", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(MODEL_PATH)


class GesturesPipeline(Pipeline):
    info = PipelineInfo(
        id="gestures",
        name="Hand gestures",
        description=(
            "Hands (21 landmarks) and 7 gestures: thumb up, thumb down, open palm, fist, pointing up, "
            "victory, I-love-you. Unrecognized poses show as 'hand'."
        ),
        default_enabled=False,
        uses_global_conf=False,
        interval_ms=0,
        options=[
            OptionSpec("num_hands", "Max hands", "number", 2, min=1, max=4, step=1),
            OptionSpec("min_conf", "Min hand confidence", "number", 0.5, min=0.1, max=0.9, step=0.05),
        ],
        skeleton=HAND_SKELETON,
    )

    def __init__(self) -> None:
        self._recognizer: vision.GestureRecognizer | None = None
        self._created_with: tuple[int, float] | None = None
        self._delegate = BaseOptions.Delegate.CPU

    def load(self, device: str) -> None:
        _download_model()
        defaults = self.info.default_options()
        num_hands, min_conf = int(defaults["num_hands"]), float(defaults["min_conf"])
        warm = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8))
        if device.startswith("cuda"):
            self._delegate = BaseOptions.Delegate.GPU
            try:
                self._ensure(num_hands, min_conf).recognize(warm)
                log.info("gestures on GPU delegate")
                return
            except Exception:  # no EGL/GL on this host: fall back to CPU
                log.warning("gestures: GPU delegate unavailable, using CPU", exc_info=True)
                self.unload()
        self._delegate = BaseOptions.Delegate.CPU
        self._ensure(num_hands, min_conf).recognize(warm)

    def unload(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
        self._recognizer = None
        self._created_with = None

    def _ensure(self, num_hands: int, min_conf: float) -> vision.GestureRecognizer:
        key = (num_hands, min_conf)
        if self._recognizer is None or self._created_with != key:
            if self._recognizer is not None:
                self._recognizer.close()
            options = vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=self._delegate),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
            self._recognizer = vision.GestureRecognizer.create_from_options(options)
            self._created_with = key
        return self._recognizer

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        opts: dict[str, Any] = ctx.options
        recognizer = self._ensure(int(opts["num_hands"]), round(float(opts["min_conf"]), 3))
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        out: list[Annotation] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            points: list[Point] = [(lm.x * w, lm.y * h, 1.0) for lm in landmarks]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            pad_x = (max(xs) - min(xs)) * BOX_PAD
            pad_y = (max(ys) - min(ys)) * BOX_PAD
            box = (
                max(0.0, min(xs) - pad_x),
                max(0.0, min(ys) - pad_y),
                min(float(w), max(xs) + pad_x),
                min(float(h), max(ys) + pad_y),
            )
            hand = result.handedness[i][0] if i < len(result.handedness) and result.handedness[i] else None
            gesture = result.gestures[i][0] if i < len(result.gestures) and result.gestures[i] else None
            if gesture is not None and gesture.category_name and gesture.category_name != "None":
                label, conf = gesture.category_name, float(gesture.score)
            else:
                label, conf = "hand", float(hand.score) if hand is not None else 1.0
            # MediaPipe labels handedness assuming a mirrored selfie image; our frames come un-mirrored from a
            # rear camera (and later the glasses' outward camera), so its "Left" is the user's right hand.
            detail = _UNMIRROR.get(hand.category_name, hand.category_name) if hand is not None else None
            out.append(Annotation(box=box, conf=conf, label=label, detail=detail, points=points))
        return out
