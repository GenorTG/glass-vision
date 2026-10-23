"""Human body pose (17 COCO keypoints) with YOLO11s-pose, named by `pose_names` ("Standing · Arms up").

Action `capture`: on the newest full frame, bodies (YOLO) + hands (MediaPipe HandLandmarker, created on first use,
CPU, IMAGE mode) → OpenPose renders and JSON for ComfyUI ControlNet in `DATA_DIR/captures/<id>/` (see
`glass_vision.openpose`, `glass_vision.captures`). Each hand is attached to the nearest visible COCO wrist, which
also decides whether it is the person's left or right hand.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from ultralytics import YOLO

from .. import captures
from ..openpose import Hand21, OpenPosePerson, coco_to_openpose, draw_poses, encode_pose_json, to_bgr
from ..pose_names import pose_name
from ._ultralytics import WARMUP_IMAGE, predict_kwargs, to_annotations
from .base import (
    MODELS_DIR,
    ActionSpec,
    Annotation,
    CommandContext,
    CommandResult,
    OptionSpec,
    Pipeline,
    PipelineInfo,
    Point,
    RunContext,
)

log = logging.getLogger(__name__)

# COCO keypoints: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]  # fmt: skip
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "mediapipe" / "hand_landmarker.task"
CAPTURE_IMGSZ = 960  # one-off on the full frame: bigger than the streaming 640 for small/far people
CAPTURE_CONF = 0.4  # min person confidence for a capture
KEYPOINT_CONF = 0.3  # = openpose.MIN_CONF: a wrist below this can't take a hand
MAX_HANDS = 4

PixelHand = list[tuple[float, float]]  # 21 MediaPipe landmarks in pixels, 0 = wrist


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("downloading %s", url)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def assign_hands(
    people: list[list[Point]], hands: list[PixelHand], boxes: list[tuple[float, float, float, float]]
) -> tuple[dict[tuple[int, str], int], list[int]]:
    """Greedy nearest-wrist matching → ({(person index, "left"|"right"): hand index}, unattached hand indices).

    `people` are COCO keypoints in pixels; sides are the person's own (COCO 9 = left wrist). A hand only attaches to
    a wrist within max(forearm length, 10% of that person's box height) of the hand's own wrist landmark."""
    pairs: list[tuple[float, int, int, str]] = []
    for pi, kp in enumerate(people):
        box_h = boxes[pi][3] - boxes[pi][1]
        for side, wrist_i, elbow_i in (("left", L_WRIST, L_ELBOW), ("right", R_WRIST, R_ELBOW)):
            wx, wy, wv = kp[wrist_i]
            if wv < KEYPOINT_CONF:
                continue
            ex, ey, ev = kp[elbow_i]
            limit = max(math.hypot(wx - ex, wy - ey) if ev >= KEYPOINT_CONF else 0.0, 0.1 * box_h)
            for hi, hand in enumerate(hands):
                d = math.hypot(hand[0][0] - wx, hand[0][1] - wy)
                if d <= limit:
                    pairs.append((d, hi, pi, side))
    assigned: dict[tuple[int, str], int] = {}
    used: set[int] = set()
    for _, hi, pi, side in sorted(pairs):
        if hi in used or (pi, side) in assigned:
            continue
        assigned[(pi, side)] = hi
        used.add(hi)
    return assigned, [hi for hi in range(len(hands)) if hi not in used]


class PosePipeline(Pipeline):
    info = PipelineInfo(
        id="pose",
        name="Body pose",
        description=(
            "Skeleton of every person, named by posture and arms (e.g. 'Standing · Arms up'; YOLO11s-pose). "
            "Capture saves OpenPose images + JSON (body and hands) for ComfyUI ControlNet."
        ),
        uses_global_conf=True,
        options=[OptionSpec("names", "Pose names", "bool", True)],
        actions=[ActionSpec("capture", "Capture pose")],
        skeleton=COCO_SKELETON,
    )

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._device = "cpu"
        self._hands: vision.HandLandmarker | None = None

    def load(self, device: str) -> None:
        self._device = device
        self._model = YOLO(str(MODELS_DIR / "yolo11s-pose.pt"))  # downloaded by ultralytics if missing
        self._model.predict(WARMUP_IMAGE, device=device, verbose=False)

    def unload(self) -> None:
        self._model = None
        if self._hands is not None:
            self._hands.close()
        self._hands = None

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        assert self._model is not None, "not loaded"
        anns = to_annotations(self._model.predict(image, **predict_kwargs(ctx, self._device))[0])
        if ctx.options.get("names", True):
            for a in anns:
                if a.points is not None:
                    a.label, a.detail = pose_name(a.points), "person"
        return anns

    # --- capture ---------------------------------------------------------------------------------

    def command(self, action: str, args: dict[str, Any], ctx: CommandContext) -> CommandResult:
        if action != "capture":
            return super().command(action, args, ctx)
        if ctx.frame is None:
            return CommandResult(ok=False, message="No frame yet: start the camera first")
        return self.capture(ctx.frame)

    def _hand_landmarker(self) -> vision.HandLandmarker:
        if self._hands is None:
            _download(HAND_MODEL_URL, HAND_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=MAX_HANDS,
            )
            self._hands = vision.HandLandmarker.create_from_options(options)
        return self._hands

    def detect_hands(self, image: np.ndarray) -> list[PixelHand]:
        """21 landmarks (pixels) per hand in a BGR image."""
        h, w = image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = self._hand_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return [[(lm.x * w, lm.y * h) for lm in hand] for hand in result.hand_landmarks]

    def capture(self, frame: np.ndarray) -> CommandResult:
        assert self._model is not None, "not loaded"
        h, w = frame.shape[:2]
        result = self._model.predict(frame, conf=CAPTURE_CONF, imgsz=CAPTURE_IMGSZ, device=self._device, verbose=False)
        bodies = [a for a in to_annotations(result[0]) if a.points is not None and a.conf >= CAPTURE_CONF]
        if not bodies:
            return CommandResult(ok=False, message="No person in view")
        body_points = [a.points for a in bodies if a.points is not None]
        hands = self.detect_hands(frame)
        assigned, loose = assign_hands(body_points, hands, [a.box for a in bodies])

        def norm(hand: PixelHand) -> Hand21:
            return [(x / w, y / h) for x, y in hand]

        def hand_of(person: int, side: str) -> Hand21 | None:
            hi = assigned.get((person, side))
            return None if hi is None else norm(hands[hi])

        people = [
            OpenPosePerson(coco_to_openpose(points, w, h), hand_of(i, "left"), hand_of(i, "right"))
            for i, points in enumerate(body_points)
        ]
        loose_hands = [norm(hands[i]) for i in loose]  # drawn, but in no person's JSON

        capture_id, path = captures.new_capture()
        cv2.imwrite(str(path / "photo.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(path / "openpose_body.png"), to_bgr(draw_poses(people, h, w, draw_hands=False)))
        cv2.imwrite(str(path / "openpose_full.png"), to_bgr(draw_poses(people, h, w, extra_hands=loose_hands)))
        (path / "pose_keypoints.json").write_text(json.dumps(encode_pose_json(people, h, w)))
        photo_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay = draw_poses(people, h, w, extra_hands=loose_hands, background=photo_rgb)
        cv2.imwrite(str(path / "pose_on_photo.jpg"), to_bgr(overlay), [cv2.IMWRITE_JPEG_QUALITY, 92])

        files = [captures.file_entry(capture_id, name) for name in captures.CAPTURE_FILES]
        n = len(people)
        log.info("pose capture %s: %d people, %d hands (%d unattached)", capture_id, n, len(hands), len(loose))
        return CommandResult(
            ok=True, message=f"Captured {n} {'person' if n == 1 else 'people'} → {len(files)} files", files=files
        )
