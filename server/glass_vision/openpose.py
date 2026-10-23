"""OpenPose-style renders + JSON for ComfyUI ControlNet, from COCO-17 body and MediaPipe-21 hand keypoints.

Ported from comfyui_controlnet_aux (Fannovel16), commit a587a4c1594524c1d2de5cd1a86cd868ff448e1c:
- `draw_bodypose`, `draw_handpose`, `eps`: src/custom_controlnet_aux/open_pose/util.py
  https://github.com/Fannovel16/comfyui_controlnet_aux/blob/a587a4c1594524c1d2de5cd1a86cd868ff448e1c/src/custom_controlnet_aux/open_pose/util.py
- `draw_poses` (black canvas, body then left/right hand per person) and `encode_poses_as_dict`:
  src/custom_controlnet_aux/open_pose/__init__.py
- COCO-17 → OpenPose-18 (neck = shoulder midpoint, present iff both shoulders > 0.3; keypoints < 0.3 dropped):
  src/custom_controlnet_aux/dwpose/wholebody.py (`mmpose_idx` / `openpose_idx`) and dwpose/__init__.py.
Those are lllyasviel's ControlNet `annotator/openpose/util.py` with the per-limb `addWeighted(0.4, 0.6)` replaced by
filling with colour * 0.6, which is what ComfyUI renders today; constants are copied verbatim, not re-derived.

Like the reference, keypoints are *normalized* (0..1) and the canvas is RGB; `to_bgr` before `cv2.imwrite`.
OpenPose-18 order: 0 nose, 1 neck, 2 R shoulder, 3 R elbow, 4 R wrist, 5 L shoulder, 6 L elbow, 7 L wrist, 8 R hip,
9 R knee, 10 R ankle, 11 L hip, 12 L knee, 13 L ankle, 14 R eye, 15 L eye, 16 R ear, 17 L ear (person's own sides).
MediaPipe's 21 hand landmarks use the same order as OpenPose's hand model (wrist, thumb 1-4, index 5-8 … pinky 17-20).
"""

from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

NormPoint = tuple[float, float]  # x, y in 0..1 of the canvas
Body18 = list[NormPoint | None]  # OpenPose-18, None = not detected
Hand21 = list[NormPoint]

MIN_CONF = 0.3  # DWPose: keypoints below this are "not detected"
EPS = 0.01  # util.py `eps`: hand points at the canvas origin count as missing

# OpenPose-18 index → COCO-17 index (None = neck, computed). Same as DWPose's mmpose_idx/openpose_idx remap.
OPENPOSE_FROM_COCO: list[int | None] = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# util.py draw_bodypose, verbatim (1-based OpenPose-18 indices; RGB colours, one per limb and per joint).
LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]  # fmt: skip
COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
]  # fmt: skip
STICKWIDTH = 4
LIMB_ALPHA = 0.6
JOINT_RADIUS = 4

# util.py draw_handpose, verbatim.
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
    [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]  # fmt: skip
HAND_POINT_COLOR = (0, 0, 255)
HAND_LINE_THICKNESS = 2
HAND_POINT_RADIUS = 4


def hand_edge_color(index: int) -> tuple[float, float, float]:
    """`matplotlib.colors.hsv_to_rgb([ie / len(edges), 1, 1]) * 255` (colorsys is the same formula)."""
    r, g, b = colorsys.hsv_to_rgb(index / float(len(HAND_EDGES)), 1.0, 1.0)
    return (r * 255, g * 255, b * 255)


@dataclass
class OpenPosePerson:
    body: Body18
    left_hand: Hand21 | None = None  # the person's own left hand (OpenPose `hand_left`)
    right_hand: Hand21 | None = None


def coco_to_openpose(points: Sequence[tuple[float, float, float]], width: int, height: int) -> Body18:
    """17 COCO keypoints (pixel x, y, confidence) → OpenPose-18, normalized; < MIN_CONF → None."""

    def norm(i: int) -> NormPoint | None:
        x, y, c = points[i]
        return (x / width, y / height) if c >= MIN_CONF else None

    out: Body18 = []
    for coco in OPENPOSE_FROM_COCO:
        if coco is not None:
            out.append(norm(coco))
            continue
        ls, rs = points[5], points[6]
        # DWPose: neck = mean of the shoulders, shown only if both are > 0.3.
        out.append(
            ((ls[0] + rs[0]) / 2 / width, (ls[1] + rs[1]) / 2 / height)
            if ls[2] > MIN_CONF and rs[2] > MIN_CONF
            else None
        )
    return out


def draw_bodypose(canvas: np.ndarray, keypoints: Body18) -> np.ndarray:
    """util.py draw_bodypose (stick_scale 1) on an RGB uint8 canvas, in place."""
    H, W = canvas.shape[:2]
    for (k1_index, k2_index), color in zip(LIMB_SEQ, COLORS, strict=False):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        if keypoint1 is None or keypoint2 is None:
            continue
        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), STICKWIDTH), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * LIMB_ALPHA) for c in color])

    for keypoint, color in zip(keypoints, COLORS, strict=False):
        if keypoint is None:
            continue
        x = int(keypoint[0] * W)
        y = int(keypoint[1] * H)
        cv2.circle(canvas, (x, y), JOINT_RADIUS, color, thickness=-1)
    return canvas


def draw_handpose(canvas: np.ndarray, keypoints: Hand21 | None) -> np.ndarray:
    """util.py draw_handpose on an RGB uint8 canvas, in place."""
    if not keypoints:
        return canvas
    H, W = canvas.shape[:2]
    for ie, (e1, e2) in enumerate(HAND_EDGES):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        x1 = int(k1[0] * W)
        y1 = int(k1[1] * H)
        x2 = int(k2[0] * W)
        y2 = int(k2[1] * H)
        if x1 > EPS and y1 > EPS and x2 > EPS and y2 > EPS:
            cv2.line(canvas, (x1, y1), (x2, y2), hand_edge_color(ie), thickness=HAND_LINE_THICKNESS)
    for keypoint in keypoints:
        x = int(keypoint[0] * W)
        y = int(keypoint[1] * H)
        if x > EPS and y > EPS:
            cv2.circle(canvas, (x, y), HAND_POINT_RADIUS, HAND_POINT_COLOR, thickness=-1)
    return canvas


def draw_poses(
    people: Sequence[OpenPosePerson],
    height: int,
    width: int,
    *,
    draw_hands: bool = True,
    extra_hands: Sequence[Hand21] = (),
    background: np.ndarray | None = None,
) -> np.ndarray:
    """__init__.py draw_poses: RGB canvas (black unless `background`, an RGB image of the same size).

    `extra_hands` are hands not attached to any body (still drawn when `draw_hands`)."""
    canvas = np.zeros((height, width, 3), np.uint8) if background is None else background.copy()
    for person in people:
        draw_bodypose(canvas, person.body)
        if draw_hands:
            draw_handpose(canvas, person.left_hand)
            draw_handpose(canvas, person.right_hand)
    if draw_hands:
        for hand in extra_hands:
            draw_handpose(canvas, hand)
    return canvas


def to_bgr(canvas: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def encode_pose_json(people: Sequence[OpenPosePerson], height: int, width: int) -> dict[str, Any]:
    """__init__.py encode_poses_as_dict: OpenPose JSON as ComfyUI's POSE_KEYPOINT frames use it.

    Like the reference: normalized x/y, confidence 1.0 for detected points and 0, 0, 0 for missing ones
    (comfyui_controlnet_aux's `decode_json_as_poses` treats c < 1.0 as missing and multiplies x/y by the canvas
    size, so real confidences or pixel coordinates would not round-trip). Differences: no face model, so
    `face_keypoints_2d` is `[]`; a hand that wasn't found is `[]` instead of `null` (the decoder maps both to
    "no hand"; 21 zero triplets would make its `draw_handpose` crash on None points)."""

    def body(keypoints: Body18) -> list[float]:
        return [v for k in keypoints for v in ([float(k[0]), float(k[1]), 1.0] if k is not None else [0.0, 0.0, 0.0])]

    def hand(keypoints: Hand21 | None) -> list[float]:
        return [v for k in keypoints for v in (float(k[0]), float(k[1]), 1.0)] if keypoints else []

    return {
        "people": [
            {
                "pose_keypoints_2d": body(p.body),
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": hand(p.left_hand),
                "hand_right_keypoints_2d": hand(p.right_hand),
            }
            for p in people
        ],
        "canvas_height": height,
        "canvas_width": width,
    }
