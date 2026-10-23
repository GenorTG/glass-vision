"""OpenPose renderer + JSON against the reference constants (comfyui_controlnet_aux open_pose/util.py)."""

from __future__ import annotations

import math

import matplotlib.colors
import numpy as np
import pytest

from glass_vision.openpose import (
    COLORS,
    HAND_EDGES,
    LIMB_SEQ,
    OpenPosePerson,
    coco_to_openpose,
    draw_poses,
    encode_pose_json,
    hand_edge_color,
)

W, H = 400, 300

# OpenPose-18 name → COCO-17 name, spelled out independently of the module's index table.
OP18 = ["nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri", "r_hip", "r_knee", "r_ank",
        "l_hip", "l_knee", "l_ank", "r_eye", "l_eye", "r_ear", "l_ear"]  # fmt: skip
COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]  # fmt: skip


def coco_points(conf: float = 1.0) -> list[tuple[float, float, float]]:
    return [(10.0 + 20 * i, 5.0 + 15 * i, conf) for i in range(17)]


def test_reference_constants() -> None:
    assert len(LIMB_SEQ) == 17 and len(COLORS) == 18 and len(HAND_EDGES) == 20
    assert LIMB_SEQ[0] == [2, 3] and LIMB_SEQ[12] == [2, 1] and LIMB_SEQ[16] == [16, 18]
    assert COLORS[0] == [255, 0, 0] and COLORS[12] == [0, 0, 255] and COLORS[17] == [255, 0, 85]
    for i in range(len(HAND_EDGES)):  # same as the reference's matplotlib call
        ref = matplotlib.colors.hsv_to_rgb([i / float(len(HAND_EDGES)), 1.0, 1.0]) * 255
        assert np.allclose(hand_edge_color(i), ref)


def test_coco_to_openpose_mapping_and_neck() -> None:
    pts = coco_points()
    body = coco_to_openpose(pts, W, H)
    assert len(body) == 18
    for op_i, name in enumerate(OP18):
        if name == "neck":
            ls, rs = pts[COCO17.index("l_sho")], pts[COCO17.index("r_sho")]
            assert body[op_i] == pytest.approx(((ls[0] + rs[0]) / 2 / W, (ls[1] + rs[1]) / 2 / H))
        else:
            x, y, _ = pts[COCO17.index(name)]
            assert body[op_i] == pytest.approx((x / W, y / H)), name


def test_low_confidence_points_and_neck_are_dropped() -> None:
    pts = coco_points()
    pts[COCO17.index("r_sho")] = (100.0, 100.0, 0.29)
    pts[COCO17.index("l_knee")] = (100.0, 100.0, 0.1)
    body = coco_to_openpose(pts, W, H)
    assert body[OP18.index("r_sho")] is None and body[OP18.index("l_knee")] is None
    assert body[OP18.index("neck")] is None  # needs both shoulders
    assert body[OP18.index("l_sho")] is not None
    assert all(p is None for p in coco_to_openpose(coco_points(0.0), W, H))


def rgb_at(canvas: np.ndarray, x: float, y: float) -> list[int]:
    return [int(v) for v in canvas[int(y * H), int(x * W)]]


@pytest.mark.parametrize("limb", range(len(LIMB_SEQ)))
def test_each_limb_and_joint_colour(limb: int) -> None:
    a, b = (i - 1 for i in LIMB_SEQ[limb])
    body: list[tuple[float, float] | None] = [None] * 18
    body[a], body[b] = (0.2, 0.3), (0.8, 0.7)
    canvas = draw_poses([OpenPosePerson(body)], H, W)
    assert canvas.shape == (H, W, 3) and canvas.dtype == np.uint8
    assert rgb_at(canvas, 0.5, 0.5) == [int(c * 0.6) for c in COLORS[limb]]  # limb: colour * 0.6
    assert rgb_at(canvas, 0.2, 0.3) == COLORS[a]  # joints: full colour of their own index
    assert rgb_at(canvas, 0.8, 0.7) == COLORS[b]
    # Black everywhere off the limb: stick half-width 4 px, joint radius 4 px.
    assert canvas[0, 0].tolist() == [0, 0, 0] and canvas[H - 1, 0].tolist() == [0, 0, 0]
    painted = np.argwhere(canvas.any(axis=2))
    (x0, y0), (x1, y1) = (0.2 * W, 0.3 * H), (0.8 * W, 0.7 * H)
    dist = np.abs((y1 - y0) * painted[:, 1] - (x1 - x0) * painted[:, 0] + x1 * y0 - y1 * x0) / math.hypot(
        x1 - x0, y1 - y0
    )
    assert dist.max() <= 6


def test_limb_is_an_ellipse_of_stickwidth_4() -> None:
    body: list[tuple[float, float] | None] = [None] * 18
    body[1], body[2] = (0.25, 0.5), (0.75, 0.5)  # neck → right shoulder, horizontal
    canvas = draw_poses([OpenPosePerson(body)], H, W)
    column = canvas[:, W // 2].any(axis=1)
    assert 8 <= int(column.sum()) <= 10  # 2 * stickwidth (+1 centre row)


def fan_hand(cx: float = 0.5, cy: float = 0.85) -> list[tuple[float, float]]:
    pts = [(cx, cy)]
    for finger in range(5):
        ang = math.radians(-60 + 30 * finger)
        for j in range(1, 5):
            r = 0.12 + 0.08 * j
            pts.append((cx + r * math.sin(ang), cy - r * math.cos(ang)))
    return pts


def test_hand_edges_and_points() -> None:
    size = 1000
    hand = fan_hand()
    body: list[tuple[float, float] | None] = [None] * 18
    canvas = draw_poses([OpenPosePerson(body, right_hand=hand)], size, size)
    for ie, (e1, e2) in enumerate(HAND_EDGES):
        mx = (int(hand[e1][0] * size) + int(hand[e2][0] * size)) // 2
        my = (int(hand[e1][1] * size) + int(hand[e2][1] * size)) // 2
        patch = canvas[my - 1 : my + 2, mx - 1 : mx + 2].reshape(-1, 3).astype(int)
        want = np.array(hand_edge_color(ie))
        assert np.abs(patch - want).max(axis=1).min() <= 1, ie
    for x, y in hand:
        assert canvas[int(y * size), int(x * size)].tolist() == [0, 0, 255]  # hand points: (0, 0, 255) RGB
    no_hands = draw_poses([OpenPosePerson(body, right_hand=hand)], size, size, draw_hands=False)
    assert not no_hands.any()
    loose = draw_poses([], size, size, extra_hands=[hand])
    assert np.array_equal(loose, canvas)


def test_background_is_not_modified() -> None:
    photo = np.full((H, W, 3), 50, np.uint8)
    body: list[tuple[float, float] | None] = [None] * 18
    body[1], body[2] = (0.25, 0.5), (0.75, 0.5)
    out = draw_poses([OpenPosePerson(body)], H, W, background=photo)
    assert (photo == 50).all() and rgb_at(out, 0.5, 0.5) == [153, 0, 0] and rgb_at(out, 0.1, 0.1) == [50, 50, 50]


def test_json_format() -> None:
    pts = coco_points()
    pts[COCO17.index("l_ank")] = (0.0, 0.0, 0.0)
    person = OpenPosePerson(coco_to_openpose(pts, W, H), left_hand=fan_hand(), right_hand=None)
    doc = encode_pose_json([person, OpenPosePerson([None] * 18)], H, W)
    assert set(doc) == {"people", "canvas_width", "canvas_height"}
    assert doc["canvas_width"] == W and doc["canvas_height"] == H
    p = doc["people"][0]
    assert set(p) == {"pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"}
    assert len(p["pose_keypoints_2d"]) == 18 * 3 and len(p["hand_left_keypoints_2d"]) == 21 * 3
    assert p["hand_right_keypoints_2d"] == [] and p["face_keypoints_2d"] == []
    triplets = [p["pose_keypoints_2d"][i : i + 3] for i in range(0, 54, 3)]
    assert triplets[OP18.index("l_ank")] == [0.0, 0.0, 0.0]
    assert all(c == 1.0 and 0 <= x <= 1 and 0 <= y <= 1 for x, y, c in triplets if c)  # normalized, c = 1
    assert triplets[OP18.index("nose")] == pytest.approx([pts[0][0] / W, pts[0][1] / H, 1.0])
    assert doc["people"][1]["pose_keypoints_2d"] == [0.0] * 54
