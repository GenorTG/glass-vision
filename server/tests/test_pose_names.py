"""Pose names from synthetic skeletons (built from segment angles) and from YOLO pose on a real photo."""

from __future__ import annotations

import itertools
import math
import random

import cv2
import pytest
from ultralytics.utils import ASSETS

from glass_vision.pipelines.base import RunContext
from glass_vision.pose_names import pose_name

T = 100.0  # torso length, px
Pt = tuple[float, float, float]


def direction(deg: float) -> tuple[float, float]:
    """Unit vector in image coordinates: 0 = down, 90 = image right, 180 = up, -90 = image left."""
    r = math.radians(deg)
    return math.sin(r), math.cos(r)


def skeleton(
    *,
    torso: float = 180,
    thigh: tuple[float, float] = (0, 0),
    shin: tuple[float, float] = (0, 0),
    upper_arm: tuple[float, float] = (0, 0),
    forearm: tuple[float, float] = (0, 0),
) -> list[Pt]:
    """COCO-17 of a person facing the camera (their left = image right). Pairs are (left, right) segment directions,
    absolute in the image (see `direction`); the torso direction points from the hips to the shoulders."""
    hip_mid = (500.0, 600.0)
    tx, ty = direction(torso)
    sh_mid = (hip_mid[0] + tx * T, hip_mid[1] + ty * T)
    px, py = -ty, tx  # perpendicular towards the person's left

    def off(base: tuple[float, float], along: float, side: float) -> tuple[float, float]:
        return base[0] + tx * along * T + px * side * T, base[1] + ty * along * T + py * side * T

    def seg(start: tuple[float, float], deg: float, length: float) -> tuple[float, float]:
        dx, dy = direction(deg)
        return start[0] + dx * length * T, start[1] + dy * length * T

    pts: list[tuple[float, float]] = [(0, 0)] * 17
    pts[0] = off(sh_mid, 0.45, 0)
    pts[1], pts[2] = off(sh_mid, 0.5, 0.08), off(sh_mid, 0.5, -0.08)
    pts[3], pts[4] = off(sh_mid, 0.45, 0.15), off(sh_mid, 0.45, -0.15)
    pts[5], pts[6] = off(sh_mid, 0, 0.4), off(sh_mid, 0, -0.4)
    pts[11], pts[12] = off(hip_mid, 0, 0.25), off(hip_mid, 0, -0.25)
    for i, (sh, hip) in enumerate(((5, 11), (6, 12))):
        pts[7 + i] = seg(pts[sh], upper_arm[i], 0.6)
        pts[9 + i] = seg(pts[7 + i], forearm[i], 0.55)
        pts[13 + i] = seg(pts[hip], thigh[i], 0.9)
        pts[15 + i] = seg(pts[13 + i], shin[i], 0.85)
    return [(x, y, 1.0) for x, y in pts]


def hide(points: list[Pt], *indices: int) -> list[Pt]:
    return [(x, y, 0.0) if i in indices else (x, y, v) for i, (x, y, v) in enumerate(points)]


UPPER_ONLY = tuple(range(11, 17))

CASES: dict[str, dict[str, object]] = {
    "Standing": {},
    "Standing · Arms up": {"upper_arm": (180, 180), "forearm": (180, 180)},
    "Standing · T-pose": {"upper_arm": (90, -90), "forearm": (90, -90)},
    "Standing · Hands on hips": {"upper_arm": (40, -40), "forearm": (-50, 50)},
    "Standing · Arms crossed": {"upper_arm": (-20, 20), "forearm": (-90, 90)},
    "Standing · Hand on head": {"upper_arm": (150, 0), "forearm": (-120, 0)},
    "Standing · Hands on head": {"upper_arm": (150, -150), "forearm": (-120, 120)},
    "Standing · Pointing left": {"upper_arm": (0, -90), "forearm": (0, -90)},  # right arm, towards image left
    "Standing · Pointing right": {"upper_arm": (90, 0), "forearm": (90, 0)},
    "Standing · One arm up": {"upper_arm": (180, 0), "forearm": (170, 0)},
    "Sitting": {"thigh": (90, 90), "shin": (0, 0)},
    "Sitting · Arms up": {"thigh": (90, 90), "upper_arm": (180, 180), "forearm": (180, 180)},
    "Squatting": {"torso": 160, "thigh": (110, 110), "shin": (-20, -20)},
    "Kneeling": {"thigh": (0, 0), "shin": (-90, -90)},
    "Kneeling · One arm up": {"thigh": (0, 10), "shin": (-90, 10), "upper_arm": (0, 180), "forearm": (0, 180)},
    "Lying down": {
        "torso": 90,
        "thigh": (-90, -90),
        "shin": (-90, -90),
        "upper_arm": (-90, -90),
        "forearm": (-90, -90),
    },
    "Lying down · Arms up": {
        "torso": 90,
        "thigh": (-90, -90),
        "shin": (-90, -90),
        "upper_arm": (90, 90),
        "forearm": (90, 90),
    },
    "Bending over": {"torso": 60, "thigh": (0, 0), "shin": (0, 0)},
}


@pytest.mark.parametrize("expected", list(CASES))
def test_synthetic_poses(expected: str) -> None:
    assert pose_name(skeleton(**CASES[expected])) == expected  # type: ignore[arg-type]


def test_upper_body_only() -> None:
    assert pose_name(hide(skeleton(), *UPPER_ONLY)) == "Upper body"
    assert pose_name(hide(skeleton(upper_arm=(180, 180), forearm=(180, 180)), *UPPER_ONLY)) == "Arms up"
    assert pose_name(hide(skeleton(upper_arm=(90, -90), forearm=(90, -90)), *UPPER_ONLY)) == "T-pose"


def test_nothing_visible() -> None:
    assert pose_name([(0.0, 0.0, 0.0)] * 17) == "person"
    assert pose_name(hide(skeleton(), *range(1, 17))) == "person"  # nose alone: no scale
    assert pose_name([]) == "person"


def test_missing_knees_keep_arm_label() -> None:
    assert pose_name(hide(skeleton(upper_arm=(180, 180), forearm=(180, 180)), 13, 14, 15, 16)) == "Arms up"


def test_one_leg_is_enough() -> None:
    assert pose_name(hide(skeleton(thigh=(90, 90)), 14, 16)) == "Sitting"
    assert pose_name(hide(skeleton(), 13, 15)) == "Standing"


def test_robust_to_any_missing_points() -> None:
    rng = random.Random(7)
    for case, seen in itertools.product(CASES.values(), range(30)):
        pts = skeleton(**case)  # type: ignore[arg-type]
        pts = [(x, y, rng.choice((0.0, 0.2, 0.9))) for x, y, _ in pts]
        assert isinstance(pose_name(pts), str)
    assert isinstance(pose_name([(1.0, 1.0, 1.0)] * 17), str)  # all points coincide
    assert isinstance(pose_name([(math.nan, 1.0, 1.0)] * 17), str)


@pytest.mark.gpu
def test_bus_people_are_standing() -> None:
    from glass_vision.pipelines.pose import PosePipeline

    p = PosePipeline()
    p.load("cuda:0")
    anns = [a for a in p.run(cv2.imread(str(ASSETS / "bus.jpg")), RunContext({"names": True}, 0.35, 640))]
    people = [a for a in anns if a.conf >= 0.5]
    assert all(a.detail == "person" for a in people)
    # The man cut off at the left edge has every keypoint < 0.3: he stays "person".
    seen = [a for a in people if a.points is not None and sum(v >= 0.3 for _, _, v in a.points) >= 8]
    assert len(seen) >= 3
    assert all(a.label.startswith("Standing") for a in seen), [a.label for a in seen]
    off = p.run(cv2.imread(str(ASSETS / "bus.jpg")), RunContext({"names": False}, 0.35, 640))
    assert all(a.label == "person" for a in off)
