"""Test vectors for the app's on-device ports, computed by the server's Python originals.

Run from server/ (read-only use of the server code; writes only into the app's test resources):

    cd server && ./run ../android/app/src/test/gen/gen_test_vectors.py

Writes android/app/src/test/resources/vectors/:
- pose_names.json: COCO-17 keypoints → `pose_names.pose_name` label (the server tests' synthetic cases, the same
  cases with points hidden, and random skeletons through every branch). PoseNamesVectorTest must match all exactly.
- openpose.json: `openpose.coco_to_openpose` inputs → outputs, `pose.assign_hands` inputs → outputs, the people of
  one rendered scene, colour samples of that render at every limb midpoint and joint, and the exact
  `json.dumps(encode_pose_json(...))` text the server writes as pose_keypoints.json.
- openpose_full.rgb.gz / openpose_body.rgb.gz: that scene rendered by `openpose.draw_poses` (what the server writes
  as openpose_full.png / openpose_body.png) as raw RGB rows, gzipped (the app's unit tests can't decode PNG).
Coordinates are rounded before they are fed to Python, so Kotlin gets bit-identical inputs.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[4] / "server"
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tests"))

from glass_vision.openpose import (  # noqa: E402
    COLORS,
    LIMB_SEQ,
    OpenPosePerson,
    coco_to_openpose,
    draw_poses,
    encode_pose_json,
    to_bgr,
)
from glass_vision.pipelines.pose import assign_hands  # noqa: E402
from glass_vision.pose_names import pose_name  # noqa: E402
from test_pose_names import CASES, UPPER_ONLY, hide, skeleton  # noqa: E402

OUT = HERE.parent / "resources" / "vectors"
Pt = tuple[float, float, float]


def rounded(points: list[Pt], digits: int = 3) -> list[Pt]:
    return [(round(x, digits), round(y, digits), round(v, 2)) for x, y, v in points]


def pose_name_vectors() -> list[dict[str, object]]:
    rng = random.Random(20261023)
    inputs: list[list[Pt]] = []
    for case in CASES.values():
        base = skeleton(**case)  # type: ignore[arg-type]
        inputs.append(base)
        inputs.append(hide(base, *UPPER_ONLY))
        for _ in range(6):
            inputs.append([(x, y, rng.choice((0.0, 0.2, 0.9, 1.0))) for x, y, _ in base])
    for _ in range(600):
        kwargs = {
            "torso": rng.uniform(0, 360),
            "thigh": (rng.uniform(-180, 180), rng.uniform(-180, 180)),
            "shin": (rng.uniform(-180, 180), rng.uniform(-180, 180)),
            "upper_arm": (rng.uniform(-180, 180), rng.uniform(-180, 180)),
            "forearm": (rng.uniform(-180, 180), rng.uniform(-180, 180)),
        }
        pts = skeleton(**kwargs)  # type: ignore[arg-type]
        if rng.random() < 0.4:
            pts = hide(pts, *rng.sample(range(17), rng.randint(1, 8)))
        inputs.append(pts)
    inputs.append([(0.0, 0.0, 0.0)] * 17)
    inputs.append([(1.0, 1.0, 1.0)] * 17)
    out = []
    for pts in inputs:
        pts = rounded(pts)
        out.append({"points": [list(p) for p in pts], "label": pose_name(pts)})
    return out


def coco_vectors() -> list[dict[str, object]]:
    rng = random.Random(5)
    out = []
    for _ in range(40):
        w, h = rng.choice(((640, 480), (720, 960), (400, 300)))
        pts = rounded([(rng.uniform(0, w), rng.uniform(0, h), rng.choice((0.0, 0.29, 0.3, 0.31, 0.8))) for _ in range(17)])
        body = coco_to_openpose(pts, w, h)
        out.append({"points": [list(p) for p in pts], "w": w, "h": h, "body": [None if b is None else list(b) for b in body]})
    return out


def fan_hand(cx: float, cy: float, scale: float, rng: random.Random) -> list[tuple[float, float]]:
    pts = [(cx, cy)]
    for finger in range(5):
        ang = math.radians(-60 + 30 * finger + rng.uniform(-8, 8))
        for j in range(1, 5):
            r = scale * (0.12 + 0.08 * j)
            pts.append((cx + r * math.sin(ang), cy - r * math.cos(ang)))
    return pts


def assign_vectors() -> list[dict[str, object]]:
    rng = random.Random(11)
    out = []
    for _ in range(30):
        people, boxes = [], []
        for _p in range(rng.randint(1, 3)):
            pts = rounded([(rng.uniform(0, 800), rng.uniform(0, 600), rng.choice((0.1, 0.5, 0.9))) for _ in range(17)])
            people.append(pts)
            x1, y1 = rng.uniform(0, 400), rng.uniform(0, 300)
            boxes.append((round(x1, 2), round(y1, 2), round(x1 + rng.uniform(50, 400), 2), round(y1 + rng.uniform(80, 300), 2)))
        hands = []
        for _h in range(rng.randint(0, 4)):
            # Mostly near some wrist so matches happen.
            p = rng.choice(people)
            wx, wy, _ = p[rng.choice((9, 10))]
            hands.append([(round(wx + rng.uniform(-40, 40), 2), round(wy + rng.uniform(-40, 40), 2))] + [(0.0, 0.0)] * 20)
        assigned, loose = assign_hands(people, hands, boxes)
        out.append({
            "people": [[list(p) for p in pts] for pts in people],
            "boxes": [list(b) for b in boxes],
            "hand_wrists": [list(h[0]) for h in hands],
            "assigned": sorted([pi, side, hi] for (pi, side), hi in assigned.items()),
            "loose": loose,
        })
    return out


def scene() -> tuple[int, int, list[OpenPosePerson], list[list[tuple[float, float]]], list[dict[str, object]]]:
    """Two people (one with both hands, one with a missing shoulder) + one loose hand on a 600x450 canvas."""
    w, h = 600, 450
    rng = random.Random(3)
    a = skeleton(upper_arm=(150, -90), forearm=(-120, -90))
    b = skeleton(torso=170, thigh=(90, 90), upper_arm=(180, 40), forearm=(170, -50))
    b = hide(b, 6)
    people_src = []
    for pts, dx in ((a, -180.0), (b, 160.0)):
        # skeleton() draws around (500, 600) at T=100 px; shrink into the canvas.
        people_src.append(rounded([(((x - 500) * 0.45 + 300 + dx), ((y - 600) * 0.45 + 300), v) for x, y, v in pts]))
    people = []
    for i, pts in enumerate(people_src):
        body = coco_to_openpose(pts, w, h)
        left = right = None
        if i == 0:
            lw, rw = pts[9], pts[10]
            left = [(round(x / w, 5), round(y / h, 5)) for x, y in fan_hand(lw[0], lw[1], 70, rng)]
            right = [(round(x / w, 5), round(y / h, 5)) for x, y in fan_hand(rw[0], rw[1], 70, rng)]
        people.append(OpenPosePerson(body, left, right))
    loose = [[(round(x / w, 5), round(y / h, 5)) for x, y in fan_hand(520, 420, 60, rng)]]
    return w, h, people, loose, [{"coco": [list(p) for p in pts]} for pts in people_src]


def samples(canvas, people: list[OpenPosePerson], w: int, h: int) -> list[dict[str, object]]:
    out = []
    for person in people:
        kp = person.body
        for (k1, k2), _color in zip(LIMB_SEQ, COLORS, strict=False):
            p1, p2 = kp[k1 - 1], kp[k2 - 1]
            if p1 is None or p2 is None:
                continue
            x = int((p1[0] * w + p2[0] * w) / 2)
            y = int((p1[1] * h + p2[1] * h) / 2)
            out.append({"kind": "limb", "x": x, "y": y, "rgb": [int(c) for c in canvas[y, x]]})
        for p in kp:
            if p is None:
                continue
            x, y = int(p[0] * w), int(p[1] * h)
            out.append({"kind": "joint", "x": x, "y": y, "rgb": [int(c) for c in canvas[y, x]]})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pose_names.json").write_text(json.dumps(pose_name_vectors()))
    w, h, people, loose, sources = scene()
    full = draw_poses(people, h, w, extra_hands=loose)
    body_only = draw_poses(people, h, w, draw_hands=False)
    # mtime=0: the same render gives the same bytes (no spurious diffs when regenerated).
    (OUT / "openpose_full.rgb.gz").write_bytes(gzip.compress(full.tobytes(), mtime=0))
    (OUT / "openpose_body.rgb.gz").write_bytes(gzip.compress(body_only.tobytes(), mtime=0))
    assert (cv2.cvtColor(to_bgr(full), cv2.COLOR_BGR2RGB) == full).all()  # what the server's PNG would hold
    openpose = {
        "coco_to_openpose": coco_vectors(),
        "assign_hands": assign_vectors(),
        "scene": {
            "w": w,
            "h": h,
            "people": [
                {
                    "coco": src["coco"],
                    "left_hand": None if p.left_hand is None else [list(q) for q in p.left_hand],
                    "right_hand": None if p.right_hand is None else [list(q) for q in p.right_hand],
                }
                for p, src in zip(people, sources, strict=True)
            ],
            "loose_hands": [[list(q) for q in hand] for hand in loose],
            "samples_full": samples(full, people, w, h),
            "samples_body": samples(body_only, people, w, h),
            "json_text": json.dumps(encode_pose_json(people, h, w)),
        },
    }
    (OUT / "openpose.json").write_text(json.dumps(openpose))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
