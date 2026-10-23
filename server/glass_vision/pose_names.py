"""Rule-based pose names from the 17 COCO keypoints of one person: "Standing · Arms up", "Sitting", "T-pose"…

`pose_name(points)` returns `<posture> · <arms>`, either part alone, "Upper body" (shoulders seen, nothing to say) or
"person" (not enough points). Pure geometry on one frame; the tracker's label vote smooths flicker between names.

Conventions:
- Points are pixel (x, y, visibility) in image coordinates (y down); visibility < MIN_VIS = not seen.
- Distances are divided by the torso length T (shoulder midpoint ↔ hip midpoint), so thresholds are scale-free.
  Without hips T is estimated from the shoulder width or the head.
- Left/right in joint names are the person's own sides (COCO index: 5 = left shoulder…). Only "Pointing left/right"
  is in IMAGE terms: the side of the picture the arm points to (a person facing the camera pointing with their
  right arm, at the image's left, is "Pointing left").

Postures (need shoulders, hips and at least one knee; the torso tilt θ is its angle from straight up):
- Bending over: θ > 45° and the legs (hip → ankle/knee) within 30° of straight down.
- Lying down: 60° ≤ θ ≤ 120° and legs not vertical (or not seen).
- With the torso upright-ish (θ ≤ 60°), per leg: thigh drop = (knee.y - hip.y) / T, shin drop = (ankle.y - knee.y) / T
  - Kneeling: a knee on the floor: shin drop < 0.3 with the thigh hanging (thigh drop > 0.45).
  - Standing: thigh drop > 0.6 and the knee straight (> 140°) or the ankle unseen.
  - Squatting: hips close to the ankles ((ankle.y - hip.y) / T < 0.75) with a vertical-ish shin (drop ≥ 0.3).
  - Sitting: thigh drop < 0.45 (thigh horizontal or pointing at the camera).
Arms (need shoulders + wrists; elbows for "straight"; first match wins), in the torso's frame when lying down:
- Hand(s) on head: wrist within 0.45 T of the head centre (nose/eyes/ears) and above it (not a hand at the chin).
- Arms up: both wrists > 0.3 T above their shoulder.
- T-pose: both arms straight (elbow > 150°), within 25° of horizontal, pointing away from the body, reach > 0.6 T.
- Arms crossed: both wrists across the body's midline, between shoulder and hip height.
- Hands on hips: both wrists within 0.4 T of their hip with the elbows out (bent < 130°, elbow further out than wrist).
- Pointing left/right: one arm (or both, the same way) straight, within 35° of horizontal, reach > 0.6 T.
- One arm up: exactly one wrist > 0.3 T above its shoulder (a wave, a raised hand).
No arm label while bending over: hanging arms there are not a gesture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

MIN_VIS = 0.3
SEP = " · "

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
SIDES = ((L_SHOULDER, L_ELBOW, L_WRIST, L_HIP), (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP))
LEGS = ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))

# Typical proportions relative to the torso length, for the estimate without hips.
SHOULDER_WIDTH_PER_TORSO = 0.75
HEAD_TO_NECK_PER_TORSO = 0.45

Vec = np.ndarray  # float64 shape (2,)


def _tilt(v: Vec) -> float:
    """Angle in degrees between v and straight up (0, -1) in image coordinates: 0 = up, 90 = sideways, 180 = down."""
    n = float(np.hypot(*v))
    if n < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(v[1]) / n))))


def _angle(a: Vec, b: Vec, c: Vec) -> float:
    """Inner angle at b of a-b-c in degrees (180 = straight)."""
    u, w = a - b, c - b
    nu, nw = float(np.hypot(*u)), float(np.hypot(*w))
    if nu < 1e-9 or nw < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ w) / (nu * nw)))))


class _Body:
    """Visible keypoints of one person as vectors, plus the torso scale and frame."""

    def __init__(self, points: Sequence[tuple[float, float, float]]) -> None:
        self.p: list[Vec | None] = [
            np.array([x, y], np.float64) if v >= MIN_VIS and math.isfinite(x) and math.isfinite(y) else None
            for x, y, v in list(points)[:17]
        ]
        self.p += [None] * (17 - len(self.p))
        self.shoulders = self._mid(L_SHOULDER, R_SHOULDER)
        self.hips = self._mid(L_HIP, R_HIP)
        head = [self.p[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if self.p[i] is not None]
        self.head: Vec | None = np.mean(head, axis=0) if head else None
        self.scale = self._scale()

    def _mid(self, a: int, b: int) -> Vec | None:
        pa, pb = self.p[a], self.p[b]
        if pa is not None and pb is not None:
            return (pa + pb) / 2
        return pa if pa is not None else pb

    def _scale(self) -> float | None:
        if self.shoulders is not None and self.hips is not None:
            t = float(np.hypot(*(self.shoulders - self.hips)))
            if t > 1e-6:
                return t
        estimates: list[float] = []
        ls, rs = self.p[L_SHOULDER], self.p[R_SHOULDER]
        if ls is not None and rs is not None:
            estimates.append(float(np.hypot(*(ls - rs))) / SHOULDER_WIDTH_PER_TORSO)
        if self.head is not None and self.shoulders is not None:
            estimates.append(float(np.hypot(*(self.head - self.shoulders))) / HEAD_TO_NECK_PER_TORSO)
        best = max(estimates, default=0.0)
        return best if best > 1e-6 else None

    def rotated_upright(self) -> _Body:
        """Copy rotated about the shoulder midpoint so the torso points straight up (for lying postures)."""
        assert self.shoulders is not None and self.hips is not None
        d = self.shoulders - self.hips
        a = math.atan2(float(d[0]), -float(d[1]))  # tilt, signed: + = leaning towards +x
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, s], [-s, c]])  # rotates the torso direction onto (0, -1)
        out = object.__new__(_Body)
        out.p = [None if q is None else self.shoulders + rot @ (q - self.shoulders) for q in self.p]
        out.shoulders = out._mid(L_SHOULDER, R_SHOULDER)
        out.hips = out._mid(L_HIP, R_HIP)
        out.head = None if self.head is None else self.shoulders + rot @ (self.head - self.shoulders)
        out.scale = self.scale
        return out


def _posture(b: _Body) -> str | None:
    if b.shoulders is None or b.hips is None or b.scale is None:
        return None
    t = b.scale
    theta = _tilt(b.shoulders - b.hips)

    knees = [b.p[k] for _, k, _ in LEGS if b.p[k] is not None]
    ankles = [b.p[a] for _, _, a in LEGS if b.p[a] is not None]
    foot = np.mean(ankles, axis=0) if ankles else (np.mean(knees, axis=0) if knees else None)
    legs_down = foot is not None and _tilt(b.hips - foot) < 30 and float(foot[1] - b.hips[1]) > 0.5 * t

    if theta > 45 and legs_down:
        return "Bending over"
    if 60 <= theta <= 120:
        return "Lying down"
    if theta > 60 or not knees:
        return None

    kneeling = standing = squatting = sitting = 0
    for hip_i, knee_i, ankle_i in LEGS:
        own_hip, knee, ankle = b.p[hip_i], b.p[knee_i], b.p[ankle_i]
        hip = own_hip if own_hip is not None else b.hips
        if knee is None:
            continue
        thigh_drop = float(knee[1] - hip[1]) / t
        if ankle is None:
            if thigh_drop > 0.6:
                standing += 1
            elif thigh_drop < 0.45:
                sitting += 1
            continue
        shin_drop = float(ankle[1] - knee[1]) / t
        knee_angle = _angle(hip, knee, ankle)
        if shin_drop < 0.3 and thigh_drop > 0.45:
            kneeling += 1
        elif thigh_drop > 0.6 and knee_angle > 140:
            standing += 1
        elif float(ankle[1] - hip[1]) / t < 0.75 and shin_drop >= 0.3:
            squatting += 1
        elif thigh_drop < 0.45:
            sitting += 1
        else:
            standing += 1  # half-bent, thigh mostly down: walking/leaning rather than sitting
    if kneeling:
        return "Kneeling"  # one knee down is enough
    votes = {"Squatting": squatting, "Sitting": sitting, "Standing": standing}
    best = max(votes, key=lambda k: votes[k])
    return best if votes[best] else None


def _arm_straight(b: _Body, s: int, e: int, w: int) -> bool:
    ps, pe, pw = b.p[s], b.p[e], b.p[w]
    return ps is not None and pe is not None and pw is not None and _angle(ps, pe, pw) > 150


def _arms(b: _Body, image: _Body) -> str | None:
    """`b` = the body in the frame used for up/down; `image` = the same body in image coordinates (directions)."""
    t = b.scale
    if t is None or b.p[L_SHOULDER] is None and b.p[R_SHOULDER] is None:
        return None
    wrists = [(s, e, w, h) for s, e, w, h in SIDES if b.p[s] is not None and b.p[w] is not None]

    if b.head is not None:
        head = b.head
        on_head = sum(
            1 for _, _, w, _ in wrists if float(np.hypot(*(b.p[w] - head))) < 0.45 * t and b.p[w][1] < head[1]
        )
        if on_head:
            return "Hands on head" if on_head == 2 else "Hand on head"

    raised = [w for s, _, w, _ in wrists if float(b.p[s][1] - b.p[w][1]) > 0.3 * t]
    if len(raised) == 2:
        return "Arms up"

    mid_x = float(b.shoulders[0]) if b.shoulders is not None else None
    horizontal: list[tuple[int, int]] = []  # (shoulder, wrist) of straight, horizontal, long arms
    for s, e, w, _ in wrists:
        d = b.p[w] - b.p[s]
        reach = float(np.hypot(*d))
        if _arm_straight(b, s, e, w) and reach > 0.6 * t and abs(90 - _tilt(d)) < 35:
            horizontal.append((s, w))
    if len(horizontal) == 2 and mid_x is not None:
        outward = all(
            (float(b.p[w][0]) - mid_x) * (float(b.p[s][0]) - mid_x) > 0 and abs(90 - _tilt(b.p[w] - b.p[s])) < 25
            for s, w in horizontal
        )
        if outward:
            return "T-pose"

    if len(wrists) == 2 and mid_x is not None and b.hips is not None:
        top, bottom = float(b.shoulders[1]), float(b.hips[1])
        crossed = all(
            (float(b.p[w][0]) - mid_x) * (float(b.p[s][0]) - mid_x) < 0
            and top - 0.1 * t <= float(b.p[w][1]) <= bottom + 0.1 * t
            for s, _, w, _ in wrists
        )
        if crossed:
            return "Arms crossed"

        def on_hip(s: int, e: int, w: int, h: int) -> bool:
            hip, elbow = b.p[h] if b.p[h] is not None else b.hips, b.p[e]
            if elbow is None or float(np.hypot(*(b.p[w] - hip))) > 0.4 * t:
                return False
            out_elbow = abs(float(elbow[0]) - mid_x) > abs(float(b.p[w][0]) - mid_x)
            return out_elbow and _angle(b.p[s], elbow, b.p[w]) < 130

        if all(on_hip(*side) for side in wrists):
            return "Hands on hips"

    directions = {"Pointing left" if image.p[w][0] < image.p[s][0] else "Pointing right" for s, w in horizontal}
    if len(directions) == 1:  # one straight horizontal arm, or both towards the same side
        return directions.pop()

    if len(raised) == 1:
        return "One arm up"
    return None


def pose_name(points: Sequence[tuple[float, float, float]]) -> str:
    """17 COCO keypoints (pixel x, y, visibility) of one person → a short human-readable pose label."""
    b = _Body(points)
    if b.scale is None:
        return "person"
    posture = _posture(b)
    if posture == "Bending over":
        arms = None
    elif posture == "Lying down":
        arms = _arms(b.rotated_upright(), b)
    else:
        arms = _arms(b, b)
    if posture and arms:
        return f"{posture}{SEP}{arms}"
    if posture or arms:
        return posture or arms or ""
    if b.p[L_SHOULDER] is not None or b.p[R_SHOULDER] is not None:
        return "Upper body"
    return "person"
