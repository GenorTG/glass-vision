"""Geometry of OCR'd lines on a still (docs/PROTOCOL.md "Read text"): reading frame, row pieces → lines, reading
order and blocks, block kind.

Pure geometry/text, no model.
- Reading frame (`Frame`): the still turned by the dominant text orientation (0/90/180/270, `angle`) and deskewed by
  the median line tilt, so layout works on rotated and tilted signs as on upright ones. Coordinates go both ways.
- Row pieces (`merge_rows`): the detector sometimes splits one row at wide letter or word spacing ("SIDI" "EL"
  "WASTI"); pieces of similar height on one baseline, at most ~1 line height apart, become one line.
- Blocks (`reading_order`): two lines are in one block (same sign / paragraph) when they have a similar height and are
  close: stacked with a gap of at most ~one line height and overlapping horizontally, or side by side on one row with a
  gap of at most 1.5 line heights. Blocks are the connected components of that relation. Output: block by block
  (blocks top to bottom, then left to right; blocks starting at about the same height count as one row), and inside a
  block the lines top to bottom, then left to right; so a block's lines are contiguous and the client can group them
  without sorting.
- Block kind (`block_kind`): "prose" (sentences running over lines: translated as one text) or "list" (sign items,
  menu, timetable: line by line).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

HEIGHT_RATIO = 1.6  # taller / shorter line: above this they're different text sizes → different blocks
STACK_GAP = 1.0  # vertical gap between stacked lines, × the mean line height
ROW_GAP = 1.5  # horizontal gap between pieces of one row, × the mean line height
ROW_OVERLAP = 0.5  # pieces of one row overlap vertically by at least this × the shorter height
SAME_ROW = 0.5  # rows: centre (lines) / top (blocks) within this × the first one's height
# merge_rows: pieces of one line (stricter than ROW_GAP: two signs side by side stay two lines of one block)
PIECE_HEIGHT_RATIO = 1.3
PIECE_OVERLAP = 0.6
PIECE_GAP = 1.2
MIN_SKEW_DEG = 1.0  # smaller median tilts are left alone
MAX_SKEW_DEG = 45.0


@dataclass(frozen=True)
class Frame:
    """Image pixels ↔ reading frame: rotate clockwise by `angle` (image of size w x h), then by -`skew` degrees about
    the rotated image's centre (positive skew = lines rising to the right... in image coordinates, y down: lines whose
    right end is lower)."""

    w: int
    h: int
    angle: int = 0
    skew: float = 0.0

    def _rotated_size(self) -> tuple[int, int]:
        return (self.h, self.w) if self.angle in (90, 270) else (self.w, self.h)

    def forward(self, pts: np.ndarray) -> np.ndarray:
        x, y = pts[..., 0], pts[..., 1]
        if self.angle == 90:
            x, y = self.h - y, x
        elif self.angle == 180:
            x, y = self.w - x, self.h - y
        elif self.angle == 270:
            x, y = y, self.w - x
        out = np.stack([x, y], axis=-1).astype(np.float64)
        return self._deskew(out, -self.skew)

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        p = self._deskew(np.asarray(pts, np.float64), self.skew)
        x, y = p[..., 0], p[..., 1]
        if self.angle == 90:
            x, y = y, self.h - x
        elif self.angle == 180:
            x, y = self.w - x, self.h - y
        elif self.angle == 270:
            x, y = self.w - y, x
        return np.stack([x, y], axis=-1)

    def _deskew(self, pts: np.ndarray, degrees: float) -> np.ndarray:
        if not degrees:
            return pts
        rw, rh = self._rotated_size()
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        centre = np.array([rw / 2, rh / 2])
        d = pts - centre
        return np.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], axis=-1) + centre


def line_skew(quads: Sequence[np.ndarray]) -> float:
    """Median tilt (degrees, y down) of reading-order quads [TL, TR, BR, BL], weighted by length; 0 if under
    MIN_SKEW_DEG or unclear."""
    angles, weights = [], []
    for q in quads:
        top, bottom = q[1] - q[0], q[2] - q[3]
        v = (top + bottom) / 2
        length = float(np.hypot(*v))
        if length > 0:
            angles.append(math.degrees(math.atan2(v[1], v[0])))
            weights.append(length)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cumulative = np.cumsum(np.array(weights)[order])
    median = float(np.array(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
    return median if MIN_SKEW_DEG <= abs(median) <= MAX_SKEW_DEG else 0.0


def bbox(points: np.ndarray) -> Box:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _height(b: Box) -> float:
    return max(b[3] - b[1], 1e-6)


# --- row pieces -------------------------------------------------------------------------------------------------------
def merge_rows(boxes: Sequence[Box]) -> list[list[int]]:
    """Groups of piece indices that form one line, each left to right (reading-frame boxes)."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ha, hb = _height(a), _height(b)
            if max(ha, hb) / min(ha, hb) > PIECE_HEIGHT_RATIO:
                continue
            if min(a[3], b[3]) - max(a[1], b[1]) < PIECE_OVERLAP * min(ha, hb):
                continue
            gap = max(a[0], b[0]) - min(a[2], b[2])
            # overlapping a lot horizontally = stacked duplicates/other text, not pieces of one row
            if -0.5 * min(a[2] - a[0], b[2] - b[0]) <= gap <= PIECE_GAP * (ha + hb) / 2:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g, key=lambda i: boxes[i][0]) for g in groups.values()]


# --- blocks -----------------------------------------------------------------------------------------------------------
def _linked(a: Box, b: Box) -> bool:
    ha, hb = _height(a), _height(b)
    if max(ha, hb) / min(ha, hb) > HEIGHT_RATIO:
        return False
    mean = (ha + hb) / 2
    overlap_y = min(a[3], b[3]) - max(a[1], b[1])
    gap_x = max(a[0], b[0]) - min(a[2], b[2])  # < 0: horizontal ranges overlap
    if overlap_y >= ROW_OVERLAP * min(ha, hb):  # same row
        return gap_x <= ROW_GAP * mean
    gap_y = -overlap_y
    return gap_x < 0 and gap_y <= STACK_GAP * mean


def _rows(items: list[int], top: Callable[[int], float], height: Callable[[int], float]) -> list[list[int]]:
    """Sort `items` top to bottom into rows: an item joins the current row if its `top` is within SAME_ROW × the
    row's first item's height of that item's."""
    rows: list[list[int]] = []
    for i in sorted(items, key=top):
        if rows and top(i) - top(rows[-1][0]) <= SAME_ROW * height(rows[-1][0]):
            rows[-1].append(i)
        else:
            rows.append([i])
    return rows


def reading_order(boxes: Sequence[Box]) -> list[tuple[int, int]]:
    """[(index into `boxes`, block id)] in reading order; block ids count up in that order from 0."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _linked(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    blocks: dict[int, list[int]] = {}
    for i in range(n):
        blocks.setdefault(find(i), []).append(i)

    def block_box(members: list[int]) -> Box:
        return (
            min(boxes[i][0] for i in members),
            min(boxes[i][1] for i in members),
            max(boxes[i][2] for i in members),
            max(boxes[i][3] for i in members),
        )

    groups = list(blocks.values())
    outer = [block_box(g) for g in groups]
    first_line = [_height(boxes[min(g, key=lambda i: boxes[i][1])]) for g in groups]
    block_rows = _rows(list(range(len(groups))), lambda k: outer[k][1], lambda k: first_line[k])

    out: list[tuple[int, int]] = []
    block_id = 0
    for row in block_rows:
        for k in sorted(row, key=lambda k: outer[k][0]):
            line_rows = _rows(groups[k], lambda i: (boxes[i][1] + boxes[i][3]) / 2, lambda i: _height(boxes[i]))
            for line_row in line_rows:
                out += [(i, block_id) for i in sorted(line_row, key=lambda i: boxes[i][0])]
            block_id += 1
    return out


# --- block kind -------------------------------------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"[.!?。！？:;…]\W*$")
PROSE = "prose"
LIST = "list"
PROSE_MIN_WORDS = 4  # mean words per line of a paragraph without continuation clues (justified text)
PROSE_WIDTH_SPREAD = 0.25  # its line widths: std / mean at most this (last line excluded)


def _continues(line: str, following: str) -> bool:
    """`following` continues `line`'s sentence: no sentence end and a lowercase start, or a hyphenated break."""
    line, following = line.rstrip(), following.lstrip()
    if not line or not following:
        return False
    if line.endswith(("-", "‐", "­")) and following[0].isalpha():
        return True
    if _SENTENCE_END.search(line):
        return False
    first = following[0]
    return first.islower() or line.endswith(",")


def block_kind(texts: Sequence[str], widths: Sequence[float]) -> str:
    """ "prose" when the lines read as sentences running on (lowercase continuations, hyphenated breaks, commas), or
    as a justified paragraph (several words per line, even widths); else "list" (one item per line). One line: prose
    if it is a sentence of PROSE_MIN_WORDS+ words, else list."""
    words = [len(t.split()) for t in texts]
    if len(texts) == 1:
        return PROSE if words[0] >= PROSE_MIN_WORDS else LIST
    links = sum(_continues(a, b) for a, b in pairwise(texts))
    if links * 2 >= len(texts) - 1:
        return PROSE
    body = list(widths[:-1]) if len(widths) > 2 else list(widths)
    mean_width = sum(body) / len(body)
    spread = float(np.std(body)) / mean_width if mean_width > 0 else 1.0
    if sum(words) / len(words) >= PROSE_MIN_WORDS and spread <= PROSE_WIDTH_SPREAD and links > 0:
        return PROSE
    return LIST


def join_prose(texts: Sequence[str]) -> str:
    """A prose block's lines as one paragraph: a line ending in a hyphen before a lowercase word is joined without the
    hyphen ("remon-" + "tem" → "remontem"); other lines with a space. CJK lines are joined without spaces."""
    out = ""
    for text in (t.strip() for t in texts):
        if not text:
            continue
        if not out:
            out = text
        elif out.endswith(("-", "‐", "­")) and text[0].islower() and len(out) > 1 and out[-2].isalpha():
            out = out[:-1] + text
        elif _is_cjk(out[-1]) and _is_cjk(text[0]):
            out += text
        else:
            out += " " + text
    return out


def _is_cjk(ch: str) -> bool:
    return "　" <= ch <= "鿿" or "가" <= ch <= "힯" or "＀" <= ch <= "￯"


_SENTENCES = re.compile(r"(?<=[.!?。！？])\s+(?=\S)")


def sentences(text: str) -> list[str]:
    """Split a paragraph at sentence ends (the translator truncates long inputs)."""
    return [s for s in _SENTENCES.split(text) if s.strip()]
