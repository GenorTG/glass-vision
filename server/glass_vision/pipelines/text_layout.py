"""Reading order and blocks of OCR'd lines on a still (docs/PROTOCOL.md "Read text": `lines` order, `block`).

Pure geometry on line boxes, no model. Two lines are in one block (same sign / paragraph) when they have a similar
height and are close: stacked with a gap of at most ~one line height and overlapping horizontally, or side by side on
one row with a gap of at most 1.5 line heights (EasyOCR splits a line at wider word gaps). Blocks are the connected
components of that relation. Output: block by block (blocks top to bottom, then left to right; blocks starting at
about the same height count as one row), and inside a block the lines top to bottom, then left to right; so a block's
lines are contiguous and the client can group them without sorting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

HEIGHT_RATIO = 1.6  # taller / shorter line: above this they're different text sizes → different blocks
STACK_GAP = 1.0  # vertical gap between stacked lines, × the mean line height
ROW_GAP = 1.5  # horizontal gap between pieces of one row, × the mean line height
ROW_OVERLAP = 0.5  # pieces of one row overlap vertically by at least this × the shorter height
SAME_ROW = 0.5  # rows: centre (lines) / top (blocks) within this × the first one's height


def _height(b: Box) -> float:
    return max(b[3] - b[1], 1e-6)


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
