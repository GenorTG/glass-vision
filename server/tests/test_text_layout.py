"""Reading order + blocks of still OCR lines (docs/PROTOCOL.md "Read text"); pure geometry, no model."""

from glass_vision.pipelines.text_layout import reading_order


def texts(boxes: dict[str, tuple[float, float, float, float]]) -> list[tuple[str, int]]:
    names = list(boxes)
    return [(names[i], block) for i, block in reading_order(list(boxes.values()))]


def test_empty() -> None:
    assert reading_order([]) == []


def test_stacked_lines_of_one_sign_are_one_block_in_order() -> None:
    sign = {
        "line 3": (100, 260, 380, 300),
        "line 1": (100, 100, 400, 140),
        "line 2": (120, 180, 390, 220),
    }
    assert texts(sign) == [("line 1", 0), ("line 2", 0), ("line 3", 0)]


def test_two_signs_side_by_side_are_two_blocks_left_first() -> None:
    boxes = {
        "right 2": (700, 160, 900, 200),
        "left 1": (100, 100, 400, 140),
        "right 1": (700, 100, 900, 140),
        "left 2": (100, 160, 380, 200),
    }
    assert texts(boxes) == [("left 1", 0), ("left 2", 0), ("right 1", 1), ("right 2", 1)]


def test_a_row_split_into_pieces_reads_left_to_right() -> None:
    # EasyOCR splits a line at a wide word gap; the pieces stay one block, slightly different tops still one row.
    boxes = {"Notausgang": (380, 104, 620, 140), "Ausgang": (100, 100, 340, 140), "below": (100, 170, 300, 210)}
    assert texts(boxes) == [("Ausgang", 0), ("Notausgang", 0), ("below", 0)]


def test_title_and_small_print_are_different_blocks() -> None:
    boxes = {
        "BIG TITLE": (100, 100, 700, 220),  # 120 px high
        "small print": (100, 240, 500, 270),  # 30 px high, right below
        "far away": (100, 600, 400, 630),  # same height as the small print, far below
    }
    assert texts(boxes) == [("BIG TITLE", 0), ("small print", 1), ("far away", 2)]


def test_blocks_ordered_top_to_bottom_then_left_to_right() -> None:
    boxes = {
        "bottom": (100, 500, 300, 540),
        "top right": (600, 102, 800, 142),
        "top left": (100, 100, 300, 140),
    }
    assert texts(boxes) == [("top left", 0), ("top right", 1), ("bottom", 2)]
