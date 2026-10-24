"""Read-text quality on the eval set of `tools/ocr_eval.py` (docs/OCR-RESEARCH.md P0): rendered signs (rotations,
perspective, tilt, paragraph, Cyrillic, Arabic, CJK, Greek) + real photos (bus.jpg + Wikimedia Commons, fetched once
into tests/data/ocr/, skipped offline). The floors below are the engine's measured level minus a small margin, so a
change that makes reading worse fails here. Run with -s for the full table."""

import pytest

from tools.ocr_eval import all_samples, cer, evaluate, partial_match, summarize, table

# Measured (RTX 3090) / floor. EasyOCR baseline (docs/research/ocr-eval/): 28/67 keys, 15/42 lines, 1.52 boxes per
# line, 0/5 rotated, median 901 ms on photos (max 67 s on the Del Taco menu).
MIN_KEY_RECALL = 0.40
MIN_LINES_EXACT = 14
MAX_BOXES_PER_LINE = 1.6
MIN_ROTATED_OK = 0
MAX_MEDIAN_MS_PHOTOS = 1500


def test_partial_match() -> None:
    assert partial_match("notausgang", "bittenotausgangfreihalten") == 1.0
    assert partial_match("notausgang", "xxnotausgamgxx") == pytest.approx(0.9)
    assert partial_match("東京", "とうきょう東京駅") == 1.0
    assert partial_match("abc", "") == 0.0


def test_cer() -> None:
    assert cer("abcd", "abcd") == 0.0
    assert cer("abcd", "abxd") == 0.25
    assert cer("", "") == 0.0


@pytest.fixture(scope="module")
def scores():
    from glass_vision.pipelines.text import TextPipeline

    pipeline = TextPipeline()
    pipeline.load("cuda:0")
    try:
        scores = evaluate(lambda image: pipeline.read_still(image), all_samples())
    finally:
        pipeline.unload()
    print("\n" + table({"current": scores}))
    return scores


@pytest.mark.gpu
def test_rendered_signs(scores) -> None:
    s = summarize(scores)
    assert s.lines_exact >= MIN_LINES_EXACT, table({"current": scores})
    assert s.boxes_per_line <= MAX_BOXES_PER_LINE
    assert s.rotated_ok >= MIN_ROTATED_OK


@pytest.mark.gpu
def test_photos(scores) -> None:
    s = summarize(scores)
    if s.keys_total <= 1:  # only bus.jpg: Commons photos not available (offline)
        pytest.skip("Commons photos not available")
    assert s.key_recall >= MIN_KEY_RECALL, table({"current": scores})
    assert s.median_ms_photos <= MAX_MEDIAN_MS_PHOTOS
