"""Text (OCR) pipeline against the real EasyOCR models on a synthetic sign. Run with -s to see timing + VRAM."""

import os

# The GTX 1070 (SM 6.1) makes torch refuse cuDNN process-wide, which the OCR recognizer needs; hide it before
# CUDA initializes (no-op if the environment already restricts devices).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import difflib
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from glass_vision.pipelines.base import Annotation, RunContext
from glass_vision.pipelines.text import TextPipeline

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

W, H = 960, 720
POLISH = "Zażółć gęślą jaźń"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
Box = tuple[float, float, float, float]


def make_sign() -> tuple[np.ndarray, dict[str, Box]]:
    """White 960x720 BGR canvas with known text lines → (image, {text: drawn box})."""
    img = np.full((H, W, 3), 255, np.uint8)
    drawn: dict[str, Box] = {}
    for text, y in (("GLASS VISION", 150), ("TEST 2026", 330)):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        cv2.putText(img, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
        drawn[text] = (60, y - th, 60 + tw, y + base)
    if FONT.exists():
        pil = Image.fromarray(img)
        font = ImageFont.truetype(str(FONT), 64)
        ImageDraw.Draw(pil).text((60, 480), POLISH, font=font, fill=(0, 0, 0))
        drawn[POLISH] = ImageDraw.Draw(pil).textbbox((60, 480), POLISH, font=font)
        img = np.asarray(pil).copy()
    return img, drawn


def ctx(languages: str = "en+pl", min_conf: float = 0.3) -> RunContext:
    return RunContext(options={"languages": languages, "min_conf": min_conf}, conf=0.25, imgsz=640)


def read_region(anns: list[Annotation], region: Box, pad: float = 20) -> str:
    """Labels whose box centre lies in `region` (padded), left to right — OCR may split a line into words."""
    x1, y1, x2, y2 = region
    inside = [
        a
        for a in anns
        if x1 - pad <= (a.box[0] + a.box[2]) / 2 <= x2 + pad and y1 - pad <= (a.box[1] + a.box[3]) / 2 <= y2 + pad
    ]
    return " ".join(a.label for a in sorted(inside, key=lambda a: a.box[0]))


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


@pytest.fixture(scope="module")
def pipeline():
    before = torch.cuda.memory_allocated()
    p = TextPipeline()
    p.load("cuda:0")
    print(f"\n[text] VRAM after load (en+pl reader): {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")
    yield p
    p.unload()
    print(f"[text] VRAM after unload: {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")


def test_reads_drawn_text(pipeline):
    img, drawn = make_sign()
    anns = pipeline.run(img, ctx())
    assert anns, "no text found"
    for a in anns:
        x1, y1, x2, y2 = a.box
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        assert a.label == a.label.strip() and a.label
        assert 0.3 <= a.conf <= 1
    for text, region in drawn.items():
        got = read_region(anns, region)
        print(f"[text] {text!r} → {got!r} ({similar(got, text):.2f})")
        assert similar(got, text) >= 0.8, (text, got, anns)


def test_downscales_and_maps_boxes_back(pipeline):
    img, drawn = make_sign()
    big = cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)  # long side 1920 → OCR'd at 1280
    anns = pipeline.run(big, ctx())
    for text, (x1, y1, x2, y2) in drawn.items():
        if text == POLISH:
            continue
        got = read_region(anns, (x1 * 2, y1 * 2, x2 * 2, y2 * 2), pad=40)
        assert similar(got, text) >= 0.8, (text, got, anns)


def test_min_conf_filters(pipeline):
    img, _ = make_sign()
    low = pipeline.run(img, ctx(min_conf=0.1))
    confs = sorted(a.conf for a in low)
    assert len(confs) >= 2
    cut = (confs[0] + confs[1]) / 2  # between the two weakest results: drops at least the weakest one
    high = pipeline.run(img, ctx(min_conf=cut))
    assert [(a.label, a.box) for a in high] == [(a.label, a.box) for a in low if a.conf >= cut]
    assert len(high) < len(low)


def test_language_switch(pipeline):
    img, drawn = make_sign()
    before = torch.cuda.memory_allocated()
    anns = pipeline.run(img, ctx(languages="en"))
    print(f"\n[text] VRAM for the extra 'en' reader: {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")
    assert set(pipeline._readers) == {"en", "en+pl"}
    for text in ("GLASS VISION", "TEST 2026"):
        assert similar(read_region(anns, drawn[text]), text) >= 0.8
    reader = pipeline._readers["en"]
    pipeline.run(img, ctx(languages="en"))
    assert pipeline._readers["en"] is reader  # cached, not rebuilt


def test_timing(pipeline):
    img, _ = make_sign()
    for languages in ("en+pl", "en"):
        pipeline.run(img, ctx(languages=languages))  # warm-up
        times = []
        for _ in range(10):
            t = time.perf_counter()
            pipeline.run(img, ctx(languages=languages))
            times.append((time.perf_counter() - t) * 1000)
        print(f"\n[text] {languages}: median {statistics.median(times):.1f} ms, max {max(times):.1f} ms (960x720)")
