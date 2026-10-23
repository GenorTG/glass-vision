"""Text (OCR) pipeline against the real EasyOCR models on synthetic signs, plus translation end to end.
Run with -s to see timing + VRAM."""

import os

# The GTX 1070 (SM 6.1) makes torch refuse cuDNN process-wide, which the OCR recognizer needs; hide it before
# CUDA initializes (no-op if the environment already restricts devices).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import difflib
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from glass_vision.pipelines.base import Annotation, RunContext
from glass_vision.pipelines.text import DEFAULT_GROUP, EASYOCR_DIR, SCRIPT_GROUPS, TextPipeline
from glass_vision.translate import DEFAULT_MODEL, MODELS, TRANSLATION_DIR

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


GROUP = {g.key: g for g in SCRIPT_GROUPS}


def ctx(languages: str = DEFAULT_GROUP.label, min_conf: float = 0.3, **options) -> RunContext:
    return RunContext(options={"languages": languages, "min_conf": min_conf, **options}, conf=0.25, imgsz=640)


def font_for(lang: str) -> Path | None:
    """A TTF/OTF with glyphs for `lang` (fontconfig), or None."""
    if not shutil.which("fc-list"):
        return None
    out = subprocess.run(["fc-list", f":lang={lang}", "file"], capture_output=True, text=True, check=False).stdout
    files = sorted(line.split(":")[0].strip() for line in out.splitlines() if line.strip())
    preferred = [f for f in files if "DejaVuSans.ttf" in f or "DroidSansFallback" in f or "NotoSansCJK" in f]
    candidates = [f for f in preferred + files if f.endswith((".ttf", ".otf", ".ttc"))]
    return Path(candidates[0]) if candidates else None


def render(lines: list[str], font: Path, size: int = 72) -> tuple[np.ndarray, dict[str, Box]]:
    """Black lines of text on a white 960x720 BGR canvas → (image, {text: drawn box})."""
    pil = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(pil)
    face = ImageFont.truetype(str(font), size)
    drawn: dict[str, Box] = {}
    for i, text in enumerate(lines):
        xy = (60, 60 + i * (size * 2))
        draw.text(xy, text, font=face, fill=(0, 0, 0))
        drawn[text] = draw.textbbox(xy, text, font=face)
    return np.asarray(pil)[:, :, ::-1].copy(), drawn


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
    print(f"\n[text] VRAM after load (Latin reader): {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")
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


def test_legacy_language_values(pipeline):
    """Legacy values "en" / "en+pl" (saved by older apps) still work: both are the Latin group now."""
    img, drawn = make_sign()
    for legacy in ("en", "en+pl"):
        anns = pipeline.run(img, ctx(languages=legacy))
        assert similar(read_region(anns, drawn["GLASS VISION"]), "GLASS VISION") >= 0.8
    assert set(pipeline._readers) == {"latin"}


def test_auto_reads_a_sign_after_a_blank_start(pipeline):
    """A phone that starts on a wall: the first auto probe finds nothing, the next frames show a sign. Regression:
    every blank run postponed the next probe, so auto mode never read anything for the rest of the connection."""
    pipeline._auto_group_key, pipeline._auto_probe_after = None, 0.0
    blank = np.full((H, W, 3), 255, np.uint8)
    img, drawn = make_sign()
    auto = ctx(languages=TextPipeline.info.options[0].default)
    assert pipeline.run(blank, auto) == []
    for _ in range(3):  # well inside the retry window: no probe is due
        assert pipeline.run(blank, auto) == []
    anns = pipeline.run(img, auto)
    assert similar(read_region(anns, drawn["GLASS VISION"]), "GLASS VISION") >= 0.8, anns
    assert pipeline._auto_group_key == DEFAULT_GROUP.key


@pytest.mark.parametrize("group", SCRIPT_GROUPS, ids=lambda g: g.key)
def test_every_script_group_builds(pipeline, group):
    if not (EASYOCR_DIR / group.model_file).is_file():
        pytest.skip(f"{group.model_file} not downloaded (scripts/download_models.py --ocr-scripts {group.key})")
    before = torch.cuda.memory_allocated()
    reader = pipeline._reader(group)
    print(f"\n[text] reader {group.key}: +{(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB VRAM")
    assert reader.detector is pipeline._readers[DEFAULT_GROUP.key].detector  # one shared CRAFT detector
    assert pipeline._reader(group) is reader  # cached, not rebuilt
    anns = pipeline.run(make_sign()[0], ctx(languages=group.label))  # every group also reads English
    assert similar(read_region(anns, make_sign()[1]["GLASS VISION"]), "GLASS VISION") >= 0.6, anns


def test_reads_cyrillic(pipeline):
    font = font_for("ru")
    if font is None:
        pytest.skip("no font with Cyrillic glyphs (fc-list :lang=ru)")
    if not (EASYOCR_DIR / GROUP["cyrillic"].model_file).is_file():
        pytest.skip("cyrillic_g2.pth not downloaded (scripts/download_models.py --ocr-scripts cyrillic)")
    img, drawn = render(["Привет мир", "Выход"], font)
    anns = pipeline.run(img, ctx(languages=GROUP["cyrillic"].label))
    for text, region in drawn.items():
        got = read_region(anns, region)
        print(f"\n[text] cyrillic {text!r} → {got!r} ({similar(got, text):.2f}) using {font.name}")
        assert similar(got, text) >= 0.8, (text, got, anns)


@pytest.mark.parametrize(("key", "lang", "lines"), [("zh_sim", "zh", ["出口", "中文"]), ("japanese", "ja", ["非常口"])])
def test_reads_cjk(pipeline, key, lang, lines):
    font = font_for(lang)
    if font is None:
        pytest.skip(f"no font with {lang} glyphs (fc-list :lang={lang})")
    if not (EASYOCR_DIR / GROUP[key].model_file).is_file():
        pytest.skip(f"{GROUP[key].model_file} not downloaded (scripts/download_models.py --ocr-scripts {key})")
    img, drawn = render(lines, font, size=96)
    anns = pipeline.run(img, ctx(languages=GROUP[key].label))
    for text, region in drawn.items():
        got = read_region(anns, region)
        print(f"\n[text] {key} {text!r} → {got!r} using {font.name}")
        assert got == text, (text, anns)


@pytest.mark.skipif(
    not (TRANSLATION_DIR / MODELS[DEFAULT_MODEL].dir / "pytorch_model.bin").is_file(),
    reason="M2M100 418M not downloaded (scripts/download_models.py --only text)",
)
def test_translates_a_german_sign(pipeline):
    img, _ = render(["NOTAUSGANG", "Rauchen verboten", "Tel. 0800 123"], FONT)
    options = {"translate_to": "English", "translate_from": "Auto-detect", "show_original": True}
    t0 = time.perf_counter()
    anns = pipeline.run(img, ctx(**options))
    first = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        again = pipeline.run(img, ctx(**options))
        times.append((time.perf_counter() - t0) * 1000)
    plain = pipeline.run(img, ctx())
    print(f"\n[text] translated sign: first run {first:.0f} ms (loads detector + M2M100), "
          f"cached runs median {statistics.median(times):.0f} ms")  # fmt: skip
    for a in anns:
        print(f"[text]   {a.label!r}  ({a.detail})")
    assert [(a.label, a.detail) for a in again] == [(a.label, a.detail) for a in anns]

    exit_sign = [a for a in anns if a.detail and "NOTAUSGANG" in a.detail]
    assert exit_sign and "exit" in exit_sign[0].label.lower(), anns
    assert exit_sign[0].detail == "DE: NOTAUSGANG"
    smoking = [a for a in anns if a.detail == "DE: Rauchen verboten"]
    assert smoking and "smok" in smoking[0].label.lower(), anns
    phone = [a for a in anns if "0800" in a.label]
    assert phone and phone[0].detail is None  # numbers stay as read
    assert [(a.box, a.conf) for a in anns] == [(a.box, a.conf) for a in plain]  # translation only relabels
    assert all(a.detail is None for a in plain)  # translate_to defaults to Off

    hidden = pipeline.run(img, ctx(**{**options, "show_original": False}))
    assert [a.label for a in hidden] == [a.label for a in anns] and all(a.detail is None for a in hidden)


def test_timing(pipeline):
    img, _ = make_sign()
    for group in (GROUP["latin"], GROUP["cyrillic"]):
        if group.key not in pipeline._readers:
            continue
        pipeline.run(img, ctx(languages=group.label))  # warm-up
        times = []
        for _ in range(10):
            t = time.perf_counter()
            pipeline.run(img, ctx(languages=group.label))
            times.append((time.perf_counter() - t) * 1000)
        print(f"\n[text] {group.key}: median {statistics.median(times):.1f} ms, max {max(times):.1f} ms (960x720)")
