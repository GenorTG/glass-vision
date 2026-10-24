"""Text (OCR) pipeline against the real PP-OCR models on synthetic signs, plus translation end to end.
Run with -s to see timing + VRAM. The ONNX-on-torch executor is checked against onnxruntime (CPU) first."""

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

from glass_vision.pipelines import ppocr
from glass_vision.pipelines.base import Annotation, RunContext
from glass_vision.pipelines.onnx_torch import OnnxTorch
from glass_vision.pipelines.text import (
    AUTO_SCRIPTS,
    DEFAULT_GROUP,
    LOOKALIKES,
    OCR_DIR,
    PRIMARY,
    RECOGNIZERS,
    SCRIPT_GROUPS,
    TextPipeline,
    _suspect,
)
from glass_vision.translate import DEFAULT_MODEL, MODELS, TRANSLATION_DIR

W, H = 960, 720
POLISH = "Zażółć gęślą jaźń"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
Box = tuple[float, float, float, float]
GROUP = {g.key: g for g in SCRIPT_GROUPS}
MODEL_DIRS = sorted(p for p in OCR_DIR.glob("*") if (p / "inference.onnx").is_file()) if OCR_DIR.is_dir() else []
needs_models = pytest.mark.skipif(not MODEL_DIRS, reason="PP-OCR models not downloaded (download_models.py)")


def test_gibberish_detector_flags_low_confidence_and_mixed_scripts_only() -> None:
    assert _suspect("NOTAUSGANG", 0.4)
    assert _suspect("BЫXOД", 0.9)  # Latin/Cyrillic look-alikes in one OCR token
    assert not _suspect("NOTAUSGANG", 0.8)
    assert not _suspect("12:45", 0.8)


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


def ctx(languages: str = AUTO_SCRIPTS, min_conf: float = 0.3, **options) -> RunContext:
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
    """Labels whose box centre lies in `region` (padded), left to right."""
    x1, y1, x2, y2 = region
    inside = [
        a
        for a in anns
        if x1 - pad <= (a.box[0] + a.box[2]) / 2 <= x2 + pad and y1 - pad <= (a.box[1] + a.box[3]) / 2 <= y2 + pad
    ]
    return " ".join(a.label for a in sorted(inside, key=lambda a: a.box[0]))


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# --- the executor ---------------------------------------------------------------------------------------------------
@needs_models
@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=lambda p: p.name)
def test_onnx_on_torch_matches_onnxruntime(model_dir: Path) -> None:
    """Every PP-OCR graph gives onnxruntime's output on the torch executor (CPU, fp32: exact up to rounding)."""
    import onnxruntime as ort

    shape = (1, 3, 160, 224) if "det" in model_dir.name else (1, 3, 224, 224) if "doc_ori" in model_dir.name else (
        (2, 3, 80, 160) if "textline" in model_dir.name else (2, 3, 48, 384)
    )  # fmt: skip
    x = np.random.default_rng(0).random(shape, dtype=np.float32) * 2 - 1
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_dir / "inference.onnx"), options, providers=["CPUExecutionProvider"])
    expected = session.run(None, {session.get_inputs()[0].name: x})[0]
    got = OnnxTorch(model_dir / "inference.onnx", "cpu").run(x)[0]
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-3
    if expected.ndim == 3:  # recognizer: identical characters
        assert (got.argmax(-1) == expected.argmax(-1)).all()


def test_executor_rejects_unknown_ops(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("LpNormalization", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    onnx.save(helper.make_model(graph), tmp_path / "inference.onnx")
    with pytest.raises(NotImplementedError, match="LpNormalization"):
        OnnxTorch(tmp_path / "inference.onnx", "cpu")


def test_crop_and_reading_order_corners() -> None:
    """A tall quad's crop is what `Line.quad` rolls: rotation r → corners rolled r/90 (reading order)."""
    image = np.zeros((100, 200, 3), np.uint8)
    quad = np.float32([[10, 10], [40, 10], [40, 90], [10, 90]])
    crop = ppocr.crop_quad(image, quad)
    assert crop.shape[:2] == (80, 30)
    assert crop.shape[0] >= ppocr.TALL_RATIO * crop.shape[1]  # tall → turned 270° clockwise, read left to right
    assert np.roll(quad, 270 // 90, axis=0)[0].tolist() == [40, 10]  # the text's top-left is the box's top-right


# --- the pipeline ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    before = torch.cuda.memory_allocated()
    p = TextPipeline()
    t0 = time.perf_counter()
    p.load("cuda:0")
    print(
        f"\n[text] load {time.perf_counter() - t0:.1f} s, VRAM {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB"
    )
    yield p
    p.unload()
    print(f"[text] VRAM after unload: {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")


@pytest.mark.gpu
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
        assert similar(got, text) >= 0.9, (text, got, anns)
    assert len(anns) == len(drawn)  # whole lines, not words


@pytest.mark.gpu
def test_downscales_and_maps_boxes_back(pipeline):
    img, drawn = make_sign()
    big = cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)  # long side 1920 → detected at 1280
    anns = pipeline.run(big, ctx())
    for text, (x1, y1, x2, y2) in drawn.items():
        got = read_region(anns, (x1 * 2, y1 * 2, x2 * 2, y2 * 2), pad=40)
        assert similar(got, text) >= 0.9, (text, got, anns)


@pytest.mark.gpu
def test_min_conf_filters(pipeline):
    img, _ = make_sign()
    low = pipeline.run(img, ctx(min_conf=0.1))
    confs = sorted(a.conf for a in low)
    assert len(confs) >= 2
    cut = (confs[0] + confs[1]) / 2  # between the two weakest results: drops at least the weakest one
    high = pipeline.run(img, ctx(min_conf=cut))
    assert [(a.label, a.box) for a in high] == [(a.label, a.box) for a in low if a.conf >= cut]
    assert len(high) < len(low)


@pytest.mark.gpu
@pytest.mark.parametrize("legacy", ["en", "en+pl", "Bengali — BN, AS + EN", "Kannada + EN"])
def test_legacy_language_values(pipeline, legacy):
    """Values saved by older apps still work: "en" / "en+pl" are Latin, the retired EasyOCR scripts read as auto."""
    spec = next(o for o in TextPipeline.info.options if o.key == "languages")
    img, drawn = make_sign()
    anns = pipeline.run(img, ctx(languages=spec.coerce(legacy)))
    assert similar(read_region(anns, drawn["GLASS VISION"]), "GLASS VISION") >= 0.9


@pytest.mark.gpu
def test_mark_mode_boxes_every_line_without_reading(pipeline):
    """`live` = "Mark text areas": detector only, one box per drawn line, label "text"."""
    img, drawn = make_sign()
    anns = pipeline.run(img, ctx(live="Mark text areas"))
    assert anns and all(a.label == "text" and a.conf == 1.0 and a.detail is None for a in anns)
    for text, (x1, y1, x2, y2) in drawn.items():
        hits = [
            a
            for a in anns
            if a.box[0] <= x1 + 20 and a.box[1] <= y1 + 20 and a.box[2] >= x2 - 20 and a.box[3] >= y2 - 20
        ]
        assert len(hits) == 1, (text, anns)  # the whole line, as one box
    assert pipeline.run(np.full((H, W, 3), 255, np.uint8), ctx(live="Mark text areas")) == []


@pytest.mark.gpu
@pytest.mark.parametrize("group", SCRIPT_GROUPS, ids=lambda g: g.key)
def test_every_script_group_reads_english_too(pipeline, group):
    """Every script's recognizer knows ASCII; a chosen script still reads the English sign."""
    for key in group.recognizers:
        if not (OCR_DIR / RECOGNIZERS[key].model).is_dir():
            pytest.skip(f"{RECOGNIZERS[key].model} not downloaded")
    img, drawn = make_sign()
    anns = pipeline.run(img, ctx(languages=group.label))
    assert similar(read_region(anns, drawn["GLASS VISION"]), "GLASS VISION") >= 0.8, anns


@pytest.mark.gpu
@pytest.mark.parametrize("languages", [AUTO_SCRIPTS, GROUP["cyrillic"].label])
def test_reads_cyrillic(pipeline, languages):
    font = font_for("ru")
    if font is None:
        pytest.skip("no font with Cyrillic glyphs (fc-list :lang=ru)")
    img, drawn = render(["Привет мир", "Выход"], font)
    anns = pipeline.run(img, ctx(languages=languages))
    for text, region in drawn.items():
        got = read_region(anns, region)
        print(f"\n[text] cyrillic {text!r} → {got!r} ({similar(got, text):.2f}) using {font.name}")
        assert similar(got, text) >= 0.8, (text, got, anns)


@pytest.mark.gpu
@pytest.mark.parametrize(("lang", "lines"), [("zh", ["出口", "中文"]), ("ja", ["非常口"])])
def test_reads_cjk(pipeline, lang, lines):
    font = font_for(lang)
    if font is None:
        pytest.skip(f"no font with {lang} glyphs (fc-list :lang={lang})")
    img, drawn = render(lines, font, size=96)
    anns = pipeline.run(img, ctx())
    for text, region in drawn.items():
        got = read_region(anns, region)
        print(f"\n[text] {lang} {text!r} → {got!r} using {font.name}")
        assert got == text, (text, anns)


@pytest.mark.gpu
def test_lookalike_script_reading_does_not_replace_confident_latin(pipeline):
    """ "EXIT" read as Greek "ΕΧΙΤ" by the Greek model must not win over a confident Latin reading."""
    assert set("ΕΧΙΤ") <= LOOKALIKES
    crop = np.full((48, 200, 3), 255, np.uint8)
    best = [(ppocr.Reading("EXIT", 0.85, 4), PRIMARY)]

    class Greek:
        def recognize(self, batches):
            return [ppocr.Reading("ΕΧΙΤ", 0.99, 4)] * batches.n

    real = pipeline._recognizer
    pipeline._recognizer = lambda key: Greek() if key == "greek" else None
    try:
        pipeline._script_fallback([crop], best)
    finally:
        pipeline._recognizer = real
    assert best[0][0].text == "EXIT"


@pytest.mark.skipif(
    not (TRANSLATION_DIR / MODELS[DEFAULT_MODEL].dir / "pytorch_model.bin").is_file(),
    reason="M2M100 418M not downloaded (scripts/download_models.py --only text)",
)
@pytest.mark.gpu
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
    print(f"\n[text] translated sign: first run {first:.0f} ms (loads M2M100), "
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


@pytest.mark.gpu
def test_timing(pipeline):
    img, _ = make_sign()
    for live in ("Mark text areas", "Read live"):
        pipeline.run(img, ctx(live=live))  # warm-up
        times = []
        for _ in range(10):
            t = time.perf_counter()
            pipeline.run(img, ctx(live=live))
            times.append((time.perf_counter() - t) * 1000)
        print(f"\n[text] {live}: median {statistics.median(times):.1f} ms, max {max(times):.1f} ms (960x720)")
    assert DEFAULT_GROUP.recognizers == (PRIMARY,)
