"""Text (OCR): PP-OCRv6 line detection + recognition (PaddleOCR's models, ONNX exports run on torch: `ppocr.py`),
reading signs, labels, screens and menus, optionally translated.

Slow pipeline: the framework runs it in the background on the newest frame (`interval_ms`).

Reading (docs/OCR-RESEARCH.md §3, docs/research/ocr-server-engines.md §7):
1. PP-OCRv6_medium_det finds text *lines* (quads) on the image downscaled to MAX_SIDE (live) / STILL_MAX_SIDE (still).
2. Each line is cut out with a perspective warp from the full-resolution image, turned 90° when it is tall (h/w >=
   1.5: vertical or sideways text) and 180° when PP-LCNet textline_ori says it is upside down.
3. PP-OCRv6_medium_rec reads every line (zh-Hans/zh-Hant/ja/en + 46 Latin-script languages). In auto mode a line it
   reads badly (low confidence, or far fewer characters than the crop's width holds: text outside its alphabet comes
   out blank) is read again by the small PP-OCRv5 script recognizers (Cyrillic, East Slavic, Arabic, Devanagari,
   Korean, Thai, Greek, Telugu, Tamil); a script reading replaces it when it is mostly that script's letters and at
   least as confident. Explicit script choices read every line with that script's recognizer (they all know ASCII too).
4. Stills only: the dominant orientation (0/90/180/270) is first taken from PP-LCNet doc_ori when it is sure (>=
   DOC_ORI_MIN; it misfires on cluttered photos below that), else from the lines' own orientations; lines that
   disagree with it are read again turned that way and keep the more confident reading. The still's `angle` is then
   the vote of the lines as read (doc_ori alone is not trusted: it calls an upright Arabic sign 180° at 0.92). Layout
   runs in that de-rotated, deskewed frame (`text_layout.Frame`).
5. Row pieces are merged into lines, lines grouped into blocks (`text_layout`).
Bengali and Kannada (EasyOCR before v0.7) have no PP-OCR recognizer and are no longer offered.

Live modes (option `live`, docs/PROTOCOL.md "Read text"): "Read live" (recognize every run) or "Mark text areas"
(detector only, label "text"): the app then reads a high-resolution still once (`read_still`, served by
`glass_vision/text_read.py`).

Translation (`glass_vision/translate.py`, M2M100) is off by default and loaded on first use.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .. import memory
from ..translate import DEFAULT_MODEL, LANGUAGES, MODELS, SourceLanguages, Translator, is_translatable, normalize
from . import ppocr
from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, RunContext
from .text_layout import Frame, bbox, block_kind, line_skew, merge_rows, reading_order

log = logging.getLogger(__name__)

OCR_DIR = MODELS_DIR / "ppocr"
DETECTOR = "PP-OCRv6_medium_det"
TEXTLINE_ORI = "PP-LCNet_x1_0_textline_ori"
DOC_ORI = "PP-LCNet_x1_0_doc_ori"
MAX_SIDE = 1280  # live frames: detector input long side (frames are smaller anyway)
STILL_MAX_SIDE = 2560  # `read_still`: detection on this downscale, recognition on full-resolution crops
DOC_ORI_MIN = 0.85  # doc_ori misfired at 0.48 / 0.74 on two scene photos; right answers scored 0.92 (bench)
DOUBTFUL_CONF = 0.9  # auto mode: a PP-OCRv6 reading below this is also tried with the script recognizers
MIN_CHARS_PER_ASPECT = 0.3  # ... as is one with fewer characters than this × crop width / height
SCRIPT_MIN_CONF = 0.6
SCRIPT_MARGIN = 0.05  # a script reading may be this much less confident than PP-OCRv6's and still win
NATIVE_MIN_SHARE = 0.5
TRANSLATE_RETRY_S = 60.0  # after a translator failure (e.g. model not downloaded), OCR goes on untranslated


@dataclass(frozen=True)
class Recognizer:
    key: str
    model: str  # directory under OCR_DIR
    scripts: tuple[str, ...]  # Unicode character-name prefixes of its own letters
    bidi: bool = False


PRIMARY = "v6"
RECOGNIZERS: dict[str, Recognizer] = {
    r.key: r
    for r in (
        Recognizer(PRIMARY, "PP-OCRv6_medium_rec", ("LATIN", "CJK", "HIRAGANA", "KATAKANA")),
        Recognizer("cyrillic", "cyrillic_PP-OCRv5_mobile_rec", ("CYRILLIC",)),
        Recognizer("eslav", "eslav_PP-OCRv5_mobile_rec", ("CYRILLIC",)),
        Recognizer("arabic", "arabic_PP-OCRv5_mobile_rec", ("ARABIC",), bidi=True),
        Recognizer("devanagari", "devanagari_PP-OCRv5_mobile_rec", ("DEVANAGARI",)),
        Recognizer("korean", "korean_PP-OCRv5_mobile_rec", ("HANGUL",)),
        Recognizer("thai", "th_PP-OCRv5_mobile_rec", ("THAI",)),
        Recognizer("greek", "el_PP-OCRv5_mobile_rec", ("GREEK",)),
        Recognizer("telugu", "te_PP-OCRv5_mobile_rec", ("TELUGU",)),
        Recognizer("tamil", "ta_PP-OCRv5_mobile_rec", ("TAMIL",)),
    )
}
SCRIPT_RECOGNIZERS = tuple(k for k in RECOGNIZERS if k != PRIMARY)
# Greek/Cyrillic capitals that look like Latin ones: a script reading made only of these may be Latin text
# ("EXIT" → "ΕΧΙΤ"), so it can't replace a confident Latin reading.
LOOKALIKES = frozenset("АВЕКМНОРСТХУІЈЅаеорсху" + "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧοικνρτυχ")


@dataclass(frozen=True)
class ScriptGroup:
    key: str  # per-line / per-still `script` in responses
    label: str  # option value, shown verbatim in the app
    recognizers: tuple[str, ...]  # RECOGNIZERS keys reading a still when this group is chosen
    sources: SourceLanguages  # translation: source languages detection may pick (M2M100 codes)


def _non_latin(*candidates: str) -> SourceLanguages:
    return SourceLanguages((*candidates, "en"), ascii_is_english=True)


# fmt: off
SCRIPT_GROUPS: tuple[ScriptGroup, ...] = (
    ScriptGroup("latin", "Latin — EN, PL, DE, FR, ES, IT, PT, NL, CS, SV, TR… (46 languages)", (PRIMARY,),
                # Every Latin-script language both lingua and M2M100 know; the first is the fallback.
                SourceLanguages(("en", "pl", "de", "fr", "es", "it", "pt", "nl", "cs", "sk", "sv", "da", "no", "fi",
                                 "hu", "ro", "tr", "hr", "sl", "bs", "sq", "et", "lt", "lv", "is", "ga", "cy", "af",
                                 "az", "id", "ms", "tl", "sw", "vi"))),
    ScriptGroup("cyrillic", "Cyrillic — RU, UK, BE, BG, SR, MN… + EN", ("cyrillic", "eslav"),
                _non_latin("ru", "uk", "be", "bg", "sr", "mn")),
    ScriptGroup("zh_sim", "Chinese (simplified) + EN", (PRIMARY,), _non_latin("zh")),
    ScriptGroup("zh_tra", "Chinese (traditional) + EN", (PRIMARY,), _non_latin("zh")),
    ScriptGroup("japanese", "Japanese + EN", (PRIMARY,), _non_latin("ja")),
    ScriptGroup("korean", "Korean + EN", ("korean",), _non_latin("ko")),
    ScriptGroup("arabic", "Arabic script — AR, FA, UR, UG + EN", ("arabic",), _non_latin("ar", "fa", "ur")),
    ScriptGroup("devanagari", "Devanagari — HI, MR, NE… + EN", ("devanagari",), _non_latin("hi", "mr", "ne")),
    ScriptGroup("thai", "Thai + EN", ("thai",), _non_latin("th")),
    ScriptGroup("greek", "Greek + EN", ("greek",), _non_latin("el")),
    ScriptGroup("telugu", "Telugu + EN", ("telugu",), _non_latin()),  # M2M100 has no Telugu
    ScriptGroup("tamil", "Tamil + EN", ("tamil",), _non_latin("ta")),
)
# fmt: on
GROUPS: dict[str, ScriptGroup] = {g.label: g for g in SCRIPT_GROUPS}
DEFAULT_GROUP = SCRIPT_GROUPS[0]
_GROUP_BY_KEY = {g.key: g for g in SCRIPT_GROUPS}

TRANSLATE_OFF = "Off"
AUTO_DETECT = "Auto-detect"
AUTO_SCRIPTS = "Auto-detect (available scripts)"
# Values older app/server versions saved: `languages` was "en" / "en+pl" (both Latin); EasyOCR (≤ v0.6) offered
# Bengali and Kannada and labelled Latin "(41 languages)".
LEGACY_LANGUAGES = {
    "en": DEFAULT_GROUP.label,
    "en+pl": DEFAULT_GROUP.label,
    "Latin — EN, PL, DE, FR, ES, IT, PT, NL, CS, SV, TR… (41 languages)": DEFAULT_GROUP.label,
    "Bengali — BN, AS + EN": AUTO_SCRIPTS,
    "Kannada + EN": AUTO_SCRIPTS,
}
LIVE_MARK = "Mark text areas"
LIVE_READ = "Read live"
MARK_LABEL = "text"
# Default "Read live", although the app's Read flow wants Mark: installed (v3.1) apps send every option at the value
# `hello` advertises, so a Mark default would turn their live OCR into unlabeled boxes they can't read (no Read button).
# Apps that offer Read (`hello.features` has "text_read") select Mark themselves.
LIVE_DEFAULT = LIVE_READ


@dataclass(frozen=True)
class _ChoiceWithAliases(OptionSpec):
    """A choice that also accepts retired values (mapped to their replacement; not listed in `hello`)."""

    aliases: tuple[tuple[str, str], ...] = ()

    def coerce(self, value: Any) -> Any:
        return super().coerce(dict(self.aliases).get(value, value) if isinstance(value, str) else value)

    def to_json(self) -> dict[str, Any]:
        out = super().to_json()
        out.pop("aliases", None)
        return out


# Unicode character name prefix → script group (`group_for_text`).
_SCRIPT_PREFIXES = {
    "CYRILLIC": "cyrillic",
    "CJK": "zh_sim",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "HANGUL": "korean",
    "ARABIC": "arabic",
    "DEVANAGARI": "devanagari",
    "THAI": "thai",
    "GREEK": "greek",
    "TELUGU": "telugu",
    "TAMIL": "tamil",
}


def group_for_text(lines: Sequence[str]) -> ScriptGroup:
    """The script group text is written in, from its letters' Unicode names (Latin when unclear). Japanese wins as
    soon as kana appear (kanji alone count as Chinese)."""
    counts: Counter[str] = Counter()
    for ch in "".join(lines):
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            counts[next((key for prefix, key in _SCRIPT_PREFIXES.items() if name.startswith(prefix)), "latin")] += 1
    if counts["japanese"]:
        return _GROUP_BY_KEY["japanese"]
    non_latin = [(n, key) for key, n in counts.items() if key != "latin"]
    return _GROUP_BY_KEY[max(non_latin)[1]] if non_latin else DEFAULT_GROUP


def native_share(text: str, recognizer: Recognizer) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(unicodedata.name(ch, "").startswith(recognizer.scripts) for ch in letters) / len(letters)


def rotate_cw(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


@dataclass
class Line:
    """One read text line. `quad` = its corners in the image's pixels, in reading order: top-left, top-right,
    bottom-right, bottom-left *of the text* (rotated with it)."""

    quad: np.ndarray
    text: str
    conf: float
    rotation: int  # clockwise degrees that make this line upright
    recognizer: str
    pieces: int = 1  # detector boxes merged into it

    @property
    def box(self) -> tuple[float, float, float, float]:
        return bbox(self.quad)


@dataclass
class Reading:
    """A still (or frame) read: lines in reading order with their block ids, the frame they were laid out in."""

    lines: list[Line]
    blocks: list[int]  # block id per line
    frame: Frame
    doc_angle: tuple[int, float] | None = None  # doc_ori's answer (stills)
    timings: dict[str, float] = field(default_factory=dict)


class TextPipeline(Pipeline):
    info = PipelineInfo(
        id="text",
        name="Text (OCR)",
        description="Reads text on signs, labels, screens and menus (PP-OCRv6 + script models, 100+ languages); "
        "auto-detects the script per line or reads a chosen one.",
        default_enabled=False,
        uses_global_conf=False,
        interval_ms=400,
        options=[
            OptionSpec("live", "While streaming", "choice", LIVE_DEFAULT, choices=[LIVE_MARK, LIVE_READ]),
            _ChoiceWithAliases(
                "languages",
                "Script / languages",
                "choice",
                AUTO_SCRIPTS,
                choices=[AUTO_SCRIPTS, *GROUPS],
                aliases=tuple(LEGACY_LANGUAGES.items()),
            ),
            OptionSpec("min_conf", "Minimum confidence", "number", 0.3, min=0.1, max=0.9, step=0.05),
            OptionSpec("translate_to", "Translate to", "choice", TRANSLATE_OFF, choices=[TRANSLATE_OFF, *LANGUAGES]),
            OptionSpec("translate_from", "Translate from", "choice", AUTO_DETECT, choices=[AUTO_DETECT, *LANGUAGES]),
            OptionSpec("show_original", "Show the original text too", "bool", True),
            OptionSpec("translation_model", "Translation model", "choice", DEFAULT_MODEL, choices=list(MODELS)),
        ],
    )

    def __init__(self) -> None:
        self._device = "cpu"
        self._detector: ppocr.TextDetector | None = None
        self._textline: ppocr.TextlineOrientation | None = None
        self._doc: ppocr.DocOrientation | None = None
        self._recognizers: dict[str, ppocr.TextRecognizer] = {}
        self._translator: Translator | None = None
        self._translate_failed_at = 0.0
        self._translate_error = ""

    # --- models -----------------------------------------------------------------------------------------------------
    def load(self, device: str) -> None:
        self._device = device if device.startswith("cuda") else "cpu"
        missing = [m for m in (DETECTOR, TEXTLINE_ORI, RECOGNIZERS[PRIMARY].model) if not (OCR_DIR / m).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"OCR models missing in {OCR_DIR}: {', '.join(missing)} (scripts/download_models.py --only text)"
            )
        t0 = time.perf_counter()
        self._detector = ppocr.TextDetector(OCR_DIR / DETECTOR, self._device)
        self._textline = ppocr.TextlineOrientation(OCR_DIR / TEXTLINE_ORI, self._device)
        self._doc = ppocr.DocOrientation(OCR_DIR / DOC_ORI, self._device) if (OCR_DIR / DOC_ORI).is_dir() else None
        self._recognizer(PRIMARY)
        warm = np.full((96, 320, 3), 255, np.uint8)
        cv2.putText(warm, "Glass Vision", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self._read(warm, MAX_SIDE, None, still=False)  # first CUDA kernels / cuDNN plans
        log.info("text: PP-OCR models loaded on %s in %.1f s", self._device, time.perf_counter() - t0)

    def unload(self) -> None:
        self._detector = None
        self._textline = self._doc = None
        self._recognizers.clear()
        if self._translator is not None:
            self._translator.unload()
            self._translator = None
        memory.release()

    def _recognizer(self, key: str) -> ppocr.TextRecognizer | None:
        """Loaded on first use; None if its model is not installed."""
        rec = self._recognizers.get(key)
        if rec is None:
            spec = RECOGNIZERS[key]
            if not (OCR_DIR / spec.model).is_dir():
                return None
            rec = self._recognizers[key] = ppocr.TextRecognizer(OCR_DIR / spec.model, self._device, bidi=spec.bidi)
        return rec

    def installed_recognizers(self) -> list[str]:
        return [k for k, r in RECOGNIZERS.items() if (OCR_DIR / r.model).is_dir()]

    # --- reading ----------------------------------------------------------------------------------------------------
    def _read(self, image: np.ndarray, max_side: int, group: ScriptGroup | None, *, still: bool) -> Reading:
        """Detect, orient, recognize, merge and order the lines of a BGR image. `group` None = auto script."""
        assert self._detector is not None and self._textline is not None
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        detections = self._detector.detect(image, max_side)
        timings["det"] = (time.perf_counter() - t0) * 1000
        if not detections:
            return Reading([], [], Frame(w, h), timings=timings)

        t1 = time.perf_counter()
        quads = [d.quad for d in detections]
        raw = [ppocr.crop_quad(image, q) for q in quads]  # min-area rectangles, as they lie in the image
        rotations = [270 if c.shape[0] >= ppocr.TALL_RATIO * c.shape[1] else 0 for c in raw]
        flips = self._textline.classify([rotate_cw(c, r) for c, r in zip(raw, rotations, strict=True)])
        rotations = [(r + 180) % 360 if flip == 180 else r for r, (flip, _) in zip(rotations, flips, strict=True)]
        timings["orient"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        lines = self._recognize(raw, quads, rotations, group)
        timings["rec"] = (time.perf_counter() - t2) * 1000

        doc_angle = None
        angle = 0
        if still:
            t3 = time.perf_counter()
            doc_angle = self._doc.classify(image) if self._doc is not None else None
            angle = self._dominant_rotation(lines, doc_angle)
            redo = [i for i, line in enumerate(lines) if line.rotation != angle]
            if redo:  # lines that disagree with the still: read them turned the still's way too, keep the better
                again = self._recognize([raw[i] for i in redo], [quads[i] for i in redo], [angle] * len(redo), group)
                for i, line in zip(redo, again, strict=True):
                    if line.text.strip() and line.conf > lines[i].conf:
                        lines[i] = line
                # doc_ori is only a hint the recognizer has to confirm: it calls an upright Arabic sign 180° at 0.92
                angle = self._dominant_rotation(lines, None)
            timings["orientation"] = (time.perf_counter() - t3) * 1000

        lines = [line for line in lines if line.text.strip()]
        frame = Frame(w, h, angle)
        frame = Frame(w, h, angle, line_skew([frame.forward(line.quad) for line in lines]))
        lines = self._merge_pieces(lines, frame)
        boxes = [bbox(frame.forward(line.quad)) for line in lines]
        ordered = reading_order(boxes)
        return Reading(
            [lines[i] for i, _ in ordered], [block for _, block in ordered], frame, doc_angle, timings
        )  # fmt: skip

    def _recognize(
        self,
        raw: list[np.ndarray],
        quads: list[np.ndarray],
        rotations: list[int],
        group: ScriptGroup | None,
    ) -> list[Line]:
        crops = [rotate_cw(c, r) for c, r in zip(raw, rotations, strict=True)]
        keys = group.recognizers if group is not None else (PRIMARY,)
        best: list[tuple[ppocr.Reading, str] | None] = [None] * len(crops)
        batches = ppocr.RecBatches(crops, ppocr.torch_device(self._device))
        for key in keys:
            rec = self._recognizer(key)
            if rec is None:
                continue
            for i, reading in enumerate(rec.recognize(batches)):
                if best[i] is None or reading.conf > best[i][0].conf:  # type: ignore[index]
                    best[i] = (reading, key)
        if group is not None and PRIMARY not in keys:  # a chosen script: its lines in Latin/digits may read better
            rec = self._recognizer(PRIMARY)
            for i, reading in enumerate(rec.recognize(batches) if rec else []):
                current = best[i]
                if current is None or (
                    native_share(current[0].text, RECOGNIZERS[current[1]]) < NATIVE_MIN_SHARE
                    and reading.conf > current[0].conf
                ):
                    best[i] = (reading, PRIMARY)
        if group is None:
            self._script_fallback(crops, best)
        out = []
        for quad, rotation, found in zip(quads, rotations, best, strict=True):
            reading, key = found if found is not None else (ppocr.Reading("", 0.0, 0), PRIMARY)
            ordered = np.roll(quad, rotation // 90, axis=0)  # corners in the text's reading order
            out.append(Line(ordered, reading.text.strip(), reading.conf, rotation, key))
        return out

    def _script_fallback(self, crops: list[np.ndarray], best: list[tuple[ppocr.Reading, str] | None]) -> None:
        """Auto mode: re-read doubtful PP-OCRv6 lines with every installed script recognizer (in place)."""
        doubtful = [i for i, found in enumerate(best) if found is None or _doubtful(found[0], crops[i])]
        if not doubtful:
            return
        candidates: dict[int, list[tuple[ppocr.Reading, str]]] = {i: [] for i in doubtful}
        batches = ppocr.RecBatches([crops[i] for i in doubtful], ppocr.torch_device(self._device))
        for key in SCRIPT_RECOGNIZERS:
            rec = self._recognizer(key)
            if rec is None:
                continue
            for i, reading in zip(doubtful, rec.recognize(batches), strict=True):
                if reading.conf >= SCRIPT_MIN_CONF and native_share(reading.text, RECOGNIZERS[key]) >= NATIVE_MIN_SHARE:
                    candidates[i].append((reading, key))
        for i, found in candidates.items():
            if not found:
                continue
            reading, key = max(found, key=lambda c: c[0].conf)
            current = best[i]
            if current is None:
                best[i] = (reading, key)
                continue
            primary = current[0]
            short = _too_short(primary, crops[i])
            lookalike = all(ch in LOOKALIKES for ch in reading.text if ch.isalpha())
            if lookalike and not short and primary.conf >= SCRIPT_MIN_CONF:
                continue  # "ΕΧΙΤ" for a fairly confident "EXIT"
            if short or reading.conf >= primary.conf - SCRIPT_MARGIN:
                best[i] = (reading, key)

    @staticmethod
    def _dominant_rotation(lines: list[Line], doc_angle: tuple[int, float] | None) -> int:
        """The still's text orientation (clockwise degrees to upright): doc_ori's if sure, else the lines' vote
        (weighted by characters x confidence)."""
        if doc_angle is not None and doc_angle[1] >= DOC_ORI_MIN:
            return (360 - doc_angle[0]) % 360
        votes: Counter[int] = Counter()
        for line in lines:
            votes[line.rotation] += len(line.text.strip()) * line.conf
        return votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1] > 0 else 0

    @staticmethod
    def _merge_pieces(lines: list[Line], frame: Frame) -> list[Line]:
        """Pieces of one row → one line (`text_layout.merge_rows`), joined in reading order (right to left for
        Arabic-script text)."""
        if len(lines) < 2:
            return lines
        boxes = [bbox(frame.forward(line.quad)) for line in lines]
        out = []
        for group in merge_rows(boxes):
            if len(group) == 1:
                out.append(lines[group[0]])
                continue
            pieces = [lines[i] for i in group]
            if sum(_is_rtl(p.text) for p in pieces) * 2 > len(pieces):
                text = " ".join(p.text for p in reversed(pieces))
            else:
                text = _join_words([p.text for p in pieces])
            weights = [max(len(p.text), 1) for p in pieces]
            conf = sum(p.conf * n for p, n in zip(pieces, weights, strict=True)) / sum(weights)
            first, last = pieces[0], pieces[-1]
            quad = np.array([first.quad[0], last.quad[1], last.quad[2], first.quad[3]], np.float32)
            recognizer = Counter(p.recognizer for p in pieces).most_common(1)[0][0]
            out.append(Line(quad, text, conf, first.rotation, recognizer, sum(p.pieces for p in pieces)))
        return out

    # --- live -------------------------------------------------------------------------------------------------------
    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        if ctx.options.get("live", LIVE_READ) == LIVE_MARK:
            return self._mark(image)
        selected = LEGACY_LANGUAGES.get(ctx.options["languages"], ctx.options["languages"])
        group = None if selected == AUTO_SCRIPTS else GROUPS[selected]
        reading = self._read(image, MAX_SIDE, group, still=False)
        min_conf = float(ctx.options["min_conf"])
        out = [
            Annotation(box=line.box, conf=float(line.conf), label=line.text)
            for line in reading.lines
            if line.conf >= min_conf
        ]
        target = ctx.options.get("translate_to", TRANSLATE_OFF)
        if out and target != TRANSLATE_OFF:
            sources = (group or group_for_text([a.label for a in out])).sources
            self._translate(out, sources, ctx.options, LANGUAGES[target])
        return out

    def _mark(self, image: np.ndarray) -> list[Annotation]:
        """ "Mark text areas": detector only, conf 1. The medium detector: on 960 px frames (3090) it costs ~2–5 ms more
        than PP-OCRv6 small/tiny (median 31 vs 30 / 26 ms per run on a busy host) and finds more lines on dense
        scenes (menu 30 vs 22 / 15)."""
        assert self._detector is not None
        out = []
        for d in self._detector.detect(image, MAX_SIDE):
            x1, y1, x2, y2 = bbox(d.quad)
            if x2 > x1 and y2 > y1:
                out.append(Annotation(box=(x1, y1, x2, y2), conf=1.0, label=MARK_LABEL))
        return out

    # --- still reading (POST /text/read) ----------------------------------------------------------------------------
    def read_still(
        self,
        image: np.ndarray,
        languages: str = AUTO_SCRIPTS,
        translate_to: Sequence[str] = (),
        translate_from: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        format_blocks: bool = False,
    ) -> dict[str, Any]:
        """Read one BGR still (detection at up to STILL_MAX_SIDE, recognition at full resolution) and translate it into
        every `translate_to` code: the `POST /text/read` response (docs/PROTOCOL.md "Read text")."""
        t0 = time.perf_counter()
        selected = LEGACY_LANGUAGES.get(languages, languages)
        chosen = None if selected == AUTO_SCRIPTS else GROUPS[selected]
        reading = self._read(image, STILL_MAX_SIDE, chosen, still=True)
        texts = [line.text for line in reading.lines]
        group = chosen or group_for_text(texts)
        h, w = image.shape[:2]
        ocr_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        translations: dict[str, list[str | None]] = {}
        errors: dict[str, str] = {}
        langs: list[str | None] = [None] * len(texts)
        if texts:
            translator = self._get_translator()
            if translate_from is not None:
                langs = [translate_from if is_translatable(normalize(t)) else None for t in texts]
            else:  # a weakly detected line borrows its block's language first (one sign), then the whole still's
                langs = translator.detect_sources(texts, group.sources, blocks=reading.blocks)
            if not format_blocks:
                known = [i for i, lang in enumerate(langs) if lang is not None]  # the rest stays as read
                for target in dict.fromkeys(translate_to):
                    try:
                        results = translator.translate(
                            [texts[i] for i in known],
                            target,
                            group.sources,
                            source=[langs[i] for i in known],
                            model=model,
                        )
                    except Exception as exc:
                        log.exception("text: still: translation to %s failed", target)
                        translator.unload_model()
                        errors[target] = str(exc) or type(exc).__name__
                        continue
                    row: list[str | None] = [None] * len(texts)
                    for i, result in zip(known, results, strict=True):
                        row[i] = result.text if result is not None else None
                    translations[target] = row
        else:
            translations = {target: [] for target in translate_to}
        lines = []
        for index, (line, block, lang) in enumerate(zip(reading.lines, reading.blocks, langs, strict=True)):
            entry = {
                "text": line.text,
                "conf": round(line.conf, 3),
                "box": _normalized_box(line.box, w, h),
                "block": block,
                "lang": lang,
            }
            if format_blocks:
                entry |= {
                    "quad": _normalized_quad(line.quad, w, h),
                    "script": group_for_text([line.text]).key,
                    "suspect": _suspect(line.text, line.conf),
                }
            lines.append(entry)
        response = {
            "w": w,
            "h": h,
            "script": group.key,
            "ms": {"ocr": round(ocr_ms, 1), "translate": round((time.perf_counter() - t1) * 1000, 1)},
            "lines": lines,
            "translations": translations,
            "errors": errors,
        }
        if format_blocks:
            blocks = _block_payload(reading, langs, w, h)
            for target in dict.fromkeys(translate_to):
                try:
                    values = self.translate_blocks(blocks, target, model)
                except Exception as exc:
                    log.exception("text: still: block translation to %s failed", target)
                    self._get_translator().unload_model()
                    errors[target] = str(exc) or type(exc).__name__
                    continue
                for block, translated in zip(blocks, values, strict=True):
                    block["translations"][target] = translated
            response["angle"] = round(reading.frame.angle + reading.frame.skew, 2)
            response["blocks"] = blocks
            response["translations"] = {}
        return response

    def translate_lines(
        self, lines: Sequence[str], to: str, sources: Sequence[str | None] | None = None, model: str = DEFAULT_MODEL
    ) -> list[str | None]:
        """`POST /text/translate`: lines read earlier; `sources` per line (None = detect, among the languages of the
        script the lines are written in)."""
        if not lines:
            return []
        translator = self._get_translator()
        try:
            results = translator.translate(lines, to, group_for_text(lines).sources, source=sources, model=model)
        except Exception:
            translator.unload_model()  # free a half-loaded / failing model; the next call loads it again
            raise
        return [r.text if r is not None else None for r in results]

    def translate_blocks(
        self, blocks: Sequence[dict[str, Any]], to: str, model: str = DEFAULT_MODEL
    ) -> list[str | None]:
        """Translate prose as a whole and list-like blocks one line at a time; preserve untranslated list entries."""
        if not blocks:
            return []
        translator = self._get_translator()
        chunks: list[list[str]] = []
        sources: list[str | None] = []
        for block in blocks:
            text = str(block["text"])
            parts = text.splitlines() or [text]
            selected = parts if block["kind"] == "list" else [text]
            chunks.append(selected)
            sources.extend([block.get("lang")] * len(selected))
        flattened = [text for chunk in chunks for text in chunk]
        try:
            results = translator.translate(
                flattened, to, group_for_text(flattened).sources, source=sources, model=model
            )
        except Exception:
            translator.unload_model()
            raise
        out: list[str | None] = []
        offset = 0
        for block, chunk in zip(blocks, chunks, strict=True):
            translated = results[offset : offset + len(chunk)]
            offset += len(chunk)
            values = [result.text if result is not None else None for result in translated]
            if all(value is None for value in values):
                out.append(None)
            elif block["kind"] == "list":
                out.append(
                    "\n".join(
                        value if value is not None else original for value, original in zip(values, chunk, strict=True)
                    )
                )
            else:
                out.append(values[0])
        return out

    def _get_translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(self._device)
        return self._translator

    def _translate(
        self, anns: list[Annotation], sources: SourceLanguages, options: dict[str, Any], target: str
    ) -> None:
        """Label = translation, detail = "<SOURCE>: <original>" (show_original). Failures leave the text as read."""
        if self._translate_error and time.monotonic() - self._translate_failed_at < TRANSLATE_RETRY_S:
            for a in anns:
                a.detail = self._translate_error
            return
        translator = self._get_translator()
        source = options.get("translate_from", AUTO_DETECT)
        try:
            results = translator.translate(
                [a.label for a in anns],
                target,
                sources,
                source=None if source == AUTO_DETECT else LANGUAGES[source],
                model=options.get("translation_model", DEFAULT_MODEL),
            )
        except Exception as exc:
            log.exception("text: translation failed; showing the text untranslated for %.0f s", TRANSLATE_RETRY_S)
            translator.unload_model()
            self._translate_failed_at = time.monotonic()
            self._translate_error = f"translation unavailable: {exc}"
            for a in anns:
                a.detail = self._translate_error
            return
        self._translate_error = ""
        show_original = bool(options.get("show_original", True))
        for a, result in zip(anns, results, strict=True):
            if result is not None:
                a.detail = f"{result.source.upper()}: {a.label}" if show_original else None
                a.label = result.text


def _too_short(reading: ppocr.Reading, crop: np.ndarray) -> bool:
    """Far fewer characters than a line this wide holds: PP-OCRv6 reads letters outside its alphabet as blanks."""
    h, w = crop.shape[:2]
    return len(reading.text.replace(" ", "")) < MIN_CHARS_PER_ASPECT * w / max(h, 1)


def _doubtful(reading: ppocr.Reading, crop: np.ndarray) -> bool:
    return reading.conf < DOUBTFUL_CONF or _too_short(reading, crop)


def _is_rtl(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(unicodedata.bidirectional(ch) in ("R", "AL") for ch in letters) * 2 > len(letters)


def _join_words(texts: list[str]) -> str:
    out = texts[0]
    for text in texts[1:]:
        cjk = out and text and _is_cjk(out[-1]) and _is_cjk(text[0])
        out += text if cjk else " " + text
    return out


def _is_cjk(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "FULLWIDTH"))


def _normalized_box(box: tuple[float, float, float, float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(min(max(x1 / w, 0.0), 1.0), 4),
        round(min(max(y1 / h, 0.0), 1.0), 4),
        round(min(max(x2 / w, 0.0), 1.0), 4),
        round(min(max(y2 / h, 0.0), 1.0), 4),
    ]


def _normalized_quad(points: np.ndarray, w: int, h: int) -> list[list[float]]:
    return [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in points]


def _suspect(text: str, confidence: float) -> bool:
    """Conservative OCR-gibberish warning: low recognizer confidence or a long, mixed-script token."""
    if confidence < 0.55:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 4:
        return False
    scripts = Counter(unicodedata.name(ch, "COMMON").split(" ", 1)[0] for ch in letters)
    return len(scripts) > 1 and max(scripts.values()) / len(letters) < 0.7


def _block_payload(reading: Reading, languages: list[str | None], w: int, h: int) -> list[dict[str, Any]]:
    by_id: dict[int, list[int]] = {}
    for index, block_id in enumerate(reading.blocks):
        by_id.setdefault(block_id, []).append(index)
    out = []
    for block_id, indices in by_id.items():
        lines = [reading.lines[index] for index in indices]
        texts = [line.text for line in lines]
        x1 = min(line.box[0] for line in lines)
        y1 = min(line.box[1] for line in lines)
        x2 = max(line.box[2] for line in lines)
        y2 = max(line.box[3] for line in lines)
        points = np.concatenate([line.quad for line in lines]).astype(np.float32)
        corners = cv2.boxPoints(cv2.minAreaRect(points))
        centre = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
        corners = corners[np.argsort(angles)]
        start = int(np.argmin(corners.sum(axis=1)))
        corners = np.roll(corners, -start, axis=0)
        block_languages = [languages[index] for index in indices if languages[index] is not None]
        language = Counter(block_languages).most_common(1)[0][0] if block_languages else None
        out.append(
            {
                "id": block_id,
                "lines": indices,
                "box": _normalized_box((x1, y1, x2, y2), w, h),
                "quad": _normalized_quad(corners, w, h),
                "text": "\n".join(texts),
                "kind": block_kind(texts, [line.box[2] - line.box[0] for line in lines]),
                "lang": language,
                "suspect": any(_suspect(line.text, line.conf) for line in lines),
                "translations": {},
            }
        )
    return out
