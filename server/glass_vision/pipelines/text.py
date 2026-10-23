"""Text (OCR): EasyOCR (CRAFT detector + CRNN recognizer) reading signs, labels and screens, optionally translated.

Slow pipeline: the framework runs it in the background on the newest frame (`interval_ms`).

Scripts: EasyOCR has one recognition model per script, and a `Reader` may only combine languages that share that
model (plus English) — `easyocr/easyocr.py` `Reader.__init__` picks the model from the language list and raises
"... is only compatible with English" otherwise; the lists are `easyocr/config.py` (`latin_lang_list`,
`cyrillic_lang_list`, …). So the user picks a *script group*; each group's Reader reads every EasyOCR language of
that script. Readers are cached per group and share one CRAFT detector; the default group is warmed in `load`.

Live modes (option `live`, docs/PROTOCOL.md "Read text"): "Read live" (recognize every run, as before) or "Mark text
areas" (CRAFT detector only, label "text"): the app then reads a high-resolution still once (`read_still`, served by
`glass_vision/text_read.py`).

Translation (`glass_vision/translate.py`, M2M100) is off by default and loaded on first use.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ..translate import DEFAULT_MODEL, LANGUAGES, MODELS, SourceLanguages, Translator, is_translatable, normalize
from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, RunContext
from .text_layout import reading_order

log = logging.getLogger(__name__)

EASYOCR_DIR = MODELS_DIR / "easyocr"
MAX_SIDE = 1280  # downscale before OCR; boxes are scaled back to the original pixels
# readtext tuning (measured on the 3090): batch_size=16 recognizes all crops in one pass instead of one call per
# word (960x720 sign 67 → 55 ms, 810x1080 street photo 130 → 92 ms). canvas_size only caps the detector input,
# which MAX_SIDE already does; cudnn.benchmark was not a clear win and is process-global, so it stays off.
# readtext = detect + recognize; they're called separately so one detection serves every recognizer tried (auto
# mode) and "Mark text areas" skips recognition. canvas_size is passed per call (MAX_SIDE or STILL_MAX_SIDE).
RECOGNIZE_KWARGS: dict[str, Any] = {"batch_size": 16, "paragraph": False, "reformat": False}
STILL_MAX_SIDE = 2560  # `read_still`: a still is read once, so it gets twice the live resolution
# easyocr Reader.detect defaults (EasyOCR 1.7.2), for `_detect`/`_craft`, which reimplement it.
DETBOX_KWARGS: dict[str, Any] = {"text_threshold": 0.7, "link_threshold": 0.4, "low_text": 0.4}
GROUP_KWARGS: dict[str, Any] = {
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
}
MIN_BOX_SIDE = 20
CRAFT_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
CRAFT_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)
TRANSLATE_RETRY_S = 60.0  # after a translator failure (e.g. model not downloaded), OCR goes on untranslated


@dataclass(frozen=True)
class ScriptGroup:
    key: str  # scripts/download_models.py --ocr-scripts name
    label: str  # option value, shown verbatim in the app
    languages: tuple[str, ...]  # EasyOCR language codes: one Reader
    model_file: str  # recognition weights EasyOCR picks for them (easyocr/config.py), in EASYOCR_DIR
    sources: SourceLanguages  # translation: source languages detection may pick (M2M100 codes)


def _non_latin(*candidates: str) -> SourceLanguages:
    return SourceLanguages((*candidates, "en"), ascii_is_english=True)


# fmt: off
LATIN_LANGUAGES = (  # easyocr/config.py latin_lang_list (EasyOCR 1.7.2)
    "en", "pl", "de", "fr", "es", "it", "pt", "nl", "cs", "sk", "sv", "da", "no", "hu", "ro", "tr", "hr", "sl", "bs",
    "sq", "et", "lt", "lv", "is", "ga", "cy", "mt", "af", "az", "id", "ms", "tl", "sw", "vi", "uz", "ku", "la", "mi",
    "oc", "pi", "rs_latin",
)
SCRIPT_GROUPS: tuple[ScriptGroup, ...] = (
    ScriptGroup("latin", "Latin — EN, PL, DE, FR, ES, IT, PT, NL, CS, SV, TR… (41 languages)", LATIN_LANGUAGES,
                "latin_g2.pth",
                # Every Latin-script language both lingua and M2M100 know, among EasyOCR's + Finnish (its letters
                # are covered); the first is the fallback.
                SourceLanguages(("en", "pl", "de", "fr", "es", "it", "pt", "nl", "cs", "sk", "sv", "da", "no", "fi",
                                 "hu", "ro", "tr", "hr", "sl", "bs", "sq", "et", "lt", "lv", "is", "ga", "cy", "af",
                                 "az", "id", "ms", "tl", "sw", "vi"))),
    ScriptGroup("cyrillic", "Cyrillic — RU, UK, BE, BG, SR, MN… + EN",
                ("ru", "uk", "be", "bg", "rs_cyrillic", "mn", "abq", "ady", "kbd", "ava", "dar", "inh", "che", "lbe",
                 "lez", "tab", "tjk", "en"),
                "cyrillic_g2.pth", _non_latin("ru", "uk", "be", "bg", "sr", "mn")),
    ScriptGroup("zh_sim", "Chinese (simplified) + EN", ("ch_sim", "en"), "zh_sim_g2.pth", _non_latin("zh")),
    ScriptGroup("zh_tra", "Chinese (traditional) + EN", ("ch_tra", "en"), "chinese.pth", _non_latin("zh")),
    ScriptGroup("japanese", "Japanese + EN", ("ja", "en"), "japanese_g2.pth", _non_latin("ja")),
    ScriptGroup("korean", "Korean + EN", ("ko", "en"), "korean_g2.pth", _non_latin("ko")),
    ScriptGroup("arabic", "Arabic script — AR, FA, UR, UG + EN", ("ar", "fa", "ur", "ug", "en"), "arabic.pth",
                _non_latin("ar", "fa", "ur")),
    ScriptGroup("devanagari", "Devanagari — HI, MR, NE… + EN",
                ("hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "en"),  # sa, bgc: see below
                "devanagari.pth", _non_latin("hi", "mr", "ne")),
    # EasyOCR 1.7.2 lists sa, bgc (Devanagari) and mni (Bengali) but ships no character files for them: a Reader
    # with them fails with FileNotFoundError (easyocr/character/<code>_char.txt). Tamil is left out entirely: its
    # published tamil.pth has 143 output classes, its character list 127 (load_state_dict size mismatch).
    ScriptGroup("bengali", "Bengali — BN, AS + EN", ("bn", "as", "en"), "bengali.pth", _non_latin("bn")),
    ScriptGroup("thai", "Thai + EN", ("th", "en"), "thai.pth", _non_latin("th")),
    ScriptGroup("telugu", "Telugu + EN", ("te", "en"), "telugu.pth", _non_latin()),  # M2M100 has no Telugu
    ScriptGroup("kannada", "Kannada + EN", ("kn", "en"), "kannada.pth", _non_latin("kn")),
)
# fmt: on
GROUPS: dict[str, ScriptGroup] = {g.label: g for g in SCRIPT_GROUPS}
DEFAULT_GROUP = SCRIPT_GROUPS[0]
# Values older app/server versions saved (`languages` was "en" / "en+pl"): both are Latin script.
LEGACY_LANGUAGES = {"en": DEFAULT_GROUP.label, "en+pl": DEFAULT_GROUP.label}

TRANSLATE_OFF = "Off"
AUTO_DETECT = "Auto-detect"
AUTO_SCRIPTS = "Auto-detect (available scripts)"
AUTO_REPROBE_S = 5.0
NATIVE_MIN_SHARE = 0.3  # `_auto_score`
CHINESE_GROUPS = frozenset({"zh_sim", "zh_tra"})  # `_best_reading`
AUTO_RETRY_S = 5.0
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


def _auto_score(annotations: list[Annotation], group: ScriptGroup) -> float:
    """Prefer high-confidence text; count digit-only lines so numeric signs can select a recognizer too.

    A non-Latin group scores 0 unless at least NATIVE_MIN_SHARE of the letters it read are its own script: every one of
    them also reads English, and on bus.jpg (Latin script) Korean's and Cyrillic's recognizers read "CerO emisiones"
    more confidently than Latin's, plus a hallucinated "Ш" or "내" on the bus graphics, so auto picked them. Real
    Cyrillic/CJK signs read >90 % native letters; those hallucinations were < 5 %."""
    if group.key != DEFAULT_GROUP.key:
        letters = [ch for a in annotations for ch in a.label if ch.isalpha()]
        if not letters or sum(_is_native(ch, group) for ch in letters) < NATIVE_MIN_SHARE * len(letters):
            return 0.0
    return sum(a.conf * min(max(sum(ch.isalpha() for ch in a.label), 1), 40) for a in annotations)


def _is_native(ch: str, group: ScriptGroup) -> bool:
    return ch.isalpha() and unicodedata.name(ch, "").startswith(_NATIVE_PREFIXES.get(group.key, ()))


def _check_cudnn_usable() -> None:
    """torch's Python cuDNN init refuses (process-wide) when *any* visible GPU is below SM 7.5, which breaks the
    recognizer's LSTM (`flatten_parameters`). Fail with a fix instead of a cryptic cuDNN error deep in EasyOCR."""
    old = [
        f"cuda:{i} {torch.cuda.get_device_name(i)}"
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(i) < (7, 5)
    ]
    if old:
        raise RuntimeError(
            f"OCR needs cuDNN, which torch disables while a GPU below SM 7.5 is visible ({', '.join(old)}); "
            "hide it with CUDA_VISIBLE_DEVICES (set before CUDA initializes)"
        )


def _available_groups() -> list[ScriptGroup]:
    """Script groups auto mode may try: Latin (always installed) + those whose recognizer weights are downloaded (auto
    never triggers a download)."""
    return [g for g in SCRIPT_GROUPS if g.key == DEFAULT_GROUP.key or (EASYOCR_DIR / g.model_file).is_file()]


# Unicode character name prefix → script group whose source languages detection picks from (`group_for_text`).
_SCRIPT_PREFIXES = {
    "CYRILLIC": "cyrillic",
    "CJK": "zh_sim",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "HANGUL": "korean",
    "ARABIC": "arabic",
    "DEVANAGARI": "devanagari",
    "BENGALI": "bengali",
    "THAI": "thai",
    "TELUGU": "telugu",
    "KANNADA": "kannada",
}
_GROUP_BY_KEY = {g.key: g for g in SCRIPT_GROUPS}
_NATIVE_PREFIXES: dict[str, tuple[str, ...]] = {"zh_tra": ("CJK",), "japanese": ("HIRAGANA", "KATAKANA", "CJK")}
for _prefix, _key in _SCRIPT_PREFIXES.items():
    _NATIVE_PREFIXES.setdefault(_key, (_prefix,))


def group_for_text(lines: Sequence[str]) -> ScriptGroup:
    """The script group text was most likely read with, from its letters' Unicode names (Latin when unclear). Japanese
    wins as soon as kana appear (kanji alone count as Chinese)."""
    counts: Counter[str] = Counter()
    for ch in "".join(lines):
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            counts[next((key for prefix, key in _SCRIPT_PREFIXES.items() if name.startswith(prefix)), "latin")] += 1
    if counts["japanese"]:
        return _GROUP_BY_KEY["japanese"]
    non_latin = [(n, key) for key, n in counts.items() if key != "latin"]
    return _GROUP_BY_KEY[max(non_latin)[1]] if non_latin else DEFAULT_GROUP


@dataclass(frozen=True)
class _Detected:
    """One CRAFT pass: EasyOCR's line boxes on the downscaled image + what maps them back to the original pixels."""

    grey: np.ndarray  # downscaled grey image the recognizers read the boxes from
    horizontal: list[Any]  # [x_min, x_max, y_min, y_max]
    free: list[Any]  # 4-point quads (rotated text)
    scale: float  # downscaled / original
    w: int  # original size
    h: int


class TextPipeline(Pipeline):
    info = PipelineInfo(
        id="text",
        name="Text (OCR)",
        description="Reads text on signs, labels and screens (80+ languages); choose a script or auto-detect installed scripts.",
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
        self._device: str | bool = False
        self._readers: dict[str, Any] = {}  # ScriptGroup.key → easyocr.Reader
        self._translator: Translator | None = None
        self._translate_failed_at = 0.0
        self._translate_error = ""
        self._auto_group_key: str | None = None
        self._auto_probe_after = 0.0

    def load(self, device: str) -> None:
        if device.startswith("cuda"):
            try:
                _check_cudnn_usable()
            except RuntimeError as exc:
                log.warning("text: %s — running OCR on the CPU instead (slow)", exc)
                device = "cpu"
        self._device = device if device.startswith("cuda") else False
        reader = self._reader(DEFAULT_GROUP)
        reader.readtext(np.full((64, 256, 3), 255, np.uint8), batch_size=16, canvas_size=MAX_SIDE)  # warm up CUDA

    def unload(self) -> None:
        self._readers.clear()
        if self._translator is not None:
            self._translator.unload()
            self._translator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reader(self, group: ScriptGroup) -> Any:
        reader = self._readers.get(group.key)
        if reader is None:
            import easyocr  # heavy import (torch, scipy…): only when the pipeline is actually used

            EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
            shared = next(iter(self._readers.values()), None)  # every group uses the same CRAFT detector
            log.info("text: building EasyOCR reader %s on %s", group.key, self._device or "cpu")
            reader = easyocr.Reader(
                list(group.languages),
                gpu=self._device,
                model_storage_directory=str(EASYOCR_DIR),
                user_network_directory=str(EASYOCR_DIR / "user_network"),
                download_enabled=True,
                detector=shared is None,
                verbose=False,
            )
            if shared is not None:
                reader.detector = shared.detector
                reader.get_textbox = shared.get_textbox
                reader.detect_network = shared.detect_network
            # EasyOCR wraps both nets in DataParallel, which scatters batches over *every* visible GPU; unwrap so
            # they stay on the requested device.
            for attr in ("detector", "recognizer"):
                net = getattr(reader, attr)
                if isinstance(net, torch.nn.DataParallel):
                    setattr(reader, attr, net.module)
            self._readers[group.key] = reader
        return reader

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        if ctx.options.get("live", LIVE_READ) == LIVE_MARK:
            return self._mark(image)
        selected = LEGACY_LANGUAGES.get(ctx.options["languages"], ctx.options["languages"])
        auto = selected == AUTO_SCRIPTS
        available = _available_groups()
        detected = self._detect(image, MAX_SIDE)  # one detection for every recognizer tried below
        now = time.monotonic()
        cached = next((g for g in available if g.key == self._auto_group_key), None) if auto else None
        probing = auto and now >= self._auto_probe_after
        if not auto:
            candidates = [GROUPS[selected]]
        elif probing:
            candidates = available
        else:
            # Between probes: the script found last time, or Latin while nothing has been found yet (a camera that
            # starts on a wall must still read the first sign it sees, not wait for the next probe).
            candidates = [cached or DEFAULT_GROUP]

        min_conf = float(ctx.options["min_conf"])
        found, best, best_score = self._best_reading(candidates, detected, min_conf, auto=auto)
        best_group = found or (candidates[0] if candidates else DEFAULT_GROUP)

        # A cached recognizer that has gone blank likely saw another script. Probe the alternatives immediately.
        fallback_probe = auto and cached is not None and not probing and best_score <= 0
        if fallback_probe:
            others = [g for g in available if g is not cached]
            found, annotations, score = self._best_reading(others, detected, min_conf, auto=True)
            if found is not None and score > best_score:
                best_group, best, best_score = found, annotations, score

        if auto:
            if best_score > 0:
                self._auto_group_key = best_group.key
                if probing or best_group is not cached:
                    self._auto_probe_after = now + AUTO_REPROBE_S
            else:
                self._auto_group_key = None
                # Only a probe that actually ran resets the retry clock; pushing it on every blank Latin run would
                # postpone the next probe forever (OCR went permanently blank after one textless frame).
                if probing or fallback_probe:
                    self._auto_probe_after = now + AUTO_RETRY_S
        if auto and not best:
            # Latin is always installed by the normal server setup; preserve its clear empty result if every
            # optional recognizer failed to initialize.
            best_group = DEFAULT_GROUP
        out = best
        target = ctx.options.get("translate_to", TRANSLATE_OFF)
        if out and target != TRANSLATE_OFF:
            self._translate(out, best_group, ctx.options, LANGUAGES[target])
        return out

    def _detect(self, image: np.ndarray, max_side: int) -> _Detected:
        """CRAFT on the image downscaled to `max_side`: text line boxes (EasyOCR merges words into lines)."""
        from easyocr.utils import diff, group_text_box, reformat_input

        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        small = (
            image
            if scale == 1.0
            else cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        )
        # EasyOCR treats ndarrays as RGB; reformat_input derives the grey image exactly as readtext does.
        rgb, grey = reformat_input(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        # Grouping into lines + size filter as in easyocr Reader.detect (its defaults; min_size 20).
        horizontal, free = group_text_box(self._craft(rgb), **GROUP_KWARGS)
        horizontal = [b for b in horizontal if max(b[1] - b[0], b[3] - b[2]) > MIN_BOX_SIDE]
        free = [q for q in free if max(diff([c[0] for c in q]), diff([c[1] for c in q])) > MIN_BOX_SIDE]
        return _Detected(grey, horizontal, free, scale, w, h)

    def _craft(self, rgb: np.ndarray) -> list[np.ndarray]:
        """easyocr.detection.get_textbox (CRAFT, `canvas_size` >= the image) with two speed-ups, measured on the 3090
        on a 1080x810 photo: normalization on the GPU instead of numpy (20 → ~1 ms) and the forward pass in fp16
        (33 → 21 ms; score maps differ by < 0.002). Returns EasyOCR's flat int32 4-point boxes."""
        from easyocr.craft_utils import adjustResultCoordinates, getDetBoxes

        net = self._reader(DEFAULT_GROUP).detector
        device = next(net.parameters()).device
        h, w = rgb.shape[:2]
        canvas = np.zeros((-(-h // 32) * 32, -(-w // 32) * 32, 3), np.uint8)  # padded to /32 like resize_aspect_ratio
        canvas[:h, :w] = rgb
        x = torch.from_numpy(canvas).to(device).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor(CRAFT_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CRAFT_STD, device=device).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
            y, _ = net((x - mean) / std)
        maps = y[0].float().cpu().numpy()
        boxes, _, _ = getDetBoxes(maps[:, :, 0], maps[:, :, 1], **DETBOX_KWARGS)
        return [np.asarray(b).astype(np.int32).reshape(-1) for b in adjustResultCoordinates(boxes, 1, 1)]

    def _pixel_box(self, quad: Any, d: _Detected) -> tuple[float, float, float, float] | None:
        pts = np.asarray(quad, dtype=np.float64) / d.scale
        x1, y1 = np.clip(pts.min(axis=0), 0, (d.w, d.h))
        x2, y2 = np.clip(pts.max(axis=0), 0, (d.w, d.h))
        if x2 <= x1 or y2 <= y1:
            return None
        return float(x1), float(y1), float(x2), float(y2)

    def _recognize(self, group: ScriptGroup, d: _Detected, min_conf: float) -> list[Annotation]:
        if not d.horizontal and not d.free:
            return []
        reader = self._reader(group)
        out: list[Annotation] = []
        for quad, text, conf in reader.recognize(d.grey, d.horizontal, d.free, **RECOGNIZE_KWARGS):
            label = text.strip()
            if not label or conf < min_conf:
                continue
            box = self._pixel_box(quad, d)
            if box is not None:
                out.append(Annotation(box=box, conf=float(conf), label=label))
        return out

    def _best_reading(
        self, groups: Sequence[ScriptGroup], detected: _Detected, min_conf: float, *, auto: bool
    ) -> tuple[ScriptGroup | None, list[Annotation], float]:
        """Recognize the detected lines with each group; the best `_auto_score` wins (ties: the earlier group). Prefer
        recognitions that return more high-confidence letters, while capping each line so a single hallucinated
        paragraph cannot swamp several valid short signs. `auto`: a group that fails is logged and skipped (else it
        raises), and the Chinese groups run after Japanese and only if its recognizer (which knows kanji) saw any CJK:
        their 6.7k-class decoding costs 0.3-2 s per call on the CPU (3090 box), every other group 20-100 ms."""
        if auto:
            groups = sorted(groups, key=lambda g: g.key in CHINESE_GROUPS)  # stable: only moves them last
        best: tuple[ScriptGroup | None, list[Annotation], float] = (None, [], -1.0)
        cjk_seen: bool | None = None  # None = Japanese not tried (not installed): try Chinese anyway
        for group in groups:
            if auto and group.key in CHINESE_GROUPS and cjk_seen is False:
                continue
            try:
                annotations = self._recognize(group, detected, min_conf)
            except Exception:
                if not auto:
                    raise
                log.exception("text: auto-detect could not run %s; trying the remaining installed scripts", group.key)
                continue
            if group.key == "japanese":
                cjk_seen = any(_is_native(ch, group) for a in annotations for ch in a.label)
            score = _auto_score(annotations, group)
            if score > best[2]:
                best = (group, annotations, score)
        return best

    def _mark(self, image: np.ndarray) -> list[Annotation]:
        """ "Mark text areas": detector only. CRAFT gives no per-box score, so conf is 1."""
        d = self._detect(image, MAX_SIDE)
        quads = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for x1, x2, y1, y2 in d.horizontal] + list(d.free)
        boxes = (self._pixel_box(q, d) for q in quads)
        return [Annotation(box=b, conf=1.0, label=MARK_LABEL) for b in boxes if b is not None]

    # --- still reading (POST /text/read) ---------------------------------------------------------------------
    def read_still(
        self,
        image: np.ndarray,
        languages: str = AUTO_SCRIPTS,
        translate_to: Sequence[str] = (),
        translate_from: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> dict[str, Any]:
        """Read one upright BGR still at up to STILL_MAX_SIDE and translate it into every `translate_to` code: the
        `POST /text/read` response (docs/PROTOCOL.md "Read text"). Doesn't touch the live auto-detect state."""
        t0 = time.perf_counter()
        selected = LEGACY_LANGUAGES.get(languages, languages)
        auto = selected == AUTO_SCRIPTS
        candidates = _available_groups() if auto else [GROUPS[selected]]
        min_conf = float(next(o.default for o in self.info.options if o.key == "min_conf"))
        detected = self._detect(image, STILL_MAX_SIDE)
        found, anns, _ = self._best_reading(candidates, detected, min_conf, auto=auto)
        group = found if found is not None and anns else DEFAULT_GROUP if auto else candidates[0]
        h, w = image.shape[:2]
        ordered = reading_order([a.box for a in anns])
        anns = [anns[i] for i, _ in ordered]
        texts = [a.label for a in anns]
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
                langs = translator.detect_sources(texts, group.sources, blocks=[block for _, block in ordered])
            known = [i for i, lang in enumerate(langs) if lang is not None]  # the rest stays as read
            for target in dict.fromkeys(translate_to):
                try:
                    results = translator.translate(
                        [texts[i] for i in known], target, group.sources, source=[langs[i] for i in known], model=model
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
        lines = [
            {
                "text": a.label,
                "conf": round(a.conf, 3),
                "box": [round(a.box[0] / w, 4), round(a.box[1] / h, 4), round(a.box[2] / w, 4), round(a.box[3] / h, 4)],
                "block": block,
                "lang": lang,
            }
            for a, (_, block), lang in zip(anns, ordered, langs, strict=True)
        ]
        return {
            "w": w,
            "h": h,
            "script": group.key,
            "ms": {"ocr": round(ocr_ms, 1), "translate": round((time.perf_counter() - t1) * 1000, 1)},
            "lines": lines,
            "translations": translations,
            "errors": errors,
        }

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

    def _get_translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(self._device if isinstance(self._device, str) else "cpu")
        return self._translator

    def _translate(self, anns: list[Annotation], group: ScriptGroup, options: dict[str, Any], target: str) -> None:
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
                group.sources,
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
