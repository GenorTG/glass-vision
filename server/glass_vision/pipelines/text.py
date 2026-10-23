"""Text (OCR): EasyOCR (CRAFT detector + CRNN recognizer) reading signs, labels and screens, optionally translated.

Slow pipeline: the framework runs it in the background on the newest frame (`interval_ms`).

Scripts: EasyOCR has one recognition model per script, and a `Reader` may only combine languages that share that
model (plus English) — `easyocr/easyocr.py` `Reader.__init__` picks the model from the language list and raises
"... is only compatible with English" otherwise; the lists are `easyocr/config.py` (`latin_lang_list`,
`cyrillic_lang_list`, …). So the user picks a *script group*; each group's Reader reads every EasyOCR language of
that script. Readers are cached per group and share one CRAFT detector; the default group is warmed in `load`.

Translation (`glass_vision/translate.py`, M2M100) is off by default and loaded on first use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ..translate import DEFAULT_MODEL, LANGUAGES, MODELS, SourceLanguages, Translator
from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, RunContext

log = logging.getLogger(__name__)

EASYOCR_DIR = MODELS_DIR / "easyocr"
MAX_SIDE = 1280  # downscale before OCR; boxes are scaled back to the original pixels
# readtext tuning (measured on the 3090): batch_size=16 recognizes all crops in one pass instead of one call per
# word (960x720 sign 67 → 55 ms, 810x1080 street photo 130 → 92 ms). canvas_size only caps the detector input,
# which MAX_SIDE already does; cudnn.benchmark was not a clear win and is process-global, so it stays off.
READTEXT_KWARGS: dict[str, Any] = {"batch_size": 16, "canvas_size": MAX_SIDE, "paragraph": False}
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


class TextPipeline(Pipeline):
    info = PipelineInfo(
        id="text",
        name="Text (OCR)",
        description="Reads text on signs, labels and screens (80+ languages); choose a script or auto-detect installed scripts.",
        default_enabled=False,
        uses_global_conf=False,
        interval_ms=400,
        options=[
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

    def load(self, device: str) -> None:
        if device.startswith("cuda"):
            try:
                _check_cudnn_usable()
            except RuntimeError as exc:
                log.warning("text: %s — running OCR on the CPU instead (slow)", exc)
                device = "cpu"
        self._device = device if device.startswith("cuda") else False
        reader = self._reader(DEFAULT_GROUP)
        reader.readtext(np.full((64, 256, 3), 255, np.uint8), **READTEXT_KWARGS)  # warm up CUDA kernels

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
        selected = LEGACY_LANGUAGES.get(ctx.options["languages"], ctx.options["languages"])
        if selected == AUTO_SCRIPTS:
            candidates = [
                group
                for group in SCRIPT_GROUPS
                if group.key == DEFAULT_GROUP.key or (EASYOCR_DIR / group.model_file).is_file()
            ]
        else:
            candidates = [GROUPS[selected]]

        best_group = candidates[0]
        best: list[Annotation] = []
        best_score = -1.0
        for group in candidates:
            try:
                annotations = self._read_group(image, group, float(ctx.options["min_conf"]))
            except Exception:
                if len(candidates) == 1:
                    raise
                log.exception("text: auto-detect could not run %s; trying the remaining installed scripts", group.key)
                continue
            # Prefer recognitions that return more high-confidence letters, while capping each line so a single
            # hallucinated paragraph cannot swamp several valid short signs. Ties retain the earlier script.
            score = sum(a.conf * min(sum(ch.isalpha() for ch in a.label), 40) for a in annotations)
            if score > best_score:
                best_group, best, best_score = group, annotations, score

        if selected == AUTO_SCRIPTS and not best:
            # Latin is always installed by the normal server setup; preserve its clear empty result if every
            # optional recognizer failed to initialize.
            best_group = DEFAULT_GROUP
        out = best
        target = ctx.options.get("translate_to", TRANSLATE_OFF)
        if out and target != TRANSLATE_OFF:
            self._translate(out, best_group, ctx.options, LANGUAGES[target])
        return out

    def _read_group(self, image: np.ndarray, group: ScriptGroup, min_conf: float) -> list[Annotation]:
        reader = self._reader(group)
        h, w = image.shape[:2]

        scale = min(1.0, MAX_SIDE / max(h, w))
        small = (
            image
            if scale == 1.0
            else cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # EasyOCR treats ndarrays as RGB

        out: list[Annotation] = []
        for quad, text, conf in reader.readtext(rgb, **READTEXT_KWARGS):
            label = text.strip()
            if not label or conf < min_conf:
                continue
            pts = np.asarray(quad, dtype=np.float64) / scale
            x1, y1 = np.clip(pts.min(axis=0), 0, (w, h))
            x2, y2 = np.clip(pts.max(axis=0), 0, (w, h))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Annotation(box=(float(x1), float(y1), float(x2), float(y2)), conf=float(conf), label=label))

        return out

    def _translate(self, anns: list[Annotation], group: ScriptGroup, options: dict[str, Any], target: str) -> None:
        """Label = translation, detail = "<SOURCE>: <original>" (show_original). Failures leave the text as read."""
        if self._translate_error and time.monotonic() - self._translate_failed_at < TRANSLATE_RETRY_S:
            for a in anns:
                a.detail = self._translate_error
            return
        if self._translator is None:
            self._translator = Translator(self._device if isinstance(self._device, str) else "cpu")
        source = options.get("translate_from", AUTO_DETECT)
        try:
            results = self._translator.translate(
                [a.label for a in anns],
                target,
                group.sources,
                source=None if source == AUTO_DETECT else LANGUAGES[source],
                model=options.get("translation_model", DEFAULT_MODEL),
            )
        except Exception as exc:
            log.exception("text: translation failed; showing the text untranslated for %.0f s", TRANSLATE_RETRY_S)
            self._translator.unload_model()
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
