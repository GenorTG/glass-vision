"""Translation of OCR lines: M2M100 (many-to-many, 100 languages) + lingua source-language detection.

Used by the text pipeline (`pipelines/text.py`) when the user picks a target language. Loaded on first use, not with
the OCR models; the text pipeline drops it when it unloads.

- Model: facebook/m2m100_418M (MIT), optionally facebook/m2m100_1.2B (MIT; better on one-word signs, ~2x slower,
  2.4 GB VRAM). `scripts/download_models.py` fetches them from their original Hugging Face repos at a pinned commit
  into `MODELS_DIR/translation/<name>`; nothing is downloaded at run time. fp16 on CUDA, fp32 on the CPU.
- Source language per line: given by the user, or detected with lingua among the languages the OCR script group can
  produce (a small candidate set makes short-text detection far more reliable). Short lines that detect weakly fall
  back to the language of the whole frame (a sign is usually one language); if that is unclear too, they stay as read.
- Numbers, codes and URLs are never translated; lines already in the target language are skipped.
- Every (line, source, target, model) is translated once (LRU cache): OCR re-reads the same sign on every run. The
  uncached lines of one run go to the model as one batch.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from . import memory
from .config import settings

log = logging.getLogger(__name__)

TRANSLATION_DIR = settings.models_dir / "translation"


@dataclass(frozen=True)
class ModelSpec:
    dir: str  # under TRANSLATION_DIR (scripts/download_models.py puts it there)
    download: str  # scripts/download_models.py --translation-models value


# Option value shown in the app → model. The first one is the default (and the one the installer downloads).
MODELS: dict[str, ModelSpec] = {
    "Fast — M2M100 418M": ModelSpec("m2m100_418M", "418M"),
    "Better — M2M100 1.2B (2.4 GB VRAM)": ModelSpec("m2m100_1.2B", "1.2B"),
}
DEFAULT_MODEL = next(iter(MODELS))

# Languages offered in the app (readable name → M2M100 language code), as translation target or source.
LANGUAGES: dict[str, str] = {
    "Arabic": "ar",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Marathi": "mr",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

# lingua confidences (0..1, relative to the other candidates) taken at face value: the best guess is >= CONFIDENT, or
# >= PLAUSIBLE and RATIO x the runner-up ("Ausgang": de 0.50 / af 0.13). Short English is lingua's weak spot
# ("Staff only": en 0.38 / cy 0.21): such lines borrow the frame's language or stay as read; "Translate from" fixes it.
CONFIDENT = 0.5
PLAUSIBLE = 0.3
RATIO = 3.0
LONG_LETTERS = 20  # a weakly detected line this long is still translated from its best guess
SHORT_LETTERS = 8  # a line this short takes its block's language when that one is sure (same rule as the phone)
CACHE_SIZE = 2000
MAX_BATCH = 32  # lines per generate call
NUM_BEAMS = 4  # measured on the 3090: 14 sign lines 83 ms greedy vs 109 ms with 4 beams, noticeably better wording

_URL = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\w+|^[\w.-]+\.[a-z]{2,6}(/\S*)?$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceLanguages:
    """The source languages detection may pick for one OCR script group (M2M100 codes, main language first)."""

    candidates: tuple[str, ...]
    # Non-Latin script groups read English as their only Latin-script language: an ASCII-only line is English.
    ascii_is_english: bool = False


class Guess(NamedTuple):
    language: str | None  # M2M100 code
    confidence: float  # 0..1
    sure: bool


@dataclass(frozen=True)
class Translation:
    text: str
    source: str  # M2M100 code the line was translated from


def letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def normalize(text: str) -> str:
    """Collapse whitespace; ALL-CAPS lines (common on signs) → sentence case, which M2M100 translates far better
    ("NOTAUSGANG" → "Noticing", "Notausgang" → "Emergency Exit"; "ВХОД ВОСПРЕЩЁН" → "Entrance allowed" (!) vs
    "Вход воспрещён" → "Entrance prohibited")."""
    text = " ".join(text.split())
    if letter_count(text) >= 3 and text == text.upper() and text != text.lower():
        text = text[:1] + text[1:].lower()
    return text


def is_translatable(text: str) -> bool:
    """False for what must stay as read: numbers, codes ("A12-B", "PL 1234"), mostly-numeric lines ("Tel. 0800 123",
    "12,99 zł"), URLs, e-mail addresses."""
    letters = letter_count(text)
    if letters < 2 or sum(ch.isdigit() for ch in text) >= letters or _URL.search(text):
        return False
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    return not all(any(ch.isdigit() for ch in w) for w in words)


def _lingua_code(m2m: str) -> list[str]:
    return ["NB", "NN"] if m2m == "no" else [m2m.upper()]


class SourceDetector:
    """Guesses a line's language among `SourceLanguages.candidates` (lingua, high-accuracy mode)."""

    def __init__(self, sources: SourceLanguages) -> None:
        self.sources = sources
        native = tuple(c for c in sources.candidates if not (sources.ascii_is_english and c == "en"))
        self.native = native
        self._lingua_to_m2m: dict[Any, str] = {}
        self._detector: Any = None
        if len(native) < 2:
            return
        from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

        for code in native:
            for iso in _lingua_code(code):
                if hasattr(IsoCode639_1, iso):
                    self._lingua_to_m2m[Language.from_iso_code_639_1(getattr(IsoCode639_1, iso))] = code
        if len(set(self._lingua_to_m2m.values())) >= 2:
            t0 = time.perf_counter()
            self._detector = (
                LanguageDetectorBuilder.from_languages(*self._lingua_to_m2m).with_preloaded_language_models().build()
            )
            log.info(
                "translate: language detector for %s built in %.1fs",
                ",".join(sorted(set(self._lingua_to_m2m.values()))),
                time.perf_counter() - t0,
            )

    @property
    def languages(self) -> set[str]:
        """Every code `guess` can return."""
        out = set(self._lingua_to_m2m.values()) if self._detector is not None else set(self.native[:1])
        return out | ({"en"} if self.sources.ascii_is_english else set())

    def guess(self, text: str) -> Guess:
        if self.sources.ascii_is_english and all(ord(ch) < 128 or not ch.isalpha() for ch in text):
            return Guess("en", 1.0, True)
        if self._detector is None:
            return Guess(self.native[0], 1.0, True) if self.native else Guess(None, 0.0, False)
        values = self._detector.compute_language_confidence_values(text)
        if not values or values[0].value <= 0:
            return Guess(None, 0.0, False)
        top = float(values[0].value)
        second = float(values[1].value) if len(values) > 1 else 0.0
        sure = top >= CONFIDENT or (top >= PLAUSIBLE and top >= RATIO * second)
        return Guess(self._lingua_to_m2m[values[0].language], top, sure)


class Translator:
    """Translate OCR lines into one target language. Not thread-safe (the text pipeline runs it under its lock)."""

    def __init__(self, device: str, cache_size: int = CACHE_SIZE) -> None:
        self.device = device
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()
        self._detectors: dict[SourceLanguages, SourceDetector] = {}
        self._model_name: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self.generate_calls = 0  # stats (tests, logs)
        self.lines_generated = 0

    # --- model ------------------------------------------------------------------------------------------------
    def _ensure_model(self, model: str) -> None:
        if self._model_name == model:
            return
        self.unload_model()
        import torch
        from transformers import AutoTokenizer, M2M100ForConditionalGeneration
        from transformers.utils import logging as hf_logging

        path = TRANSLATION_DIR / MODELS[model].dir
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"translation model {path.name} is not downloaded: "
                f"scripts/download_models.py --only text --translation-models {MODELS[model].download}"
            )
        hf_logging.disable_progress_bar()
        t0 = time.perf_counter()
        before = torch.cuda.memory_allocated() if self.device.startswith("cuda") else 0
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = M2M100ForConditionalGeneration.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self._model_name = model
        memory.release()  # the checkpoint is read on the CPU first (fp32 .bin: 1.2B peaks at ~8.7 GB RSS)
        vram = (torch.cuda.memory_allocated() - before) / 2**20 if self.device.startswith("cuda") else 0
        log.info(
            "translate: loaded %s on %s in %.1fs (%.0f MiB VRAM)",
            path.name,
            self.device,
            time.perf_counter() - t0,
            vram,
        )

    def unload_model(self) -> None:
        if self._model is None:
            return
        self._model = self._tokenizer = self._model_name = None
        memory.release()

    def unload(self) -> None:
        self.unload_model()
        self._detectors.clear()
        self._cache.clear()

    def _generate(self, items: Sequence[tuple[str, str]], target: str, model: str) -> list[str]:
        """[(text, source code)] → translations, in batches (the source token is per line, so one batch may mix)."""
        import torch

        self._ensure_model(model)
        tok, net = self._tokenizer, self._model
        out: list[str] = []
        for start in range(0, len(items), MAX_BATCH):
            chunk = items[start : start + MAX_BATCH]
            ids = []
            for text, source in chunk:
                tok.src_lang = source
                ids.append(tok(text, truncation=True, max_length=128).input_ids)
            batch = tok.pad({"input_ids": ids}, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = net.generate(
                    **batch,
                    forced_bos_token_id=tok.get_lang_id(target),
                    num_beams=NUM_BEAMS,
                    max_length=min(256, 2 * max(len(i) for i in ids) + 10),
                )
            out += [t.strip() for t in tok.batch_decode(generated, skip_special_tokens=True)]
            self.generate_calls += 1
            self.lines_generated += len(chunk)
        return out

    # --- detection --------------------------------------------------------------------------------------------
    def detector(self, sources: SourceLanguages) -> SourceDetector:
        found = self._detectors.get(sources)
        if found is None:
            found = self._detectors[sources] = SourceDetector(sources)
        return found

    def _sources_for(
        self, texts: Sequence[str], sources: SourceLanguages, blocks: Sequence[int] | None = None
    ) -> list[str | None]:
        """Source language per (normalized, translatable) line; None = leave it as read. A weakly detected line takes
        its block's language (`blocks`: block id per line, lines of one sign), else the whole frame's, if that one is
        sure."""
        detector = self.detector(sources)
        guesses: dict[int | None, Guess] = {}  # block id (None = all lines) → guess on its joined text

        def joined(block: int | None) -> Guess:
            if block not in guesses:
                group = [t for t, b in zip(texts, blocks or [None] * len(texts), strict=True) if block in (None, b)]
                guesses[block] = detector.guess(" ".join(group)) if len(group) > 1 else Guess(None, 0.0, False)
            return guesses[block]

        out: list[str | None] = []
        for k, text in enumerate(texts):
            # A short line inside a sign follows the rest of the sign: one misread letter makes lingua "sure" of
            # nonsense ("cerỌ" next to "emisiones" → Vietnamese, because only Vietnamese has Ọ), and it drags the
            # sign's joined text along, so the line itself is left out of that guess.
            if blocks is not None and letter_count(text) < SHORT_LETTERS:
                others = [t for j, t in enumerate(texts) if j != k and blocks[j] == blocks[k]]
                rest = detector.guess(" ".join(others)) if others else Guess(None, 0.0, False)
                if rest.sure:
                    out.append(rest.language)
                    continue
            guess = detector.guess(text)
            code = guess.language
            if not guess.sure:
                context = [joined(blocks[k]), joined(None)] if blocks is not None else [joined(None)]
                sure = next((g for g in context if g.sure), None)
                if sure is not None:
                    code = sure.language
                elif letter_count(text) < LONG_LETTERS:
                    code = None
            out.append(code)
        return out

    # --- API --------------------------------------------------------------------------------------------------
    def detect_sources(
        self, lines: Sequence[str], sources: SourceLanguages, blocks: Sequence[int] | None = None
    ) -> list[str | None]:
        """Source language per line as `translate` would detect it (M2M100 code); None = not translatable (number,
        code, URL) or no confident guess. `blocks` (block id per line) adds the line's block as detection context."""
        texts = [normalize(line) for line in lines]
        todo = [i for i, t in enumerate(texts) if is_translatable(t)]
        todo_blocks = [blocks[i] for i in todo] if blocks is not None else None
        out: list[str | None] = [None] * len(lines)
        for i, code in zip(todo, self._sources_for([texts[i] for i in todo], sources, todo_blocks), strict=True):
            out[i] = code
        return out

    def translate(
        self,
        lines: Sequence[str],
        target: str,
        sources: SourceLanguages,
        *,
        source: str | Sequence[str | None] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> list[Translation | None]:
        """One result per line: None = keep the line as read (number/code/URL, already `target`, unknown source, or
        the model returned it unchanged). `source` (M2M100 code) skips detection; a sequence gives one code per line,
        None entries are detected (as part of all lines, so a weak line still borrows the others' language)."""
        texts = [normalize(line) for line in lines]
        todo = [i for i, t in enumerate(texts) if is_translatable(t)]
        if isinstance(source, str):
            src: list[str | None] = [source] * len(todo)
        else:
            given = list(source) if source is not None else [None] * len(lines)
            if len(given) != len(lines):
                raise ValueError(f"{len(given)} source languages for {len(lines)} lines")
            detected = self.detect_sources(lines, sources) if any(given[i] is None for i in todo) else given
            src = [given[i] if given[i] is not None else detected[i] for i in todo]

        results: list[Translation | None] = [None] * len(lines)
        keys: dict[int, tuple[str, str, str, str]] = {}
        for i, code in zip(todo, src, strict=True):
            if code is not None and code != target:
                keys[i] = (texts[i], code, target, model)
        missing = list(dict.fromkeys(k for k in keys.values() if k not in self._cache))
        if missing:
            for key, text in zip(missing, self._generate([(k[0], k[1]) for k in missing], target, model), strict=True):
                self._cache[key] = text
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        for i, key in keys.items():
            text = self._cache.get(key)
            if text is None:  # evicted while this batch was stored (batch > cache size): translate it again
                text = self._generate([(key[0], key[1])], target, model)[0]
            else:
                self._cache.move_to_end(key)
            if text and text.casefold() != texts[i].casefold():
                results[i] = Translation(text, key[1])
        return results
