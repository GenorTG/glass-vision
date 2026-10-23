"""Translation (glass_vision/translate.py): rules, language detection and caching with a fake model (CPU, real
lingua), then the real M2M100 on the GPU. Run with -s to see translations, latency and VRAM."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from glass_vision.pipelines.text import AUTO_SCRIPTS, SCRIPT_GROUPS, TextPipeline
from glass_vision.translate import (
    DEFAULT_MODEL,
    LANGUAGES,
    MODELS,
    TRANSLATION_DIR,
    SourceDetector,
    Translator,
    is_translatable,
    normalize,
)

ROOT = Path(__file__).resolve().parents[2]
GROUP = {g.key: g for g in SCRIPT_GROUPS}
LATIN = GROUP["latin"].sources
CYRILLIC = GROUP["cyrillic"].sources


def download_models():
    spec = importlib.util.spec_from_file_location("download_models", ROOT / "scripts" / "download_models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


# --- pure rules -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NOTAUSGANG", "Notausgang"),
        ("RAUCHEN  VERBOTEN", "Rauchen verboten"),
        ("Rauchen verboten", "Rauchen verboten"),
        ("  Wyjście\tewakuacyjne ", "Wyjście ewakuacyjne"),
        ("ВХОД ВОСПРЕЩЁН", "Вход воспрещён"),
        ("OK", "OK"),  # too short to be sure it is a word, not an acronym
        ("出口", "出口"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["12:30", "1", "A12-B", "PL 1234", "B-52", "Tel. 0800 123 456", "12,99 zł", "https://example.com/x", "www.foo.pl",
     "kontakt@firma.pl", "example.com", "-", "x"],
)  # fmt: skip
def test_untranslatable(text):
    assert not is_translatable(text)


@pytest.mark.parametrize("text", ["Ausgang", "Open 24h", "Mo-Fr 8-18 Uhr", "Wyjście ewakuacyjne", "出口", "Выход"])
def test_translatable(text):
    assert is_translatable(text)


# --- detection --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def detectors():
    """Shared between tests: building the Latin detector preloads ~34 lingua models (~1 s, ~600 MB RAM)."""
    return {}


def test_detection_is_restricted_to_the_script_group(detectors):
    latin = detectors.setdefault(LATIN, SourceDetector(LATIN))
    assert latin.languages == set(LATIN.candidates)
    assert "ru" not in latin.languages and "zh" not in latin.languages
    assert latin.guess("Bitte nicht berühren")[0] == "de"
    assert latin.guess("Wyjście ewakuacyjne")[::2] == ("pl", True)
    assert latin.guess("Ausgang")[::2] == ("de", True)  # 0.50 vs 0.13: sure by the margin rule
    assert not latin.guess("Uwaga").sure  # Tagalog 0.27, Malay 0.19…
    assert latin.guess("Sortie de secours")[0] == "fr"

    cyrillic = detectors.setdefault(CYRILLIC, SourceDetector(CYRILLIC))
    assert cyrillic.languages == set(CYRILLIC.candidates)
    assert cyrillic.guess("Ausgang") == ("en", 1.0, True)  # ASCII in a non-Latin group: EasyOCR's only Latin language
    assert cyrillic.guess("Посторонним вход воспрещён")[0] == "ru"
    assert cyrillic.guess("Вихід")[0] in CYRILLIC.candidates

    japanese = SourceDetector(GROUP["japanese"].sources)  # one native language: no lingua at all
    assert japanese.guess("非常口") == ("ja", 1.0, True)
    assert japanese.guess("EXIT") == ("en", 1.0, True)
    telugu = SourceDetector(GROUP["telugu"].sources)  # M2M100 has no Telugu: only English is translatable
    assert telugu.languages == {"en"}
    assert telugu.guess("నిష్క్రమణ") == (None, 0.0, False)


# --- translator with a fake model -------------------------------------------------------------------------------
class FakeTranslator(Translator):
    """Records what would go to the model; "translates" by tagging the text."""

    def __init__(self, detectors: dict) -> None:
        super().__init__("cpu")
        self._detectors = detectors
        self.batches: list[list[tuple[str, str]]] = []

    def _generate(self, items, target, model):
        self.batches.append(list(items))
        self.generate_calls += 1
        self.lines_generated += len(items)
        return [f"[{src}>{target}] {text}" for text, src in items]


@pytest.fixture
def fake(detectors):
    return FakeTranslator(detectors)


def test_skips_numbers_urls_and_target_language(fake):
    lines = ["12:30", "www.bahn.de", "Bitte nicht berühren", "Notausgang"]
    out = fake.translate(lines, "de", LATIN)  # already German
    assert out == [None, None, None, None]
    assert fake.generate_calls == 0

    out = fake.translate(lines, "en", LATIN)
    assert out[:2] == [None, None]
    assert out[2].text == "[de>en] Bitte nicht berühren" and out[2].source == "de"
    assert out[3].source == "de"  # short line: the frame's language


def test_batches_and_caches(fake):
    lines = ["Rauchen verboten", "Zutritt nur für Personal", "Bitte nicht berühren"]
    first = fake.translate(lines, "en", LATIN)
    assert fake.generate_calls == 1 and fake.lines_generated == 3  # one batch for the whole run
    again = fake.translate(lines, "en", LATIN)
    assert again == first
    assert fake.generate_calls == 1 and fake.lines_generated == 3  # everything from the cache

    fake.translate([*lines, "Kein Ausgang"], "en", LATIN)
    assert fake.batches[-1] == [("Kein Ausgang", "de")]  # only the new line
    fake.translate(lines, "pl", LATIN)  # another target is another cache entry
    assert fake.lines_generated == 3 + 1 + 3


def test_cache_is_bounded(detectors):
    small = FakeTranslator(detectors)
    small.cache_size = 2
    lines = ["Rauchen verboten", "Zutritt nur für Personal", "Bitte nicht berühren"]
    out = small.translate(lines, "en", LATIN, source="de")
    assert all(o is not None for o in out) and len(small._cache) == 2


def test_weak_short_line_alone_stays_as_read_unless_the_source_is_given(fake):
    # lingua on a lone "Uwaga" (Polish "attention"): Tagalog 0.28, Malay 0.19… → not guessed
    assert fake.translate(["Uwaga"], "en", LATIN) == [None]
    assert fake.generate_calls == 0
    out = fake.translate(["Uwaga"], "en", LATIN, source="pl")
    assert out[0].text == "[pl>en] Uwaga" and out[0].source == "pl"


def test_per_line_sources(fake):
    """Read text: `source` per line (None = detect, with all lines as context); a mismatched length is an error."""
    lines = ["Bitte nicht berühren", "Uwaga", "Rauchen verboten"]
    out = fake.translate(lines, "en", LATIN, source=[None, "pl", None])
    assert [o.source for o in out] == ["de", "pl", "de"]
    assert out[1].text == "[pl>en] Uwaga"
    with pytest.raises(ValueError, match="2 source languages for 3 lines"):
        fake.translate(lines, "en", LATIN, source=["de", "de"])


def test_detect_sources_uses_the_block_before_the_whole_frame(fake):
    # "Notausgang" alone is a weak guess; its block says German. The Polish block next to it doesn't drag it along.
    lines = ["Notausgang", "Bitte nicht berühren", "Wyjście ewakuacyjne", "Zakaz palenia tytoniu", "12:30"]
    assert fake.detect_sources(lines, LATIN, blocks=[0, 0, 1, 1, 2]) == ["de", "de", "pl", "pl", None]
    assert fake.detect_sources(["Uwaga"], LATIN) == [None]  # alone and weak: unknown


def test_unchanged_output_is_not_a_translation(detectors):
    class Echo(FakeTranslator):
        def _generate(self, items, target, model):
            return [text.upper() for text, _ in items]

    assert Echo(detectors).translate(["Bitte nicht berühren"], "en", LATIN, source="de") == [None]


def test_missing_model_is_a_clear_error(tmp_path, monkeypatch, detectors):
    monkeypatch.setattr("glass_vision.translate.TRANSLATION_DIR", tmp_path)
    t = Translator("cpu")
    t._detectors = detectors
    with pytest.raises(FileNotFoundError, match="download_models.py --only text --translation-models 418M"):
        t.translate(["Bitte nicht berühren"], "en", LATIN)


# --- consistency with the option lists and scripts/download_models.py --------------------------------------------
def test_languages_are_m2m100_codes():
    from transformers.models.m2m_100.tokenization_m2m_100 import FAIRSEQ_LANGUAGE_CODES

    m2m = set(FAIRSEQ_LANGUAGE_CODES["m2m100"])
    assert set(LANGUAGES.values()) <= m2m
    for name in ("English", "Polish", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch", "Czech",
                 "Ukrainian", "Russian", "Turkish", "Swedish", "Chinese", "Japanese", "Korean", "Arabic", "Hindi"):  # fmt: skip
        assert name in LANGUAGES
    for group in SCRIPT_GROUPS:
        assert set(group.sources.candidates) <= m2m, group.key


def test_download_script_matches_the_pipeline():
    dm = download_models()
    assert {g.key for g in SCRIPT_GROUPS} == set(dm.OCR_SCRIPTS)
    for group in SCRIPT_GROUPS:
        assert dm.OCR_SCRIPTS[group.key][2] == group.model_file, group.key
    assert dm.DEFAULT_OCR_SCRIPTS == (SCRIPT_GROUPS[0].key,)
    assert {spec.download for spec in MODELS.values()} == set(dm.TRANSLATION_MODELS)
    for spec in MODELS.values():
        assert dm.TRANSLATION_MODELS[spec.download][2] == f"translation/{spec.dir}"
    assert dm.DEFAULT_TRANSLATION_MODELS == (MODELS[DEFAULT_MODEL].download,)


def test_options_and_legacy_language_values():
    specs = {o.key: o for o in TextPipeline.info.options}
    latin = SCRIPT_GROUPS[0].label
    assert specs["languages"].default == AUTO_SCRIPTS
    # Installed v3.1 apps send every option at its advertised default: "Read live" keeps their live OCR as it was.
    assert specs["live"].default == "Read live" and specs["live"].choices == ["Mark text areas", "Read live"]
    assert specs["languages"].coerce("en") == latin  # saved by older app versions
    assert specs["languages"].coerce("en+pl") == latin
    with pytest.raises(ValueError):
        specs["languages"].coerce("klingon")
    assert "aliases" not in specs["languages"].to_json()
    assert specs["languages"].to_json()["choices"] == [AUTO_SCRIPTS, *(g.label for g in SCRIPT_GROUPS)]
    assert specs["translate_to"].default == "Off" and specs["translate_to"].choices[1:] == list(LANGUAGES)
    assert specs["translate_from"].default == "Auto-detect"
    assert specs["show_original"].default is True


# --- the real model ---------------------------------------------------------------------------------------------
def needs_model(name: str):
    path = TRANSLATION_DIR / MODELS[name].dir / "pytorch_model.bin"
    return pytest.mark.skipif(
        not path.is_file(),
        reason=f"{path.parent.name} not downloaded (scripts/download_models.py --translation-models "
        f"{MODELS[name].download})",
    )


@pytest.fixture(scope="module")
def real(detectors):
    t = Translator("cuda:0")
    t._detectors = detectors
    yield t
    t.unload()


@pytest.mark.gpu
@needs_model(DEFAULT_MODEL)
def test_real_translations(real):
    cases = [
        ("Notausgang", "de", LATIN),
        ("NOTAUSGANG", "de", LATIN),
        ("Wyjście ewakuacyjne", "pl", LATIN),
        ("Выход", "ru", CYRILLIC),
        ("Посторонним вход воспрещён", "ru", CYRILLIC),
    ]
    import torch

    out = []
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for text, source, sources in cases:
        result = real.translate([text], "en", sources, source=source if len(text) < 8 else None)[0]
        print(f"\n[translate] {source} {text!r} → {result.text!r} ({result.source})")
        out.append(result)
    print(f"[translate] first {len(cases)} lines incl. model load: {(time.perf_counter() - t0) * 1000:.0f} ms, "
          f"VRAM {(torch.cuda.memory_allocated() - before) / 2**20:.0f} MiB")  # fmt: skip
    for (text, source, _), result in zip(cases, out, strict=True):
        assert result.source == source, text
    for i in range(4):
        assert "exit" in out[i].text.lower(), (cases[i][0], out[i].text)
    assert "prohibit" in out[4].text.lower() or "forbidden" in out[4].text.lower(), out[4].text

    lines = ["Rauchen verboten", "Bitte nicht berühren", "Zutritt nur für Personal", "Kein Ausgang", "Achtung Stufe"]
    t0 = time.perf_counter()
    first = real.translate(lines, "en", LATIN)
    uncached = (time.perf_counter() - t0) * 1000
    calls = real.generate_calls
    t0 = time.perf_counter()
    assert real.translate(lines, "en", LATIN) == first
    cached = (time.perf_counter() - t0) * 1000
    assert real.generate_calls == calls
    print(f"[translate] 5-line sign: {uncached:.0f} ms uncached (one batch), {cached:.1f} ms cached")
    print("[translate] " + " | ".join(r.text for r in first if r))
    assert uncached < 1000


@pytest.mark.gpu
@needs_model("Better — M2M100 1.2B (2.4 GB VRAM)")
def test_real_large_model_reads_one_word_signs(real):
    # The 418M model turns a lone "Ausgang" into "output"; the 1.2B one gets "Exit".
    model = "Better — M2M100 1.2B (2.4 GB VRAM)"
    import torch

    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    result = real.translate(["Ausgang"], "en", LATIN, source="de", model=model)[0]
    print(f"\n[translate] 1.2B: 'Ausgang' → {result.text!r} ({(time.perf_counter() - t0) * 1000:.0f} ms incl. load, "
          f"VRAM {(torch.cuda.memory_allocated() - before) / 2**20:+.0f} MiB vs 418M)")  # fmt: skip
    assert "exit" in result.text.lower()
    real.unload_model()
