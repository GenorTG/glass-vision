"""docs/PROTOCOL.md "Read text" (v3.2) through the real FastAPI app: POST /text/read, POST /text/translate,
GET /text/languages, hello `features`, and the text pipeline's `live` option over the WebSocket.

Validation, auth and 503 run without models (the text pipeline is registered but never loaded); everything that reads
or translates uses the real EasyOCR + M2M100 models on the GPU."""

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from ultralytics.utils import ASSETS

from glass_vision import app as app_module
from glass_vision.app import create_app
from glass_vision.config import Settings
from glass_vision.pipelines.base import Annotation, RunContext
from glass_vision.pipelines.text import (
    AUTO_SCRIPTS,
    DEFAULT_GROUP,
    LIVE_MARK,
    MARK_LABEL,
    SCRIPT_GROUPS,
    TextPipeline,
    _auto_score,
    group_for_text,
)
from glass_vision.protocol import pack_frame
from glass_vision.translate import DEFAULT_MODEL, MODELS, TRANSLATION_DIR, Translator

TOKEN = "s3cret-token"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BUS = ASSETS / "bus.jpg"
GROUP = {g.key: g for g in SCRIPT_GROUPS}
needs_translation = pytest.mark.skipif(
    not (TRANSLATION_DIR / MODELS[DEFAULT_MODEL].dir / "pytorch_model.bin").is_file(),
    reason="M2M100 418M not downloaded (scripts/download_models.py --only text)",
)


def jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def render_sign() -> np.ndarray:
    """A German sign, 1600x1200: a title, a two-line notice below it, and a phone number far bottom right."""
    pil = Image.new("RGB", (1600, 1200), "white")
    draw = ImageDraw.Draw(pil)
    draw.text((100, 80), "NOTAUSGANG", font=ImageFont.truetype(str(FONT), 150), fill="black")
    small = ImageFont.truetype(str(FONT), 64)
    draw.text((100, 420), "Rauchen verboten", font=small, fill="black")
    draw.text((100, 510), "Bitte nicht berühren", font=small, fill="black")
    draw.text((1000, 1050), "Tel. 0800 123", font=small, fill="black")
    return np.asarray(pil)[:, :, ::-1].copy()


def make_client(tmp_path: Path, token: str | None) -> TestClient:
    settings = Settings(device="cpu", token=token, data_dir=tmp_path / "data", apk_path=tmp_path / "none.apk")
    return TestClient(create_app(settings))


@pytest.fixture
def unloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """App with the text pipeline registered but not loaded (and never loaded by these tests)."""
    monkeypatch.setattr(app_module, "available_pipelines", lambda: {"text": TextPipeline})
    with make_client(tmp_path, None) as client:
        yield client
        assert client.get("/health").json()["pipelines"] == {"text": "not_loaded"}  # validation came first


# --- no models ----------------------------------------------------------------------------------------------------
def test_reads_default_to_the_largest_installed_translation_model(tmp_path: Path, monkeypatch) -> None:
    from glass_vision import text_read

    monkeypatch.setattr(text_read, "TRANSLATION_DIR", tmp_path)
    assert text_read.best_installed_model() == DEFAULT_MODEL  # nothing installed: the usual default
    for spec in MODELS.values():
        (tmp_path / spec.dir).mkdir()
        (tmp_path / spec.dir / "config.json").write_text("{}")
        assert text_read.best_installed_model() == next(n for n, s in MODELS.items() if s is spec)


def test_languages_list(unloaded: TestClient) -> None:
    languages = unloaded.get("/text/languages").json()["languages"]
    assert {"code": "pl", "name": "Polish"} in languages and {"code": "de", "name": "German"} in languages
    assert [entry["name"] for entry in languages] == sorted(entry["name"] for entry in languages)


@pytest.mark.parametrize(
    ("query", "body"),
    [
        ({}, b"not a jpeg"),
        ({}, b""),
        ({}, b"\xff\xd8\xff\xe0 truncated"),  # JPEG magic, undecodable
        ({"languages": "Klingon"}, None),
        ({"translate_to": "pl,xx"}, None),
        ({"translate_from": "german"}, None),
        ({"model": "huge"}, None),
    ],
)
def test_read_rejects_bad_input(unloaded: TestClient, query: dict, body: bytes | None) -> None:
    content = body if body is not None else jpeg(np.full((64, 64, 3), 255, np.uint8))
    r = unloaded.post("/text/read", params=query, content=content, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"to": "fr"}',
        b'{"lines": "Ausgang", "to": "fr"}',
        b'{"lines": [1], "to": "fr"}',
        b'{"lines": ["Ausgang"], "to": "xx"}',
        b'{"lines": ["Ausgang"]}',
        b'{"lines": ["Ausgang"], "from": ["de", "de"], "to": "fr"}',
        b'{"lines": ["Ausgang"], "from": ["deutsch"], "to": "fr"}',
        b'{"lines": ["Ausgang"], "from": [1], "to": "fr"}',
        b'{"lines": ["Ausgang"], "to": "fr", "model": "huge"}',
    ],
)
def test_translate_rejects_bad_input(unloaded: TestClient, body: bytes) -> None:
    r = unloaded.post("/text/translate", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400, r.text


def test_translate_nothing(unloaded: TestClient) -> None:
    r = unloaded.post("/text/translate", json={"lines": [], "to": "fr"})
    assert r.json() == {"to": "fr", "translations": []}


def test_text_routes_need_the_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "available_pipelines", lambda: {"text": TextPipeline})
    with make_client(tmp_path, TOKEN) as client:
        image = jpeg(np.full((64, 64, 3), 255, np.uint8))
        assert client.get("/text/languages").status_code == 401
        assert client.post("/text/read", content=image).status_code == 401
        assert client.post("/text/translate", json={"lines": [], "to": "fr"}).status_code == 401
        assert client.get("/text/languages", params={"token": TOKEN}).status_code == 200
        headers = {"Authorization": f"Bearer {TOKEN}"}
        assert client.post("/text/translate", json={"lines": [], "to": "fr"}, headers=headers).status_code == 200
        # Authorized but invalid → 400, not 401.
        assert client.post("/text/read", params={"token": TOKEN}, content=b"x").status_code == 400


class BrokenText(TextPipeline):
    def load(self, device: str) -> None:
        raise RuntimeError("EasyOCR weights missing")


def test_503_when_ocr_cannot_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "available_pipelines", lambda: {"text": BrokenText})
    with make_client(tmp_path, None) as client:
        r = client.post("/text/read", content=jpeg(np.full((64, 64, 3), 255, np.uint8)))
        assert r.status_code == 503 and "EasyOCR weights missing" in r.json()["detail"]
        assert client.post("/text/translate", json={"lines": ["Ausgang"], "to": "fr"}).status_code == 503


def test_without_a_text_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "available_pipelines", dict)
    with make_client(tmp_path, None) as client:
        assert client.post("/text/read", content=jpeg(np.full((64, 64, 3), 255, np.uint8))).status_code == 503
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["features"] == []


def test_hello_announces_text_read_and_the_live_option(unloaded: TestClient) -> None:
    with unloaded.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
    assert hello["features"] == ["text_read"]
    text = next(p for p in hello["pipelines"] if p["id"] == "text")
    live = next(o for o in text["options"] if o["key"] == "live")
    assert live == {
        "key": "live",
        "label": "While streaming",
        "type": "choice",
        "default": "Read live",
        "choices": ["Mark text areas", "Read live"],
    }


def test_group_for_text() -> None:
    assert group_for_text(["Ausgang", "12"]) is DEFAULT_GROUP
    assert group_for_text(["Выход", "Exit"]) is GROUP["cyrillic"]
    assert group_for_text(["出口"]) is GROUP["zh_sim"]
    assert group_for_text(["非常口", "でぐち"]) is GROUP["japanese"]
    assert group_for_text(["출구"]) is GROUP["korean"]
    assert group_for_text([]) is DEFAULT_GROUP


def test_auto_score_needs_the_groups_own_script() -> None:
    """bus.jpg regression: non-Latin recognizers read the English too, plus a hallucinated letter of their script."""
    english = [Annotation((0, 0, 1, 1), 0.95, "CerO"), Annotation((0, 0, 1, 1), 0.9, "emisiones")]
    hallucinated = [*english, Annotation((0, 0, 1, 1), 0.99, "Ш")]
    assert _auto_score(english, DEFAULT_GROUP) > 0
    assert _auto_score(english, GROUP["korean"]) == 0
    assert _auto_score(hallucinated, GROUP["cyrillic"]) == 0  # 1 of 14 letters
    assert _auto_score([Annotation((0, 0, 1, 1), 0.9, "Выход запрещён"), *english], GROUP["cyrillic"]) > 0  # 13 of 26
    assert _auto_score([Annotation((0, 0, 1, 1), 0.9, "非常口")], GROUP["japanese"]) > 0


# --- real models ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    patch = pytest.MonkeyPatch()
    patch.setattr(app_module, "available_pipelines", lambda: {"text": TextPipeline})
    settings = Settings(device="cuda:0", data_dir=tmp_path_factory.mktemp("data"))
    with TestClient(create_app(settings)) as c:
        yield c
    patch.undo()


@pytest.mark.gpu
@needs_translation
def test_read_a_sign_in_order_with_blocks_languages_and_two_translations(client: TestClient) -> None:
    t0 = time.perf_counter()
    r = client.post("/text/read", params={"translate_to": "pl,en"}, content=jpeg(render_sign()))
    print(f"\n[read] sign 1600x1200: {(time.perf_counter() - t0) * 1000:.0f} ms, {r.json()['ms']}")
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["w"], out["h"], out["script"], out["errors"]) == (1600, 1200, "latin", {})
    lines = out["lines"]
    for line in lines:
        print(f"[read]   {line}")
        x1, y1, x2, y2 = line["box"]
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and 0 < line["conf"] <= 1
    assert [line["text"] for line in lines] == [
        "NOTAUSGANG",
        "Rauchen verboten",
        "Bitte nicht berühren",
        "Tel. 0800 123",
    ]
    assert [line["block"] for line in lines] == [0, 1, 1, 2]
    assert [line["lang"] for line in lines] == ["de", "de", "de", None]
    assert set(out["translations"]) == {"pl", "en"}
    en, pl = out["translations"]["en"], out["translations"]["pl"]
    print(f"[read]   en {en}\n[read]   pl {pl}")
    assert len(en) == len(pl) == 4 and en[3] is None and pl[3] is None  # numbers stay as read
    assert "exit" in en[0].lower() and "smok" in en[1].lower()
    assert all(isinstance(t, str) and t for t in pl[:3])


@pytest.mark.gpu
@needs_translation
def test_read_a_real_photo(client: TestClient) -> None:
    """ultralytics' bus.jpg (810x1080 street photo): "cero emisiones" on the bus."""
    t0 = time.perf_counter()
    r = client.post("/text/read", content=BUS.read_bytes(), params={"translate_to": "en"})
    print(f"\n[read] bus.jpg: {(time.perf_counter() - t0) * 1000:.0f} ms, {r.json()['ms']}")
    out = r.json()
    print("[read]  ", [(line["text"], line["block"], line["lang"]) for line in out["lines"]])
    assert (out["w"], out["h"], out["script"]) == (810, 1080, "latin")
    texts = [line["text"].lower() for line in out["lines"]]
    assert "emisiones" in texts
    emisiones = texts.index("emisiones")
    assert out["lines"][emisiones]["lang"] == "es"
    assert "emission" in out["translations"]["en"][emisiones].lower()


@pytest.mark.gpu
def test_read_finds_nothing_on_a_blank_still(client: TestClient) -> None:
    r = client.post("/text/read", params={"translate_to": "pl"}, content=jpeg(np.full((900, 1200, 3), 255, np.uint8)))
    assert r.status_code == 200
    assert {k: r.json()[k] for k in ("lines", "translations", "errors")} == {
        "lines": [],
        "translations": {"pl": []},
        "errors": {},
    }


@pytest.mark.gpu
@needs_translation
def test_a_failing_target_lands_in_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    generate = Translator._generate

    def flaky(self, items, target, model):
        if target == "fr":
            raise RuntimeError("CUDA out of memory (simulated)")
        return generate(self, items, target, model)

    monkeypatch.setattr(Translator, "_generate", flaky)
    r = client.post("/text/read", params={"translate_to": "fr,en"}, content=jpeg(render_sign()))
    out = r.json()
    assert r.status_code == 200 and set(out["translations"]) == {"en"}
    assert "simulated" in out["errors"]["fr"]


@pytest.mark.gpu
@needs_translation
def test_translate_endpoint(client: TestClient) -> None:
    body = {"lines": ["Ausgang", "Notausgang 2", "12:30", "Rauchen verboten"], "from": ["de", None, None, "de"]}
    r = client.post("/text/translate", json=body | {"to": "fr"})
    assert r.status_code == 200, r.text
    out = r.json()
    print(f"\n[translate] {out}")
    assert out["to"] == "fr" and len(out["translations"]) == 4
    assert "sortie" in out["translations"][0].lower()
    assert out["translations"][2] is None
    assert "fum" in out["translations"][3].lower()  # "Défense de fumer" / "Interdiction de fumer"
    # Without `from`: detected; a line already in the target language stays null.
    r = client.post("/text/translate", json={"lines": ["Wyjście awaryjne", "Bitte nicht berühren"], "to": "pl"})
    assert r.json()["translations"][0] is None and r.json()["translations"][1]


@pytest.mark.gpu
def test_explicit_script_and_legacy_value(client: TestClient) -> None:
    for languages in (DEFAULT_GROUP.label, "en"):
        r = client.post("/text/read", params={"languages": languages}, content=jpeg(render_sign()))
        assert r.status_code == 200 and r.json()["script"] == "latin"
        assert r.json()["lines"][0]["text"] == "NOTAUSGANG"


@pytest.mark.gpu
def test_mark_mode_over_the_websocket(client: TestClient) -> None:
    """`live` = Mark: detector-only tracks labeled "text"; switching back to Read live labels them with the text."""
    frame = jpeg(cv2.resize(render_sign(), (960, 720), interpolation=cv2.INTER_AREA))

    def stream(ws, predicate, start: int) -> list[dict]:
        for seq in range(start, start + 40):
            ws.send_bytes(pack_frame(seq, seq, frame))
            while (msg := ws.receive_json())["type"] != "detections":
                pass
            tracks = [t for t in msg["tracks"] if t["pipeline"] == "text"]
            if tracks and predicate(tracks):
                return tracks
            time.sleep(0.1)
        raise AssertionError("no text tracks")

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        text = {"enabled": True, "options": {"live": LIVE_MARK, "languages": AUTO_SCRIPTS}}
        ws.send_json({"type": "config", "protocol": 3, "pipelines": {"text": text}})
        tracks = stream(ws, lambda t: len(t) >= 4, 0)
        assert all(t["label"] == MARK_LABEL for t in tracks)
        ws.send_json({"type": "config", "pipelines": {"text": {"options": {"live": "Read live"}}}})
        tracks = stream(ws, lambda t: any(x["label"] == "NOTAUSGANG" for x in t), 100)
        assert not any(t["label"] == MARK_LABEL for t in tracks)  # tracks restarted: no "text" vote left over


@pytest.mark.gpu
def test_mark_is_cheaper_than_read(client: TestClient) -> None:
    client.post("/text/read", content=jpeg(np.full((64, 64, 3), 255, np.uint8)))  # loads the pipeline
    pipeline = app_module.state.manager.handles["text"].instance
    assert isinstance(pipeline, TextPipeline)
    sign = cv2.resize(render_sign(), (960, 720), interpolation=cv2.INTER_AREA)
    bus = cv2.imread(str(BUS))
    for name, image in (("sign 960x720", sign), ("bus.jpg 810x1080", bus)):
        timings = {}
        for live in ("Mark text areas", "Read live"):
            ctx = RunContext({"live": live, "languages": DEFAULT_GROUP.label, "min_conf": 0.3}, 0.25, 640)
            pipeline.run(image, ctx)
            runs = []
            for _ in range(10):
                t0 = time.perf_counter()
                anns = pipeline.run(image, ctx)
                runs.append((time.perf_counter() - t0) * 1000)
            timings[live] = sorted(runs)[len(runs) // 2]
            if live == "Mark text areas":
                assert anns and all(a.label == MARK_LABEL and a.conf == 1.0 for a in anns)
        print(f"\n[mark] {name}: mark {timings['Mark text areas']:.1f} ms, read {timings['Read live']:.1f} ms")
        assert timings["Mark text areas"] < timings["Read live"]
