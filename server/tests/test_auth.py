"""docs/PROTOCOL.md "Auth": GV_TOKEN gates every HTTP route but /health and the WebSocket (close 4401).

Runs the real app with no pipelines registered (no models, no GPU needed)."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from glass_vision import app as app_module
from glass_vision.app import RedactToken, create_app, request_token, token_ok
from glass_vision.config import Settings
from glass_vision.pipelines import base

TOKEN = "s3cret-token"


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator:
    monkeypatch.setattr(app_module, "available_pipelines", dict)  # no models
    monkeypatch.setattr(base, "DATA_DIR", tmp_path / "data")  # captures.router reads it at request time
    apk = tmp_path / "data" / "app" / "glass-vision.apk"
    apk.parent.mkdir(parents=True)
    apk.write_bytes(b"fake-apk")
    (apk.parent / "glass-vision.apk.json").write_text(
        json.dumps({"application_id": "org.example.gv", "version_code": 5, "version_name": "0.5.0.1"})
    )
    capture = tmp_path / "data" / "captures" / "20261023-101500-123"
    capture.mkdir(parents=True)
    (capture / "photo.jpg").write_bytes(b"jpeg")
    clients: list[TestClient] = []

    def make(token: str | None) -> TestClient:
        settings = Settings(device="cpu", token=token, data_dir=tmp_path / "data", apk_path=apk)
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


PROTECTED = ["/", "/app/version", "/app.apk", "/captures/20261023-101500-123/photo.jpg"]


@pytest.mark.parametrize("path", PROTECTED)
def test_http_needs_token(make_client, path: str) -> None:
    client = make_client(TOKEN)
    r = client.get(path)
    assert r.status_code == 401 and r.json() == {"detail": "missing or wrong token"}
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert client.get(path, params={"token": "wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, params={"token": TOKEN}).status_code == 200
    assert client.get(path, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_health_stays_open(make_client) -> None:
    r = make_client(TOKEN).get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_websocket_wrong_or_missing_token_closes_4401(make_client) -> None:
    client = make_client(TOKEN)
    for url in ("/ws", "/ws?token=wrong"):
        with client.websocket_connect(url) as ws, pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4401


def test_websocket_with_token(make_client) -> None:
    client = make_client(TOKEN)
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        assert ws.receive_json()["type"] == "hello"
    with client.websocket_connect("/ws", headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        assert ws.receive_json()["type"] == "hello"


def test_no_token_configured_everything_open(make_client) -> None:
    client = make_client(None)
    for path in PROTECTED:
        assert client.get(path).status_code == 200, path
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_landing_page_passes_token_to_apk_link(make_client) -> None:
    page = make_client(TOKEN).get("/", params={"token": TOKEN}).text
    assert f"/app.apk?token={TOKEN}" in page


def test_app_version_from_sidecar(make_client) -> None:
    info = make_client(TOKEN).get("/app/version", params={"token": TOKEN}).json()
    assert (info["application_id"], info["version_code"], info["version_name"]) == ("org.example.gv", 5, "0.5.0.1")


@pytest.mark.parametrize("path", ["/captures/..%2F..%2Fapp/glass-vision.apk", "/captures/.hidden/photo.jpg"])
def test_captures_no_traversal(make_client, path: str) -> None:
    assert make_client(None).get(path).status_code == 404


def test_request_token_sources() -> None:
    def scope(query: bytes = b"", headers: list[tuple[bytes, bytes]] | None = None) -> dict:
        return {"query_string": query, "headers": headers or []}

    assert request_token(scope(b"token=abc&x=1")) == "abc"
    assert request_token(scope(headers=[(b"authorization", b"Bearer xyz")])) == "xyz"
    assert request_token(scope(headers=[(b"authorization", b"bearer  xyz ")])) == "xyz"
    assert request_token(scope(headers=[(b"authorization", b"Basic xyz")])) is None
    assert request_token(scope()) is None


def test_token_ok() -> None:
    assert token_ok(None, None) and token_ok(None, "anything")
    assert token_ok("a", "a") and not token_ok("a", "b") and not token_ok("a", None) and not token_ok("a", "")


def test_log_redaction() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s "GET %s"', ("1.2.3.4", "/ws?token=abc&x=1"), None
    )
    RedactToken().filter(record)
    assert record.getMessage() == '1.2.3.4 "GET /ws?token=***&x=1"'
