"""FastAPI app: WebSocket frame stream → recognition pipelines → tracks. Protocol: docs/PROTOCOL.md (v3).

Settings come from `glass_vision.config` (GV_* environment / `server/config.env`).
"""

from __future__ import annotations

import asyncio
import hmac
import html
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import captures, session, text_read
from .apk_info import read_apk_info
from .config import REPO_ROOT, Settings, resolve_device, settings
from .pipelines import available_pipelines
from .session import PipelineManager, StreamSession

VERSION_FILE = REPO_ROOT / "VERSION"
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.is_file() else "0.0.0"
WS_CLOSE_UNAUTHORIZED = 4401
OPEN_PATHS = frozenset({"/health"})  # reachable without the token (no data beyond status)

log = logging.getLogger("glass_vision")


# --- auth ------------------------------------------------------------------------------------------------------


def request_token(scope: Scope) -> str | None:
    """The token a request carries: `?token=…` or `Authorization: Bearer …` (query wins)."""
    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if query.get("token"):
        return query["token"][0]
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            scheme, _, credentials = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and credentials.strip():
                return credentials.strip()
    return None


def token_ok(expected: str | None, given: str | None) -> bool:
    if expected is None:
        return True
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


class TokenAuth:
    """ASGI middleware for docs/PROTOCOL.md "Auth": with a token configured, every HTTP request except OPEN_PATHS
    needs it (else 401) and every WebSocket is accepted and immediately closed with 4401 — accepting first so the
    client sees the close code instead of a bare handshake failure."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.token is None
            or scope["type"] not in ("http", "websocket")
            or scope["path"] in OPEN_PATHS
            or token_ok(self.token, request_token(scope))
        ):
            await self.app(scope, receive, send)
            return
        client = scope.get("client") or ("?", 0)
        log.warning("rejected %s %s from %s: missing or wrong token", scope["type"], scope["path"], client[0])
        if scope["type"] == "websocket":
            if (await receive())["type"] == "websocket.connect":
                await send({"type": "websocket.accept"})
                await send({"type": "websocket.close", "code": WS_CLOSE_UNAUTHORIZED, "reason": "unauthorized"})
            return
        body = b'{"detail":"missing or wrong token"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b'Bearer realm="glass-vision"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class RedactToken(logging.Filter):
    """Keep `?token=…` out of uvicorn's access/handshake log lines."""

    PATTERN = re.compile(r"(token=)[^&\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(self.PATTERN.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args)
        if isinstance(record.msg, str):
            record.msg = self.PATTERN.sub(r"\1***", record.msg)
        return True


for _name in ("uvicorn.access", "uvicorn.error"):
    logging.getLogger(_name).addFilter(RedactToken())


# --- app -------------------------------------------------------------------------------------------------------


class State:
    manager: PipelineManager
    device_label: str


state = State()


def device_label(device: str) -> str:
    if device.startswith("cuda"):
        import torch

        return f"{device} ({torch.cuda.get_device_name(device)})"
    return f"{device} (no GPU: slow)"


def create_app(config: Settings) -> FastAPI:
    session.IDLE_UNLOAD_S = config.idle_unload_s  # read by PipelineManager.unload_idle_forever on every pass

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        device = resolve_device(config.device)
        state.device_label = device_label(device)
        state.manager = PipelineManager(available_pipelines(), device)
        log.info(
            "glass-vision %s on %s, pipelines: %s; auth %s; models %s; data %s; APK %s",
            VERSION,
            state.device_label,
            ", ".join(state.manager.handles),
            "token" if config.token else "OFF (trusted network only)",
            config.models_dir,
            config.data_dir,
            config.apk_path,
        )
        # Load default-on pipelines now so the first client doesn't wait.
        for pid, handle in state.manager.handles.items():
            if handle.info.default_enabled:
                await state.manager.ensure_ready(pid)
        unloader = asyncio.create_task(state.manager.unload_idle_forever())
        yield
        unloader.cancel()
        await state.manager.shutdown()

    api = FastAPI(title="glass-vision", lifespan=lifespan)
    api.add_middleware(TokenAuth, token=config.token)

    @api.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "version": VERSION,
            "device": state.device_label,
            "pipelines": {pid: h.status for pid, h in state.manager.handles.items()},
        }

    @api.get("/app.apk")
    async def apk() -> FileResponse:
        if not config.apk_path.is_file():
            raise HTTPException(404, f"no APK at {config.apk_path.name}: see README (Updating the app)")
        return FileResponse(
            config.apk_path, media_type="application/vnd.android.package-archive", filename="glass-vision.apk"
        )

    @api.get("/app/version")
    async def apk_version() -> dict[str, object]:
        """What the in-app updater compares against: the APK currently served at /app.apk."""
        try:
            return await asyncio.to_thread(read_apk_info, config.apk_path)
        except FileNotFoundError as exc:
            raise HTTPException(404, "no APK to distribute") from exc
        except ValueError as exc:
            log.warning("bad APK metadata: %s", exc)
            raise HTTPException(500, "bad APK metadata") from exc

    api.include_router(captures.router)  # GET /captures/<id>/<file> (`command_result.files`), behind TokenAuth
    api.include_router(text_read.create_router(lambda: state.manager))  # /text/* ("Read text"), behind TokenAuth

    @api.get("/", response_class=HTMLResponse)
    async def index(token: str | None = None) -> str:
        rows = "".join(
            f"<li><b>{html.escape(h.info.name)}</b> — {h.status}</li>" for h in state.manager.handles.values()
        )
        apk_link = "/app.apk" + (f"?token={quote(token)}" if token else "")
        return (
            "<!doctype html><meta name=viewport content='width=device-width'>"
            "<title>glass-vision</title><body style='font-family:sans-serif;padding:2em'>"
            f"<h1>glass-vision {html.escape(VERSION)}</h1><p>Device: <b>{html.escape(state.device_label)}</b></p>"
            f"<ul>{rows}</ul>"
            f"<p><a href='{html.escape(apk_link)}' style='font-size:1.5em'>Download the Android app (APK)</a></p>"
            "<p>WebSocket: <code>/ws</code> (protocol v3)</p></body>"
        )

    @api.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        log.info("client connected %s", client)
        try:
            await StreamSession(ws, state.manager, VERSION, state.device_label).run()
        finally:
            log.info("client disconnected %s", client)

    return api


app = create_app(settings)
