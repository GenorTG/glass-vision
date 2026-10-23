"""HTTP side of "Read text" (docs/PROTOCOL.md, v3.2): read one high-resolution still, translate lines, list targets.

The work runs on the `text` pipeline instance the live streams use (`PipelineManager.call`): same models, same lock,
lazy load and idle unload. Auth: these routes sit behind `app.TokenAuth` like every other route. Everything a client
sends is validated before the model is touched, so a bad request is a 400 even while OCR can't load (503).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from .pipelines.base import OptionSpec
from .session import PipelineManager, PipelineUnavailable
from .translate import DEFAULT_MODEL, LANGUAGES, MODELS, TRANSLATION_DIR

log = logging.getLogger("glass_vision")

MAX_STILL_BYTES = 40 * 2**20  # a 50 MP phone JPEG is ~20 MB
MAX_LINES = 500  # POST /text/translate
MAX_LINE_CHARS = 1000
CODES = frozenset(LANGUAGES.values())
JPEG_MAGIC = b"\xff\xd8\xff"


def _language_code(value: str, what: str) -> str:
    if value not in CODES:
        raise HTTPException(400, f"{what}: unknown language code {value!r} (GET /text/languages)")
    return value


def best_installed_model() -> str:
    """The largest downloaded translation model. A read is one request, not a per-frame lane: the 1.2B model is as
    fast as the 418M one once loaded (~0.3 s per sign on the 3090) and fixes most one-word signs
    ("AUSGANG" → "Exit" instead of "output", "Notausgang 2" → "Emergency Exit 2" instead of "Emergency 2")."""
    installed = [name for name, spec in MODELS.items() if (TRANSLATION_DIR / spec.dir / "config.json").is_file()]
    return installed[-1] if installed else DEFAULT_MODEL


def _model(value: str | None) -> str:
    if value is None:
        return best_installed_model()
    if value not in MODELS:
        raise HTTPException(400, f"model must be one of {list(MODELS)}")
    return value


def _decode_jpeg(body: bytes) -> np.ndarray:
    if len(body) > MAX_STILL_BYTES:
        raise HTTPException(413, f"image larger than {MAX_STILL_BYTES // 2**20} MB")
    if not body.startswith(JPEG_MAGIC):
        raise HTTPException(400, "body must be a JPEG image")
    # IMREAD_COLOR applies the EXIF orientation, so a camera JPEG with an orientation tag is read upright too.
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "cannot decode JPEG")
    return image


def create_router(manager: Callable[[], PipelineManager]) -> APIRouter:
    """`manager` returns the app's PipelineManager (created in the app's lifespan, after the routes)."""
    router = APIRouter(prefix="/text")

    def text_option(key: str) -> OptionSpec:
        handle = manager().handles.get("text")
        if handle is None:
            raise HTTPException(503, "text reading is not available on this server (OCR pipeline missing)")
        return next(o for o in handle.info.options if o.key == key)

    async def on_text_pipeline(fn: Callable[[Any], Any]) -> Any:
        """fn(TextPipeline instance) under its lock (pipelines/text.py is not imported here: it pulls in torch, and
        a server without OCR must still start)."""
        try:
            return await manager().call("text", fn)
        except PipelineUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.post("/read")
    async def read(
        request: Request,
        languages: str | None = None,
        translate_to: str | None = None,
        translate_from: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        option = text_option("languages")
        try:
            group = option.default if languages is None else option.coerce(languages)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        targets = list(
            dict.fromkeys(
                _language_code(code.strip(), "translate_to") for code in (translate_to or "").split(",") if code.strip()
            )
        )
        source = None if translate_from is None else _language_code(translate_from, "translate_from")
        model_name = _model(model)
        image = _decode_jpeg(await request.body())
        return await on_text_pipeline(lambda p: p.read_still(image, group, targets, source, model_name))

    @router.post("/translate")
    async def translate(request: Request) -> dict[str, Any]:
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        lines = body.get("lines")
        if (
            not isinstance(lines, list)
            or len(lines) > MAX_LINES
            or not all(isinstance(line, str) and len(line) <= MAX_LINE_CHARS for line in lines)
        ):
            raise HTTPException(400, f"lines must be a list of at most {MAX_LINES} strings")
        sources = body.get("from")
        if sources is not None:
            if not isinstance(sources, list) or len(sources) != len(lines):
                raise HTTPException(400, "from must be a list with one entry (code or null) per line")
            if not all(s is None or isinstance(s, str) for s in sources):
                raise HTTPException(400, "from entries must be language codes or null")
            sources = [None if s is None else _language_code(s, "from") for s in sources]
        to = body.get("to")
        if not isinstance(to, str):
            raise HTTPException(400, "to must be a language code")
        target = _language_code(to, "to")
        model_name = _model(body.get("model"))
        text_option("languages")  # 503 without an OCR pipeline: the translator lives in it
        if not lines:
            return {"to": target, "translations": []}
        try:
            translations = await on_text_pipeline(lambda p: p.translate_lines(lines, target, sources, model_name))
        except HTTPException:
            raise
        except Exception as exc:  # translation model missing or failing: nothing to return
            log.exception("text: translation to %s failed", target)
            raise HTTPException(503, f"translation unavailable: {exc}") from exc
        return {"to": target, "translations": translations}

    @router.get("/languages")
    async def languages_list() -> dict[str, Any]:
        return {"languages": [{"code": code, "name": name} for name, code in sorted(LANGUAGES.items())]}

    return router
