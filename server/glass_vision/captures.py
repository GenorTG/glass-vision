"""Pose captures on disk (`DATA_DIR/captures/<id>/`) and the HTTP route that serves them (docs/PROTOCOL.md, `files`).

A capture id is a local timestamp `YYYYMMDD-HHMMSS-mmm`; the route only serves ids of that shape and the file names
in `CAPTURE_FILES`, so no client-supplied string ever becomes a path component unchecked.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .pipelines import base

FileKind = Literal["photo", "openpose", "json", "overlay"]


@dataclass(frozen=True)
class CaptureFile:
    kind: FileKind
    media_type: str


# Everything a capture can contain, in the order it's listed to the client.
CAPTURE_FILES: dict[str, CaptureFile] = {
    "photo.jpg": CaptureFile("photo", "image/jpeg"),
    "openpose_body.png": CaptureFile("openpose", "image/png"),
    "openpose_full.png": CaptureFile("openpose", "image/png"),
    "pose_keypoints.json": CaptureFile("json", "application/json"),
    "pose_on_photo.jpg": CaptureFile("overlay", "image/jpeg"),
}
CAPTURE_ID_RE = re.compile(r"\d{8}-\d{6}-\d{3}", re.ASCII)


def captures_root() -> Path:
    return base.DATA_DIR / "captures"  # read at call time: tests point DATA_DIR elsewhere


def new_capture() -> tuple[str, Path]:
    """Create an empty capture directory named after the current local time; returns (id, path)."""
    root = captures_root()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now(UTC).astimezone()  # the server's local time, as the id is read by people
        capture_id = f"{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}"
        path = root / capture_id
        try:
            path.mkdir()
            return capture_id, path
        except FileExistsError:  # two captures in the same millisecond
            time.sleep(0.001)


def file_entry(capture_id: str, name: str) -> dict[str, str]:
    return {"name": name, "kind": CAPTURE_FILES[name].kind, "url": f"/captures/{capture_id}/{name}"}


router = APIRouter()


@router.get("/captures/{capture_id}/{name}")
def get_capture_file(capture_id: str, name: str) -> FileResponse:
    spec = CAPTURE_FILES.get(name)
    if spec is None or not CAPTURE_ID_RE.fullmatch(capture_id):
        raise HTTPException(status_code=404)
    path = captures_root() / capture_id / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type=spec.media_type)
