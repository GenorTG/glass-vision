"""Version info of the served APK, for the in-app updater (`GET /app/version`).

Two sources, first match wins:
1. a sidecar `<apk>.json` next to the APK (`glass-vision.apk.json`) — written by `scripts/install.sh` and the release
   workflow: `{"application_id": ..., "version_code": ..., "version_name": ...}` (extra keys are ignored);
2. Gradle's `output-metadata.json` next to the APK (a developer's `./gradlew assembleDebug` output).

The sha256 is always computed from the APK itself, never taken from metadata.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


def sidecar_path(apk_path: Path) -> Path:
    return apk_path.with_name(apk_path.name + ".json")


def read_apk_info(apk_path: Path) -> dict[str, object]:
    """Raises FileNotFoundError if the APK or both metadata sources are missing, ValueError if metadata is bad."""
    stat = apk_path.stat()
    meta = sidecar_path(apk_path)
    meta_mtime = meta.stat().st_mtime_ns if meta.is_file() else 0
    return _read_cached(apk_path, stat.st_mtime_ns, stat.st_size, meta_mtime)


def _version(apk_path: Path) -> tuple[str, int, str]:
    """→ (application_id, version_code, version_name)."""
    sidecar = sidecar_path(apk_path)
    if sidecar.is_file():
        data = json.loads(sidecar.read_text())
        try:
            return str(data["application_id"]), int(data["version_code"]), str(data["version_name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{sidecar}: needs application_id, version_code, version_name ({exc!r})") from exc
    gradle = apk_path.parent / "output-metadata.json"
    if not gradle.is_file():
        raise FileNotFoundError(f"no version metadata for {apk_path}: expected {sidecar.name} or {gradle.name}")
    metadata = json.loads(gradle.read_text())
    element = next((e for e in metadata["elements"] if e["outputFile"] == apk_path.name), None)
    if element is None:
        raise ValueError(f"{gradle}: no element for {apk_path.name}")
    return str(metadata["applicationId"]), int(element["versionCode"]), str(element["versionName"])


@lru_cache(maxsize=4)
def _read_cached(apk_path: Path, mtime_ns: int, size: int, meta_mtime_ns: int) -> dict[str, object]:
    # Keyed on mtime+size (+ sidecar mtime), so a rebuilt or replaced APK is re-hashed exactly once.
    application_id, version_code, version_name = _version(apk_path)
    digest = hashlib.sha256()
    with apk_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "application_id": application_id,
        "version_code": version_code,
        "version_name": version_name,
        "size": size,
        "sha256": digest.hexdigest(),
        "url": "/app.apk",
    }
