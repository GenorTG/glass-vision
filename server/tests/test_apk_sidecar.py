"""Release APKs (no Gradle output-metadata.json) carry their version in a `<apk>.json` sidecar."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from glass_vision.apk_info import read_apk_info, sidecar_path


def write_release(directory: Path, content: bytes, version_code: int, version_name: str) -> Path:
    apk = directory / "glass-vision.apk"
    apk.write_bytes(content)
    sidecar_path(apk).write_text(
        json.dumps({"application_id": "org.example.gv", "version_code": version_code, "version_name": version_name})
    )
    return apk


def test_sidecar_path() -> None:
    assert sidecar_path(Path("/d/glass-vision.apk")) == Path("/d/glass-vision.apk.json")


def test_reads_sidecar(tmp_path: Path) -> None:
    apk = write_release(tmp_path, b"release-apk", 5000007, "0.5.0.7")
    assert read_apk_info(apk) == {
        "application_id": "org.example.gv",
        "version_code": 5000007,
        "version_name": "0.5.0.7",
        "size": 11,
        "sha256": hashlib.sha256(b"release-apk").hexdigest(),
        "url": "/app.apk",
    }


def test_sidecar_wins_over_gradle_metadata(tmp_path: Path) -> None:
    apk = write_release(tmp_path, b"x", 2, "0.0.0.2")
    (tmp_path / "output-metadata.json").write_text(
        json.dumps({"applicationId": "a", "elements": [{"outputFile": apk.name, "versionCode": 1, "versionName": "1"}]})
    )
    assert read_apk_info(apk)["version_code"] == 2


def test_replaced_sidecar_is_picked_up(tmp_path: Path) -> None:
    apk = write_release(tmp_path, b"same-bytes", 1, "0.0.0.1")
    read_apk_info(apk)
    meta = sidecar_path(apk)
    meta.write_text(json.dumps({"application_id": "org.example.gv", "version_code": 9, "version_name": "0.0.0.9"}))
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert read_apk_info(apk)["version_code"] == 9


def test_bad_sidecar(tmp_path: Path) -> None:
    apk = tmp_path / "glass-vision.apk"
    apk.write_bytes(b"x")
    sidecar_path(apk).write_text(json.dumps({"version_code": "seven"}))
    with pytest.raises(ValueError, match="application_id"):
        read_apk_info(apk)


def test_no_metadata_at_all(tmp_path: Path) -> None:
    apk = tmp_path / "glass-vision.apk"
    apk.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="glass-vision.apk.json"):
        read_apk_info(apk)
