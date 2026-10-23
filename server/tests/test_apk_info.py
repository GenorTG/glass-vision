import hashlib
import json
import os
from pathlib import Path

import pytest

from glass_vision.apk_info import read_apk_info


def write_build(directory: Path, content: bytes, version_code: int, version_name: str) -> Path:
    apk = directory / "app-debug.apk"
    apk.write_bytes(content)
    (directory / "output-metadata.json").write_text(
        json.dumps(
            {
                "applicationId": "pl.genor.glassvision",
                "elements": [{"versionCode": version_code, "versionName": version_name, "outputFile": "app-debug.apk"}],
            }
        )
    )
    return apk


def test_reads_version_and_hash(tmp_path: Path) -> None:
    apk = write_build(tmp_path, b"apk-v1", 3000001, "0.3.0.1")
    assert read_apk_info(apk) == {
        "application_id": "pl.genor.glassvision",
        "version_code": 3000001,
        "version_name": "0.3.0.1",
        "size": 6,
        "sha256": hashlib.sha256(b"apk-v1").hexdigest(),
        "url": "/app.apk",
    }


def test_rebuild_is_picked_up(tmp_path: Path) -> None:
    apk = write_build(tmp_path, b"apk-v1", 3000001, "0.3.0.1")
    read_apk_info(apk)
    write_build(tmp_path, b"apk-v2!", 3000002, "0.3.0.2")
    os.utime(apk, ns=(apk.stat().st_atime_ns, apk.stat().st_mtime_ns + 1_000_000))
    info = read_apk_info(apk)
    assert (info["version_code"], info["sha256"]) == (3000002, hashlib.sha256(b"apk-v2!").hexdigest())


def test_missing_apk(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_apk_info(tmp_path / "app-debug.apk")
