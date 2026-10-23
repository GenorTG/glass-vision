"""Settings: env file parsing, env-over-file precedence, defaults, validation, APK path fallback."""

from pathlib import Path

import pytest

from glass_vision import config
from glass_vision.config import Settings, load_settings, parse_env_file


def test_parse_env_file() -> None:
    text = """
# comment
GV_HOST=192.168.1.5
export GV_PORT = 9000
GV_TOKEN="abc def"
GV_DEVICE='cpu'
GV_EMPTY=
"""
    assert parse_env_file(text) == {
        "GV_HOST": "192.168.1.5",
        "GV_PORT": "9000",
        "GV_TOKEN": "abc def",
        "GV_DEVICE": "cpu",
        "GV_EMPTY": "",
    }


def test_parse_env_file_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_env_file("GV_HOST=x\nthis is not a setting\n")


def test_defaults_without_env_or_file(tmp_path: Path) -> None:
    s = load_settings({}, env_file=tmp_path / "missing.env")
    assert (s.host, s.port, s.token, s.device, s.idle_unload_s) == ("127.0.0.1", 8770, None, "auto", 600.0)
    assert s.models_dir == config.SERVER_ROOT / "models"
    assert s.data_dir == config.SERVER_ROOT / "data"


def test_env_beats_file_and_empty_means_unset(tmp_path: Path) -> None:
    env_file = tmp_path / "config.env"
    env_file.write_text("GV_HOST=10.0.0.2\nGV_PORT=9001\nGV_TOKEN=from-file\nGV_IDLE_UNLOAD_S=30\n")
    s = load_settings({"GV_PORT": "9002", "GV_TOKEN": "", "OTHER": "x"}, env_file=env_file)
    assert (s.host, s.port, s.token, s.idle_unload_s) == ("10.0.0.2", 9002, None, 30.0)


def test_env_file_override_via_gv_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "other.env"
    env_file.write_text("GV_TOKEN=t0k3n\n")
    assert load_settings({"GV_ENV_FILE": str(env_file)}).token == "t0k3n"
    assert load_settings({"GV_ENV_FILE": ""}).token is None  # empty = no file at all


def test_dirs_and_apk_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DEV_APK", tmp_path / "no-dev-build.apk")
    s = load_settings({"GV_MODELS_DIR": str(tmp_path / "m"), "GV_DATA_DIR": str(tmp_path / "d")}, env_file=None)
    assert (s.models_dir, s.data_dir) == (tmp_path / "m", tmp_path / "d")
    assert s.apk_path == tmp_path / "d" / "app" / "glass-vision.apk"
    explicit = load_settings({"GV_APK_PATH": str(tmp_path / "x.apk")}, env_file=None)
    assert explicit.apk_path == tmp_path / "x.apk"


def test_apk_path_falls_back_to_dev_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = tmp_path / "app-debug.apk"
    dev.write_bytes(b"apk")
    monkeypatch.setattr(config, "DEV_APK", dev)
    data = tmp_path / "data"
    assert load_settings({"GV_DATA_DIR": str(data)}, env_file=None).apk_path == dev
    installed = data / "app" / "glass-vision.apk"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"release")
    assert load_settings({"GV_DATA_DIR": str(data)}, env_file=None).apk_path == installed  # installed one wins


@pytest.mark.parametrize(
    ("key", "value"),
    [("GV_PORT", "http"), ("GV_PORT", "70000"), ("GV_DEVICE", "tpu"), ("GV_IDLE_UNLOAD_S", "soon")],
)
def test_bad_values(key: str, value: str) -> None:
    with pytest.raises(ValueError, match=key):
        load_settings({key: value}, env_file=None)


def test_resolve_device_explicit_passthrough() -> None:
    assert config.resolve_device("cpu") == "cpu"
    assert config.resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_is_usable() -> None:
    import torch

    device = config.resolve_device("auto")
    assert device in ("cpu", "cuda:0")
    (torch.ones(2, device=device) * 2).sum().item()  # whatever auto picked actually runs


def test_settings_is_frozen() -> None:
    with pytest.raises(AttributeError):
        Settings().port = 1  # type: ignore[misc]
