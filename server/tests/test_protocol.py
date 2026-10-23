import pytest

from glass_vision.pipelines.base import OptionSpec, PipelineInfo
from glass_vision.protocol import HEADER, ProtocolError, StreamConfig, pack_frame, unpack_frame

INFOS = {
    "objects": PipelineInfo("objects", "Objects", "", default_enabled=True, uses_global_conf=True),
    "open_vocab": PipelineInfo(
        "open_vocab",
        "Find by text",
        "",
        options=[OptionSpec("prompt", "Prompt", "text", "person"), OptionSpec("strict", "Strict", "bool", False)],
    ),
    "faces": PipelineInfo(
        "faces", "Faces", "", options=[OptionSpec("threshold", "T", "number", 0.4, min=0.2, max=0.8, step=0.05)]
    ),
    "text": PipelineInfo(
        "text", "Text", "", options=[OptionSpec("languages", "L", "choice", "en", choices=["en", "en+pl"])]
    ),
}


def test_header_is_16_bytes_big_endian() -> None:
    data = pack_frame(0x01020304, 0x0A0B0C0D0E0F1011, b"\xff\xd8jpeg")
    assert HEADER.size == 16
    assert data[:16] == bytes.fromhex("010203040a0b0c0d0e0f101100000000")
    frame = unpack_frame(data)
    assert (frame.seq, frame.client_ts, frame.jpeg) == (0x01020304, 0x0A0B0C0D0E0F1011, b"\xff\xd8jpeg")


def test_short_frame_rejected() -> None:
    with pytest.raises(ProtocolError):
        unpack_frame(b"\x00" * 16)


def test_defaults_come_from_pipeline_infos() -> None:
    config = StreamConfig(INFOS)
    assert config.enabled() == ["objects"]
    assert config.pipelines["open_vocab"].options == {"prompt": "person", "strict": False}
    assert (config.conf, config.imgsz, config.hold_ms) == (0.35, 640, 300)


def test_config_merges_globals_and_pipelines() -> None:
    config = StreamConfig(INFOS)
    config.apply({"type": "config", "conf": 0.5, "hold_ms": 0})
    config.apply({"pipelines": {"open_vocab": {"enabled": True, "options": {"prompt": "red mug, keys"}}}})
    config.apply(
        {"pipelines": {"faces": {"enabled": True, "options": {"threshold": 0.55}}, "objects": {"enabled": False}}}
    )
    assert (config.conf, config.imgsz, config.hold_ms) == (0.5, 640, 0)
    assert config.enabled() == ["open_vocab", "faces"]
    assert config.pipelines["open_vocab"].options == {"prompt": "red mug, keys", "strict": False}
    assert config.pipelines["faces"].options["threshold"] == 0.55


@pytest.mark.parametrize(
    "bad",
    [
        {"conf": 1.5},
        {"conf": "x"},
        {"imgsz": 650},
        {"imgsz": 640.0},
        {"hold_ms": True},
        {"pipelines": {"nope": {"enabled": True}}},
        {"pipelines": {"faces": {"enabled": "yes"}}},
        {"pipelines": {"faces": {"options": {"threshold": 0.9}}}},
        {"pipelines": {"faces": {"options": {"unknown": 1}}}},
        {"pipelines": {"text": {"options": {"languages": "de"}}}},
        {"pipelines": {"open_vocab": {"options": {"strict": 1}}}},
        {"pipelines": []},
    ],
)
def test_invalid_config_is_rejected_atomically(bad: dict) -> None:
    config = StreamConfig(INFOS)
    # A valid part in the same message must not be applied either.
    with pytest.raises(ProtocolError):
        config.apply({"conf": 0.6, **bad})
    assert config.conf == 0.35
    assert config.enabled() == ["objects"]
