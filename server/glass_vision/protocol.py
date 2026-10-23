"""Wire format, see docs/PROTOCOL.md (v3)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from .pipelines.base import PipelineInfo

PROTOCOL_VERSION = 3
HEADER = struct.Struct(">IqI")  # seq, client_ts (ms), reserved
CONF_RANGE = (0.1, 0.95)  # floor = tracking.LOW_CONF: YOLO-family pipelines never return less
IMGSZ_RANGE = (320, 1280)
HOLD_MS_RANGE = (0, 2000)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Frame:
    seq: int
    client_ts: int
    jpeg: bytes


def pack_frame(seq: int, client_ts: int, jpeg: bytes) -> bytes:
    return HEADER.pack(seq, client_ts, 0) + jpeg


def unpack_frame(data: bytes) -> Frame:
    if len(data) <= HEADER.size:
        raise ProtocolError(f"frame too short ({len(data)} bytes)")
    seq, client_ts, _ = HEADER.unpack_from(data)
    return Frame(seq, client_ts, data[HEADER.size :])


@dataclass
class PipelineConfig:
    enabled: bool
    options: dict[str, Any]


@dataclass
class StreamConfig:
    """One connection's settings. `pipelines` is filled from the server's pipeline infos (defaults)."""

    infos: dict[str, PipelineInfo]
    conf: float = 0.35
    imgsz: int = 640
    hold_ms: int = 300
    pipelines: dict[str, PipelineConfig] = field(init=False)

    def __post_init__(self) -> None:
        self.pipelines = {
            pid: PipelineConfig(info.default_enabled, info.default_options()) for pid, info in self.infos.items()
        }

    def enabled(self) -> list[str]:
        return [pid for pid, cfg in self.pipelines.items() if cfg.enabled]

    def apply(self, message: dict[str, Any]) -> None:
        """Merge a client config message. Validates everything first, so an invalid message changes nothing."""
        staged = _Staged(self.conf, self.imgsz, self.hold_ms)
        if "conf" in message:
            conf = message["conf"]
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not _within(conf, CONF_RANGE):
                raise ProtocolError(f"conf must be in {CONF_RANGE}")
            staged.conf = float(conf)
        if "imgsz" in message:
            imgsz = message["imgsz"]
            if isinstance(imgsz, bool) or not isinstance(imgsz, int) or not _within(imgsz, IMGSZ_RANGE) or imgsz % 32:
                raise ProtocolError(f"imgsz must be a multiple of 32 in {IMGSZ_RANGE}")
            staged.imgsz = imgsz
        if "hold_ms" in message:
            hold_ms = message["hold_ms"]
            if isinstance(hold_ms, bool) or not isinstance(hold_ms, int) or not _within(hold_ms, HOLD_MS_RANGE):
                raise ProtocolError(f"hold_ms must be an integer in {HOLD_MS_RANGE}")
            staged.hold_ms = hold_ms

        pipeline_updates: dict[str, tuple[bool | None, dict[str, Any]]] = {}
        raw_pipelines = message.get("pipelines", {})
        if not isinstance(raw_pipelines, dict):
            raise ProtocolError("pipelines must be an object")
        for pid, raw in raw_pipelines.items():
            info = self.infos.get(pid)
            if info is None:
                raise ProtocolError(f"unknown pipeline {pid!r}")
            if not isinstance(raw, dict):
                raise ProtocolError(f"pipelines.{pid} must be an object")
            enabled = raw.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                raise ProtocolError(f"pipelines.{pid}.enabled must be true/false")
            raw_options = raw.get("options", {})
            if not isinstance(raw_options, dict):
                raise ProtocolError(f"pipelines.{pid}.options must be an object")
            specs = {o.key: o for o in info.options}
            options: dict[str, Any] = {}
            for key, value in raw_options.items():
                if key not in specs:
                    raise ProtocolError(f"pipelines.{pid}: unknown option {key!r}")
                try:
                    options[key] = specs[key].coerce(value)
                except ValueError as exc:
                    raise ProtocolError(f"pipelines.{pid}: {exc}") from exc
            pipeline_updates[pid] = (enabled, options)

        self.conf, self.imgsz, self.hold_ms = staged.conf, staged.imgsz, staged.hold_ms
        for pid, (enabled, options) in pipeline_updates.items():
            cfg = self.pipelines[pid]
            if enabled is not None:
                cfg.enabled = enabled
            cfg.options.update(options)


@dataclass
class _Staged:
    conf: float
    imgsz: int
    hold_ms: int


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]
