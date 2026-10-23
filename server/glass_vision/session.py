"""Pipeline lifecycle (shared) and per-connection streaming sessions. Protocol: docs/PROTOCOL.md (v3).

PipelineManager — one per process: lazy load on first enable, idle unload (frees VRAM), one lock per
pipeline instance (models aren't thread-safe; different pipelines run in parallel threads), status
broadcast to every connected session.

StreamSession — one per WebSocket: latest frame wins; fast pipelines (interval_ms == 0) run inline and
concurrently on every frame; slow ones run in background lanes on the newest frame at most every
interval_ms. Every pipeline has its own TrackSmoother, and every reply carries all enabled pipelines'
current tracks, so slow results never delay fast ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .pipelines.base import Annotation, CommandContext, CommandResult, Pipeline, PipelineInfo, RunContext
from .protocol import PROTOCOL_VERSION, Frame, ProtocolError, StreamConfig, unpack_frame
from .tracking import TrackSmoother, normalize

log = logging.getLogger("glass_vision")

IDLE_UNLOAD_S = 600  # unload a pipeline nobody used for 10 min
ERROR_RETRY_S = 30  # a pipeline that failed to load/run may be retried after this
ASYNC_HOLD_FACTOR = 2.5  # slow pipelines' tracks are held at least this many intervals
# Options that change what a pipeline's labels mean: a change restarts its tracks, else the label vote keeps the old
# kind of label for a while (text `live`: Mark's "text" boxes would linger as "text" after switching to Read live).
RESET_TRACKS_OPTIONS = {("text", "live")}
LOAD_TIMEOUT_S = 120

Listener = Callable[[dict[str, Any]], Awaitable[None]]
T = TypeVar("T")


class PipelineUnavailable(RuntimeError):
    """`PipelineManager.call`: the pipeline isn't registered or could not be loaded (message = why)."""


class PipelineHandle:
    def __init__(self, cls: type[Pipeline]) -> None:
        self.cls = cls
        self.info: PipelineInfo = cls.info
        self.instance: Pipeline | None = None
        self.status = "not_loaded"  # not_loaded | loading | ready | error
        self.message: str | None = None
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.failed_at = 0.0
        self.loading: asyncio.Task[None] | None = None


class PipelineManager:
    def __init__(self, classes: dict[str, type[Pipeline]], device: str) -> None:
        self.device = device
        self.handles = {pid: PipelineHandle(cls) for pid, cls in classes.items()}
        self._listeners: set[Listener] = set()

    def infos(self) -> dict[str, PipelineInfo]:
        return {pid: h.info for pid, h in self.handles.items()}

    def features(self) -> list[str]:
        """`hello.features`: optional server capabilities beyond the pipelines (docs/PROTOCOL.md)."""
        return ["text_read"] if "text" in self.handles else []

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for h in self.handles.values():
            entry = h.info.to_json() | {"status": h.status}
            if h.message:
                entry["message"] = h.message
            out.append(entry)
        return out

    def subscribe(self, listener: Listener) -> None:
        self._listeners.add(listener)

    def unsubscribe(self, listener: Listener) -> None:
        self._listeners.discard(listener)

    async def _set_status(self, h: PipelineHandle, status: str, message: str | None = None) -> None:
        h.status, h.message = status, message
        event = {"type": "pipeline_status", "pipeline": h.info.id, "status": status}
        if message:
            event["message"] = message
        for listener in list(self._listeners):
            try:
                await listener(event)
            except Exception:
                log.debug("status listener failed", exc_info=True)

    def request(self, pid: str) -> bool:
        """Mark as used; start loading if needed. True if ready to run right now."""
        h = self.handles[pid]
        h.last_used = time.monotonic()
        if h.status == "ready":
            return True
        retry_ok = h.status == "error" and time.monotonic() - h.failed_at > ERROR_RETRY_S
        if (h.status == "not_loaded" or retry_ok) and h.loading is None:
            h.loading = asyncio.create_task(self._load(h))
        return False

    async def ensure_ready(self, pid: str) -> bool:
        if self.request(pid):
            return True
        h = self.handles[pid]
        if h.loading is not None:
            await asyncio.wait_for(asyncio.shield(h.loading), LOAD_TIMEOUT_S)
        return h.status == "ready"

    async def _load(self, h: PipelineHandle) -> None:
        await self._set_status(h, "loading")
        t0 = time.perf_counter()
        try:
            instance = h.cls()

            def load() -> None:
                with h.lock:
                    instance.load(self.device)

            await asyncio.to_thread(load)
        except Exception as exc:
            log.exception("pipeline %s failed to load", h.info.id)
            h.failed_at = time.monotonic()
            await self._set_status(h, "error", f"load failed: {exc}")
        else:
            h.instance = instance
            log.info("pipeline %s loaded in %.1fs", h.info.id, time.perf_counter() - t0)
            await self._set_status(h, "ready")
        finally:
            h.loading = None

    async def run(self, pid: str, image: np.ndarray, ctx: RunContext) -> tuple[list[Annotation], float]:
        h = self.handles[pid]
        instance = h.instance
        if instance is None:
            return [], 0.0

        def run() -> tuple[list[Annotation], float]:
            with h.lock:
                t0 = time.perf_counter()
                return instance.run(image, ctx), (time.perf_counter() - t0) * 1000

        try:
            return await asyncio.to_thread(run)
        except Exception as exc:
            log.exception("pipeline %s failed", pid)
            h.failed_at = time.monotonic()
            await self._unload(h)
            await self._set_status(h, "error", f"run failed: {exc}")
            return [], 0.0

    async def command(self, pid: str, action: str, args: dict[str, Any], ctx: CommandContext) -> CommandResult:
        if not await self.ensure_ready(pid):
            h = self.handles[pid]
            return CommandResult(False, f"{h.info.name} is not available: {h.message or h.status}")
        h = self.handles[pid]
        instance = h.instance
        assert instance is not None

        def command() -> CommandResult:
            with h.lock:
                return instance.command(action, args, ctx)

        return await asyncio.to_thread(command)

    async def call(self, pid: str, fn: Callable[[Pipeline], T]) -> T:
        """Run `fn(instance)` in a worker thread under the pipeline's lock, loading it first if needed: HTTP endpoints
        that use the same model instance as the live streams (`glass_vision/text_read.py`). Exceptions from `fn`
        propagate and leave the pipeline loaded (a bad request is not a broken model)."""
        h = self.handles.get(pid)
        if h is None:
            raise PipelineUnavailable(f"pipeline {pid!r} is not available on this server")
        try:
            ready = await self.ensure_ready(pid)
        except TimeoutError as exc:
            raise PipelineUnavailable(f"{h.info.name} is still loading") from exc
        instance = h.instance
        if not ready or instance is None:
            raise PipelineUnavailable(f"{h.info.name} could not be loaded: {h.message or h.status}")

        def call() -> T:
            with h.lock:
                try:
                    return fn(instance)
                finally:
                    h.last_used = time.monotonic()  # a long call must not count as idle time

        return await asyncio.to_thread(call)

    async def _unload(self, h: PipelineHandle) -> None:
        instance, h.instance = h.instance, None
        if instance is None:
            return

        def unload() -> None:
            with h.lock:
                instance.unload()
            if self.device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()

        await asyncio.to_thread(unload)

    async def unload_idle_forever(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for h in self.handles.values():
                if h.status == "ready" and now - h.last_used > IDLE_UNLOAD_S:
                    log.info("unloading idle pipeline %s", h.info.id)
                    await self._unload(h)
                    await self._set_status(h, "not_loaded")

    async def shutdown(self) -> None:
        for h in self.handles.values():
            await self._unload(h)


class StreamSession:
    """One client. Latest frame wins: a frame still waiting when a newer one arrives is answered 'dropped'."""

    def __init__(self, ws: WebSocket, manager: PipelineManager, version: str, device_label: str) -> None:
        self.ws = ws
        self.manager = manager
        self.version = version
        self.device_label = device_label
        self.config = StreamConfig(manager.infos())
        self.trackers: dict[str, TrackSmoother] = {}
        self.pending: Frame | None = None
        self.frame_ready = asyncio.Event()
        self.send_lock = asyncio.Lock()
        # Newest decoded frame: input of slow lanes and commands.
        self.latest_image: np.ndarray | None = None
        self.latest_image_ms = 0.0
        self.latest_image_id = 0
        self.image_events: dict[str, asyncio.Event] = {}
        self.lanes: dict[str, asyncio.Task[None]] = {}
        self.async_raw: dict[str, list[dict[str, Any]]] = {}
        self.async_ms: dict[str, float] = {}
        self.commands: set[asyncio.Task[None]] = set()
        # App builds up to 0.3 (protocol v2) parse track ids as integers and choke on "objects:7", and they can't
        # update themselves until they get working results. Until a client shows it speaks v3 (a config with
        # `pipelines` or `protocol` >= 3; every v3 app sends one right after hello), track ids go out as ints.
        self.legacy_client = True
        self.legacy_ids: dict[str, int] = {}

    # --- plumbing -------------------------------------------------------------------------------

    async def send(self, message: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(message, separators=(",", ":")))

    async def run(self) -> None:
        self.manager.subscribe(self.send)
        await self.send(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "device": self.device_label,
                "version": self.version,
                "features": self.manager.features(),
                "pipelines": self.manager.describe(),
            }
        )
        self.reconcile()
        worker = asyncio.create_task(self.infer_loop())
        try:
            await self.receive_loop()
        except WebSocketDisconnect:
            pass
        finally:
            self.manager.unsubscribe(self.send)
            worker.cancel()
            for lane in self.lanes.values():
                lane.cancel()

    async def receive_loop(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if (data := message.get("bytes")) is not None:
                await self.on_frame(data)
            elif (text := message.get("text")) is not None:
                await self.on_text(text)

    async def on_frame(self, data: bytes) -> None:
        try:
            frame = unpack_frame(data)
        except ProtocolError as exc:
            await self.send({"type": "error", "seq": None, "message": str(exc)})
            return
        if self.pending is not None:
            await self.send({"type": "dropped", "seq": self.pending.seq})
        self.pending = frame
        self.frame_ready.set()

    async def on_text(self, text: str) -> None:
        try:
            message = json.loads(text)
            if not isinstance(message, dict):
                raise ProtocolError("expected a JSON object")
            kind = message.get("type")
            if kind == "config":
                before = {
                    (pid, key): self.config.pipelines[pid].options.get(key)
                    for pid, key in RESET_TRACKS_OPTIONS
                    if pid in self.config.pipelines
                }
                self.config.apply(message)
                for (pid, key), value in before.items():
                    if self.config.pipelines[pid].options.get(key) != value and pid in self.trackers:
                        self.trackers[pid] = TrackSmoother(pid)
                        self.async_raw.pop(pid, None)
                if "pipelines" in message or message.get("protocol", 0) >= PROTOCOL_VERSION:
                    self.legacy_client = False
                self.reconcile()
            elif kind == "command":
                task = asyncio.create_task(self.on_command(message))
                self.commands.add(task)  # keep a reference until done
                task.add_done_callback(self.commands.discard)
            else:
                raise ProtocolError(f"unknown message type {kind!r}")
        except (json.JSONDecodeError, ProtocolError) as exc:
            await self.send({"type": "error", "seq": None, "message": str(exc)})

    async def on_command(self, message: dict[str, Any]) -> None:
        req, pid, action = message.get("req"), message.get("pipeline"), message.get("action")
        args = message.get("args") or {}
        reply: dict[str, Any] = {"type": "command_result", "req": req, "pipeline": pid, "action": action}
        if pid not in self.manager.handles or not isinstance(action, str) or not isinstance(args, dict):
            await self.send(reply | {"ok": False, "message": f"bad command for pipeline {pid!r}"})
            return
        ctx = CommandContext(options=dict(self.config.pipelines[pid].options), frame=self.latest_image)
        try:
            result = await self.manager.command(pid, action, args, ctx)
        except Exception as exc:
            log.exception("command %s.%s failed", pid, action)
            result = CommandResult(False, f"{action} failed: {exc}")
        reply |= {"ok": result.ok, "message": result.message}
        if result.items is not None:
            reply["items"] = result.items
        if result.item_action is not None:
            reply["item_action"] = result.item_action
        if result.files is not None:
            reply["files"] = result.files
        await self.send(reply)

    def reconcile(self) -> None:
        """Bring trackers, background lanes and model loading in line with the config."""
        enabled = set(self.config.enabled())
        for pid in list(self.trackers):
            if pid not in enabled:
                del self.trackers[pid]
                self.async_raw.pop(pid, None)
                self.async_ms.pop(pid, None)
        for pid, lane in list(self.lanes.items()):
            if pid not in enabled:
                lane.cancel()
                del self.lanes[pid]
                del self.image_events[pid]
        for pid in enabled:
            self.trackers.setdefault(pid, TrackSmoother(pid))
            self.manager.request(pid)
            if self.manager.handles[pid].info.interval_ms > 0 and pid not in self.lanes:
                self.image_events[pid] = asyncio.Event()
                self.lanes[pid] = asyncio.create_task(self.slow_lane(pid))

    def run_context(self, pid: str) -> RunContext:
        return RunContext(dict(self.config.pipelines[pid].options), self.config.conf, self.config.imgsz)

    def new_track_conf(self, pid: str) -> float | None:
        return self.config.conf if self.manager.handles[pid].info.uses_global_conf else None

    def hold_ms(self, pid: str) -> float:
        interval = self.manager.handles[pid].info.interval_ms
        if not interval:
            return self.config.hold_ms
        # A slow lane refreshes every max(interval, run time): OCR on a CPU takes ~2.7 s per run, so holding for
        # 2.5 × the 400 ms interval alone would drop every track before its next confirmation and nothing would show.
        period = max(interval, self.async_ms.get(pid, 0.0))
        return max(self.config.hold_ms, ASYNC_HOLD_FACTOR * period)

    def raw_json(self, pid: str, anns: list[Annotation], width: int, height: int) -> list[dict[str, Any]]:
        floor = self.new_track_conf(pid)
        return [normalize(a, width, height, pid) for a in anns if floor is None or a.conf >= floor]

    # --- frame processing -----------------------------------------------------------------------

    async def infer_loop(self) -> None:
        while True:
            await self.frame_ready.wait()
            self.frame_ready.clear()
            frame, self.pending = self.pending, None
            if frame is not None:
                await self.process(frame)

    async def process(self, frame: Frame) -> None:
        t0 = time.perf_counter()
        image = await asyncio.to_thread(cv2.imdecode, np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            await self.send({"type": "error", "seq": frame.seq, "message": "cannot decode JPEG"})
            return
        decode_ms = (time.perf_counter() - t0) * 1000
        height, width = image.shape[:2]
        now_ms = time.monotonic() * 1000
        self.latest_image, self.latest_image_ms = image, now_ms
        self.latest_image_id += 1
        for event in self.image_events.values():
            event.set()

        fast = [
            pid
            for pid in self.config.enabled()
            if self.manager.handles[pid].info.interval_ms == 0 and self.manager.request(pid)
        ]
        results = await asyncio.gather(*(self.manager.run(pid, image, self.run_context(pid)) for pid in fast))

        dets: list[dict[str, Any]] = []
        pipeline_ms: dict[str, float] = {}
        for pid, (anns, ms) in zip(fast, results, strict=True):
            tracker = self.trackers.get(pid)
            if tracker is None:  # disabled while running
                continue
            tracker.update(anns, width, height, self.new_track_conf(pid), now_ms)
            dets += self.raw_json(pid, anns, width, height)
            pipeline_ms[pid] = round(ms, 2)
        for pid, raw in self.async_raw.items():
            dets += raw
            pipeline_ms[pid] = round(self.async_ms.get(pid, 0.0), 2)

        tracks = [
            view.to_json()
            for pid, tracker in self.trackers.items()
            for view in tracker.views(self.hold_ms(pid), now_ms)
        ]
        if self.legacy_client:
            for track in tracks:
                track["id"] = self.legacy_ids.setdefault(track["id"], len(self.legacy_ids) + 1)
        await self.send(
            {
                "type": "detections",
                "seq": frame.seq,
                "client_ts": frame.client_ts,
                "w": width,
                "h": height,
                "decode_ms": round(decode_ms, 2),
                "server_ms": round((time.perf_counter() - t0) * 1000, 2),
                "pipeline_ms": pipeline_ms,
                "dets": dets,
                "tracks": tracks,
            }
        )

    async def slow_lane(self, pid: str) -> None:
        """Background loop of one slow pipeline: newest frame, at most every interval_ms."""
        interval_s = self.manager.handles[pid].info.interval_ms / 1000
        event = self.image_events[pid]
        last_run = 0.0
        while True:
            await event.wait()
            event.clear()
            if not self.manager.request(pid):
                continue
            await asyncio.sleep(max(0.0, last_run + interval_s - time.monotonic()))
            image, image_ms = self.latest_image, self.latest_image_ms
            if image is None:
                continue
            last_run = time.monotonic()
            anns, ms = await self.manager.run(pid, image, self.run_context(pid))
            tracker = self.trackers.get(pid)
            if tracker is None:
                continue
            height, width = image.shape[:2]
            tracker.update(anns, width, height, self.new_track_conf(pid), image_ms)
            self.async_raw[pid] = self.raw_json(pid, anns, width, height)
            self.async_ms[pid] = ms
