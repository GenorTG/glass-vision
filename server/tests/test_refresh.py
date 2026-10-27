"""Idle memory release: `memory.release`, PipelineManager idle bookkeeping, and the in-place idle restart of a real
server process (`glass_vision/refresh.py`): same PID, listening socket kept, no request refused."""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from glass_vision import memory, refresh, session
from glass_vision.pipelines.base import Annotation, Pipeline, PipelineInfo, RunContext
from glass_vision.session import PipelineManager

SERVER_DIR = Path(__file__).resolve().parents[1]


class Dummy(Pipeline):
    info = PipelineInfo(id="dummy", name="Dummy", description="test")
    unloads = 0

    def load(self, device: str) -> None:
        self.weights = np.ones(2**20, np.float32)

    def unload(self) -> None:
        Dummy.unloads += 1
        self.weights = None

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        return []


class Broken(Dummy):
    info = PipelineInfo(id="broken", name="Broken", description="test")

    def load(self, device: str) -> None:
        raise RuntimeError("no weights")


def test_release_and_measure_without_cuda() -> None:
    memory.release()  # must not create a CUDA context when torch never touched the GPU
    assert memory.rss_mib() > 0
    assert "RSS" in memory.summary()


def test_activity_quiet() -> None:
    activity = refresh.Activity()
    activity.enter()
    assert not activity.quiet(0.0)  # something is open
    activity.exit()
    now = activity.last_active
    assert not activity.quiet(5.0, now=now + 1)
    assert activity.quiet(5.0, now=now + 5)


def test_manager_idle_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        manager = PipelineManager({"dummy": Dummy, "broken": Broken}, "cpu")
        assert manager.idle() and not manager.used
        assert await manager.ensure_ready("dummy")
        assert manager.used and not manager.idle()
        assert not await manager.ensure_ready("broken")
        assert manager.handles["broken"].status == "error" and manager.handles["broken"].instance is None

        monkeypatch.setattr(session, "IDLE_UNLOAD_S", 0)
        before = Dummy.unloads
        await manager.unload_idle()
        assert Dummy.unloads == before + 1
        assert manager.handles["dummy"].status == "not_loaded"
        assert manager.idle()  # an errored pipeline holds nothing either

    asyncio.run(scenario())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_idle_restart_keeps_pid_and_socket(tmp_path: Path) -> None:
    port = free_port()
    log_path = tmp_path / "server.log"
    env = os.environ | {
        "GV_ENV_FILE": "",  # ignore server/config.env (host, token of this machine)
        "GV_HOST": "127.0.0.1",
        "GV_PORT": str(port),
        "GV_TOKEN": "",
        "GV_IDLE_UNLOAD_S": "1",
        "GV_DATA_DIR": str(tmp_path / "data"),
        "GV_APK_PATH": str(tmp_path / "none.apk"),
    }
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "glass_vision"], cwd=SERVER_DIR, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
    base = f"http://127.0.0.1:{port}"
    failures: list[str] = []
    stop = threading.Event()

    def health() -> dict:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            return json.loads(r.read())

    def poll() -> None:
        # Slower than the 1 s quiet window, so the restart can happen; every poll lands during or around it.
        while not stop.is_set():
            try:
                health()
            except (urllib.error.URLError, OSError) as exc:
                failures.append(repr(exc))
            stop.wait(1.3)

    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                assert health()["pipelines"]["objects"] == "ready"  # default-on pipeline preloaded at startup
                break
            except (urllib.error.URLError, OSError):
                assert proc.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.5)
        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        while "restarted after idle" not in log_path.read_text():
            assert proc.poll() is None, log_path.read_text()
            assert time.monotonic() < deadline, log_path.read_text()
            time.sleep(0.5)
        time.sleep(2)  # a few more polls against the new process
        stop.set()
        poller.join()

        assert proc.poll() is None  # exec kept the same process (and PID): systemd sees no restart
        assert failures == [], failures
        status = health()["pipelines"]
        assert set(status.values()) == {"not_loaded"}  # no preload after an idle restart
        log = log_path.read_text()
        assert re.search(r"everything idle: restarting", log)
        assert "Traceback" not in log, log
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
