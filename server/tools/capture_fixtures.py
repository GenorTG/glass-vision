"""Record real v3 server messages as Android test fixtures (android/app/src/test/resources/*_v3.json).

    ./run tools/capture_fixtures.py [--url ws://127.0.0.1:8770/ws]

Needs the live service. Captures hello, a detections message on zidane.jpg with objects + faces + pose enabled,
and a faces `list` command_result. `ServerFixtureTest` (Android) parses them, so the two sides can't drift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets
from ultralytics.utils import ASSETS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glass_vision.protocol import pack_frame

OUT = Path(__file__).resolve().parents[2] / "android" / "app" / "src" / "test" / "resources"


async def receive(ws: websockets.ClientConnection, kind: str) -> str:
    while True:
        raw = await ws.recv()
        if json.loads(raw)["type"] == kind:
            return raw


async def capture(url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(url, max_size=None) as ws:
        (OUT / "hello_v3.json").write_text(await ws.recv())
        await ws.send(
            json.dumps({"type": "config", "pipelines": {"pose": {"enabled": True}, "faces": {"enabled": True}}})
        )
        jpeg = (ASSETS / "zidane.jpg").read_bytes()
        for seq in range(4):  # a few frames so tracks are confirmed and all pipelines have loaded
            await ws.send(pack_frame(seq, int(time.time() * 1000), jpeg))
            detections = await receive(ws, "detections")
        (OUT / "detections_v3.json").write_text(detections)
        await ws.send(json.dumps({"type": "command", "req": 7, "pipeline": "faces", "action": "list", "args": {}}))
        (OUT / "command_result_v3.json").write_text(await receive(ws, "command_result"))
    print(f"wrote fixtures to {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8770/ws")
    asyncio.run(capture(parser.parse_args().url))


if __name__ == "__main__":
    main()
