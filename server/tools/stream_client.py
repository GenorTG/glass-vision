"""Protocol v3 test client: streams JPEGs at a target FPS with an in-flight window, like the phone app does.

    ./run tools/stream_client.py [--url ws://127.0.0.1:8770/ws] [--token SECRET] [--fps 15] [--seconds 10]
        [--pipelines objects,pose] [--prompt "bus, person"] [--blank-every 3] IMG...

Enables exactly the given pipelines, waits until they report `ready`, streams, then prints per-pipeline
timings and track labels. Exits non-zero if a frame went unanswered, a pipeline failed, or no enabled
pipeline produced a single track.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glass_vision.protocol import pack_frame


async def run(args: argparse.Namespace, images: list[bytes]) -> int:
    wanted = [p for p in args.pipelines.split(",") if p]
    rtts: list[float] = []
    pipeline_ms: dict[str, list[float]] = defaultdict(list)
    counts = Counter({"sent": 0, "detections": 0, "dropped": 0, "error": 0})
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    empty = {"dets": 0, "tracks": 0}  # results with nothing to draw: raw vs smoothed
    failed: list[str] = []
    in_flight = 0

    # The token may also be in the URL (`?token=…`); --token sends it as a header (docs/PROTOCOL.md "Auth").
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else None
    async with websockets.connect(args.url, max_size=None, additional_headers=headers) as ws:
        hello = json.loads(await ws.recv())
        status = {p["id"]: p["status"] for p in hello["pipelines"]}
        print(f"hello: v{hello['version']} protocol {hello['protocol']} on {hello['device']}; pipelines {status}")
        unknown = set(wanted) - set(status)
        if unknown:
            print(f"server has no pipeline(s) {sorted(unknown)}")
            return 1
        config: dict[str, object] = {
            "type": "config",
            "conf": args.conf,
            "imgsz": args.imgsz,
            "hold_ms": args.hold_ms,
            "pipelines": {pid: {"enabled": pid in wanted} for pid in status},
        }
        if args.prompt is not None:
            config["pipelines"]["open_vocab"]["options"] = {"prompt": args.prompt}  # type: ignore[index]
        await ws.send(json.dumps(config))

        # Enabling starts loading; keep a frame flowing meanwhile (also warms the fast pipelines).
        deadline = time.monotonic() + 180
        while any(status[p] != "ready" for p in wanted):
            if time.monotonic() > deadline:
                print(f"timeout waiting for pipelines: {status}")
                return 1
            await ws.send(pack_frame(0, int(time.time() * 1000), images[0]))
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 2))
            except TimeoutError:
                continue
            if msg["type"] == "pipeline_status":
                status[msg["pipeline"]] = msg["status"]
                print(f"  {msg['pipeline']}: {msg['status']} {msg.get('message', '')}")
                if msg["status"] == "error" and msg["pipeline"] in wanted:
                    return 1
        # Drain replies to the warm-up frames.
        while True:
            try:
                await asyncio.wait_for(ws.recv(), 1)
            except TimeoutError:
                break

        async def reader() -> None:
            nonlocal in_flight
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg["type"]
                counts[kind] += 1
                if kind == "detections":
                    rtts.append(time.time() * 1000 - msg["client_ts"])
                    for pid, ms in msg["pipeline_ms"].items():
                        pipeline_ms[pid].append(ms)
                    empty["dets"] += not msg["dets"]
                    empty["tracks"] += not msg["tracks"]
                    for track in msg["tracks"]:
                        labels[track["pipeline"]][track["label"]] += 1
                elif kind == "error":
                    print("error:", msg)
                elif kind == "pipeline_status" and msg["status"] == "error":
                    failed.append(f"{msg['pipeline']}: {msg.get('message')}")
                if kind in ("detections", "dropped", "error") and msg.get("seq") is not None:
                    in_flight -= 1

        read_task = asyncio.create_task(reader())
        interval = 1 / args.fps
        deadline = time.monotonic() + args.seconds
        seq = 1
        while time.monotonic() < deadline:
            tick = time.monotonic()
            if in_flight < args.window:
                await ws.send(pack_frame(seq, int(time.time() * 1000), images[seq % len(images)]))
                seq += 1
                in_flight += 1
                counts["sent"] += 1
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - tick)))
        for _ in range(50):  # drain: every frame must get exactly one reply
            if in_flight == 0:
                break
            await asyncio.sleep(0.1)
        read_task.cancel()

    answered = counts["detections"] + counts["dropped"] + counts["error"]
    print(f"frames: {dict(counts)}  unanswered={counts['sent'] - answered}")
    if rtts:
        p95 = sorted(rtts)[max(0, int(len(rtts) * 0.95) - 1)]
        print(
            f"rtt ms: median={statistics.median(rtts):.1f} p95={p95:.1f}  "
            f"result fps={counts['detections'] / args.seconds:.1f}"
        )
    for pid in wanted:
        times = pipeline_ms.get(pid, [])
        median = f"{statistics.median(times):.1f} ms" if times else "never ran"
        print(f"  {pid}: {median}; tracks {dict(labels[pid].most_common(6))}")
    print(f"empty results: raw dets={empty['dets']}  smoothed tracks={empty['tracks']}")
    for failure in failed:
        print("FAILED", failure)
    ok = counts["sent"] == answered and not failed and all(labels[p] for p in wanted)
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8770/ws")
    parser.add_argument("--token", help="server token (GV_TOKEN), sent as Authorization: Bearer")
    parser.add_argument("--pipelines", default="objects", help="comma-separated pipeline ids to enable")
    parser.add_argument("--prompt", help="open_vocab prompt")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hold-ms", type=int, default=300)
    parser.add_argument(
        "--blank-every", type=int, default=0, help="replace every Nth frame with a black one (flicker test)"
    )
    args = parser.parse_args()
    images = [p.read_bytes() for p in args.images]
    if args.blank_every:
        ok, blank = cv2.imencode(".jpg", np.zeros((720, 960, 3), np.uint8))
        assert ok
        images = [
            blank.tobytes() if (i + 1) % args.blank_every == 0 else images[i % len(images)]
            for i in range(args.blank_every * len(images))
        ]
    sys.exit(asyncio.run(run(args, images)))


if __name__ == "__main__":
    main()
