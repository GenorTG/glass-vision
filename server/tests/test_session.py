"""End-to-end through the real FastAPI app (in-process WebSocket, real models): hello → config → lazy load →
frames → tracks from several pipelines; errors and commands."""

import cv2
import pytest
from fastapi.testclient import TestClient
from ultralytics.utils import ASSETS

from glass_vision.app import app
from glass_vision.protocol import PROTOCOL_VERSION, pack_frame

BUS_JPEG = cv2.imencode(".jpg", cv2.imread(str(ASSETS / "bus.jpg")))[1].tobytes()


def receive_until(ws, predicate, limit: int = 50) -> dict:
    for _ in range(limit):
        msg = ws.receive_json()
        if predicate(msg):
            return msg
    raise AssertionError("expected message never arrived")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_hello_describes_pipelines(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["protocol"] == PROTOCOL_VERSION
        by_id = {p["id"]: p for p in hello["pipelines"]}
        assert {"objects", "open_vocab", "pose"} <= set(by_id)
        assert by_id["objects"]["status"] == "ready" and by_id["objects"]["default_enabled"]
        assert by_id["open_vocab"]["options"][0]["key"] == "prompt"
        assert by_id["pose"]["skeleton"]


def test_two_pipelines_tracks_and_timing(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "pipelines": {"pose": {"enabled": True}}})
        receive_until(
            ws, lambda m: m["type"] == "pipeline_status" and m["pipeline"] == "pose" and m["status"] == "ready"
        )
        for seq in range(3):
            ws.send_bytes(pack_frame(seq, 1000 + seq, BUS_JPEG))
            msg = receive_until(ws, lambda m: m["type"] == "detections")
            assert msg["seq"] == seq and msg["client_ts"] == 1000 + seq
        assert set(msg["pipeline_ms"]) == {"objects", "pose"}
        by_pipeline: dict[str, list[dict]] = {}
        for track in msg["tracks"]:
            by_pipeline.setdefault(track["pipeline"], []).append(track)
            assert track["id"].startswith(track["pipeline"] + ":")
            assert all(0 <= v <= 1 for v in track["box"])
        assert any(t["label"] == "bus" for t in by_pipeline["objects"])
        assert len(by_pipeline["pose"]) >= 3 and all(len(t["points"]) == 17 for t in by_pipeline["pose"])

        # Disabling a pipeline drops its tracks on the next frame.
        ws.send_json({"type": "config", "pipelines": {"pose": {"enabled": False}}})
        ws.send_bytes(pack_frame(9, 9, BUS_JPEG))
        msg = receive_until(ws, lambda m: m["type"] == "detections")
        assert {t["pipeline"] for t in msg["tracks"]} == {"objects"}


def test_bad_input_gets_errors_not_disconnects(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "pipelines": {"nope": {"enabled": True}}})
        assert "unknown pipeline" in receive_until(ws, lambda m: m["type"] == "error")["message"]
        ws.send_text("not json")
        receive_until(ws, lambda m: m["type"] == "error")
        ws.send_bytes(pack_frame(5, 5, b"not a jpeg"))
        assert receive_until(ws, lambda m: m["type"] == "error")["seq"] == 5
        ws.send_json({"type": "command", "req": 1, "pipeline": "objects", "action": "dance", "args": {}})
        result = receive_until(ws, lambda m: m["type"] == "command_result")
        assert (result["req"], result["ok"]) == (1, False)
        ws.send_json({"type": "command", "req": 2, "pipeline": "ghost", "action": "x"})
        assert receive_until(ws, lambda m: m["type"] == "command_result")["ok"] is False
        # Still streaming fine afterwards.
        ws.send_bytes(pack_frame(6, 6, BUS_JPEG))
        assert receive_until(ws, lambda m: m["type"] == "detections")["seq"] == 6


def test_legacy_client_gets_integer_track_ids_until_it_speaks_v3(client: TestClient) -> None:
    # App ≤ 0.3 (protocol v2) only ever sends conf/imgsz/hold_ms and parses track ids with getInt().
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "conf": 0.35, "imgsz": 640, "hold_ms": 300})
        for seq in range(2):
            ws.send_bytes(pack_frame(seq, seq, BUS_JPEG))
            msg = receive_until(ws, lambda m: m["type"] == "detections")
        ids = [t["id"] for t in msg["tracks"]]
        assert ids and all(isinstance(i, int) for i in ids) and len(set(ids)) == len(ids)
        # A v3 client announces itself with a pipelines config (or protocol >= 3): string ids from then on.
        ws.send_json({"type": "config", "pipelines": {"objects": {"enabled": True}}})
        ws.send_bytes(pack_frame(9, 9, BUS_JPEG))
        msg = receive_until(ws, lambda m: m["type"] == "detections")
        assert all(isinstance(t["id"], str) and t["id"].startswith("objects:") for t in msg["tracks"])
