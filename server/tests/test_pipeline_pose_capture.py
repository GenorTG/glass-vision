"""Pose `capture` end-to-end on real images (YOLO pose + MediaPipe hands → OpenPose files), the `/captures` route,
and `files` in the WebSocket `command_result`. DATA_DIR points at a temporary directory."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ultralytics.utils import ASSETS

from glass_vision import captures
from glass_vision.pipelines import base
from glass_vision.pipelines.base import CommandContext
from glass_vision.pipelines.pose import PosePipeline, assign_hands

DATA = Path(__file__).parent / "data"
BUS = cv2.imread(str(ASSETS / "bus.jpg"))  # 4 people
THUMBS_UP = cv2.imread(str(DATA / "thumbs_up.jpg"))  # 1 person, right hand raised in front of the body
NAMES = list(captures.CAPTURE_FILES)


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(base, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def pose() -> Iterator[PosePipeline]:
    p = PosePipeline()
    p.load("cuda:0")
    yield p
    p.unload()


def capture(p: PosePipeline, image: np.ndarray | None) -> base.CommandResult:
    return p.command("capture", {}, CommandContext(options={"names": True}, frame=image))


def capture_dir(result: base.CommandResult) -> Path:
    assert result.files is not None
    return captures.captures_root() / result.files[0]["url"].split("/")[2]


@pytest.mark.gpu
def test_capture_bus(pose: PosePipeline) -> None:
    result = capture(pose, BUS)
    assert result.ok, result.message
    assert result.files is not None and [f["name"] for f in result.files] == NAMES
    assert [f["kind"] for f in result.files] == ["photo", "openpose", "openpose", "json", "overlay"]
    d = capture_dir(result)
    assert captures.CAPTURE_ID_RE.fullmatch(d.name) and d.parent == base.DATA_DIR / "captures"
    assert all(f["url"] == f"/captures/{d.name}/{f['name']}" for f in result.files)

    doc = json.loads((d / "pose_keypoints.json").read_text())
    n = len(doc["people"])
    assert n >= 4 and result.message == f"Captured {n} people → 5 files"
    assert (doc["canvas_width"], doc["canvas_height"]) == (BUS.shape[1], BUS.shape[0])
    assert all(len(p["pose_keypoints_2d"]) == 54 for p in doc["people"])

    for name in ("photo.jpg", "openpose_body.png", "openpose_full.png", "pose_on_photo.jpg"):
        img = cv2.imread(str(d / name))
        assert img is not None and img.shape == BUS.shape, name
    body = cv2.imread(str(d / "openpose_body.png"))
    painted = body.any(axis=2).mean()
    assert 0.003 < painted < 0.2  # skeletons on black
    assert body[0, 0].tolist() == [0, 0, 0]
    assert np.abs(cv2.imread(str(d / "photo.jpg")).astype(int) - BUS.astype(int)).mean() < 3  # the frame itself


@pytest.mark.gpu
def test_capture_thumbs_up_has_a_hand(pose: PosePipeline) -> None:
    result = capture(pose, THUMBS_UP)
    assert result.ok and result.message == "Captured 1 person → 5 files"
    d = capture_dir(result)
    (person,) = json.loads((d / "pose_keypoints.json").read_text())["people"]
    # Her right arm crosses in front of the body; MediaPipe's hand attaches to YOLO's right wrist.
    assert len(person["hand_right_keypoints_2d"]) == 63 and person["hand_left_keypoints_2d"] == []
    hand = np.array(person["hand_right_keypoints_2d"]).reshape(21, 3)
    assert ((hand[:, :2] > 0) & (hand[:, :2] < 1)).all() and (hand[:, 2] == 1).all()
    body = cv2.imread(str(d / "openpose_body.png"))
    full = cv2.imread(str(d / "openpose_full.png"))
    diff = np.argwhere((body != full).any(axis=2))
    assert len(diff) > 200  # the hand is drawn only in the full render…
    x, y = hand[0, 0] * full.shape[1], hand[0, 1] * full.shape[0]
    assert np.hypot(diff[:, 1] - x, diff[:, 0] - y).min() < 5  # …around the hand's landmarks
    assert full[int(y), int(x)].tolist() == [255, 0, 0]  # hand point colour (0, 0, 255) RGB, stored BGR


@pytest.mark.gpu
def test_no_person_no_files(pose: PosePipeline) -> None:
    result = capture(pose, np.zeros((480, 640, 3), np.uint8))
    assert not result.ok and result.message == "No person in view" and result.files is None
    assert not (base.DATA_DIR / "captures").exists()
    assert not capture(pose, None).ok
    assert not pose.command("dance", {}, CommandContext({}, BUS)).ok


def test_assign_hands_nearest_wrist() -> None:
    def person(lw: tuple[float, float], rw: tuple[float, float], vis: float = 1.0) -> list[tuple[float, float, float]]:
        pts = [(0.0, 0.0, 0.0)] * 17
        pts[7], pts[8] = (lw[0], lw[1] - 50, 1.0), (rw[0], rw[1] - 50, 1.0)  # elbows 50 px above
        pts[9], pts[10] = (*lw, vis), (*rw, vis)
        return pts

    def hand(x: float, y: float) -> list[tuple[float, float]]:
        return [(x, y)] + [(x, y - 10)] * 20

    people = [person((300, 300), (200, 300)), person((700, 300), (600, 300))]
    boxes = [(150.0, 100.0, 350.0, 500.0), (550.0, 100.0, 750.0, 500.0)]
    hands = [hand(605, 310), hand(296, 290), hand(450, 50), hand(302, 305)]
    assigned, loose = assign_hands(people, hands, boxes)
    assert assigned == {(1, "right"): 0, (0, "left"): 3}  # 3 is nearer than 1 to the same wrist
    assert loose == [1, 2]
    assert assign_hands([person((300, 300), (200, 300), vis=0.1)], [hand(300, 300)], boxes[:1]) == ({}, [0])


# --- HTTP route -------------------------------------------------------------------------------------


@pytest.fixture
def http() -> TestClient:
    app = FastAPI()
    app.include_router(captures.router)
    return TestClient(app)


@pytest.mark.gpu
def test_route_serves_capture_files(pose: PosePipeline, http: TestClient) -> None:
    result = capture(pose, BUS)
    assert result.files is not None
    d = capture_dir(result)
    for f in result.files:
        r = http.get(f["url"])
        assert r.status_code == 200, f
        assert r.content == (d / f["name"]).read_bytes()
        assert r.headers["content-type"].startswith(captures.CAPTURE_FILES[f["name"]].media_type)
    assert http.get(result.files[3]["url"]).json()["canvas_width"] == BUS.shape[1]


@pytest.mark.parametrize(
    "url",
    [
        "/captures/20261023-101500-123/photo.jpg",  # well-formed but doesn't exist
        "/captures/20261023-101500-123/secret.txt",
        "/captures/2026/photo.jpg",
        "/captures/20261023-101500-12a/photo.jpg",
        "/captures/20261023-101500-1234/photo.jpg",
        "/captures/20261023-101500-123%0A/photo.jpg",
        "/captures/..%2F..%2Fpyproject.toml/photo.jpg",
        "/captures/20261023-101500-123/..%2F..%2F..%2Fpyproject.toml",
        "/captures/../../pyproject.toml",
        "/captures/%2e%2e/photo.jpg",
        "/captures/20261023-101500-123/",
    ],
)
def test_route_rejects(url: str, http: TestClient, data_dir: Path) -> None:
    d = data_dir / "captures" / "20261023-101500-123"
    d.mkdir(parents=True)
    (d / "secret.txt").write_text("no")
    (data_dir / "pyproject.toml").write_text("no")
    r = http.get(url)
    assert r.status_code == 404, (url, r.status_code)
    assert b"no" != r.content


def test_websocket_command_result_carries_files() -> None:
    from glass_vision.app import app
    from glass_vision.protocol import pack_frame

    jpeg = cv2.imencode(".jpg", BUS)[1].tobytes()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "protocol": 3, "pipelines": {"pose": {"enabled": True}}})
        ws.send_bytes(pack_frame(1, 1, jpeg))
        for _ in range(50):
            if ws.receive_json()["type"] == "detections":
                break
        ws.send_json({"type": "command", "req": 7, "pipeline": "pose", "action": "capture", "args": {}})
        for _ in range(50):
            msg = ws.receive_json()
            if msg["type"] == "command_result":
                break
        assert msg["ok"] and msg["req"] == 7, msg
        assert [f["name"] for f in msg["files"]] == NAMES
        assert all(f["url"].startswith("/captures/") for f in msg["files"])
