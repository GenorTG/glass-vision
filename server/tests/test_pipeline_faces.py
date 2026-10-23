"""Faces pipeline on the real InsightFace buffalo_l model and ultralytics' zidane.jpg (two faces).

zidane.jpg (1280x720): Zidane on the right (face around x 910-1060), Ancelotti on the left (x 545-665).
Run via `./run -m pytest` so onnxruntime finds cuDNN (falls back to CPU otherwise, still correct).
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics.utils import ASSETS

from glass_vision.pipelines.base import CommandContext, RunContext
from glass_vision.pipelines.faces import FacesPipeline

DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    assert img is not None and img.shape[:2] == (720, 1280)
    return img


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("faces_data")


@pytest.fixture(scope="module")
def pipe(data_dir: Path) -> FacesPipeline:
    p = FacesPipeline(data_dir=data_dir)
    p.load(DEVICE)
    yield p
    p.unload()


def ctx(threshold: float = 0.4) -> RunContext:
    return RunContext(options={"threshold": threshold}, conf=0.35, imgsz=640)


def cmd(p: FacesPipeline, action: str, frame: np.ndarray | None, **args: str):
    return p.command(action, args, CommandContext(options={"threshold": 0.4}, frame=frame))


def right_face(anns):
    return max(anns, key=lambda a: a.box[0])


def test_info_contract() -> None:
    info = FacesPipeline.info.to_json()
    assert info["id"] == "faces" and info["interval_ms"] == 0 and not info["default_enabled"]
    assert [a["id"] for a in info["actions"]] == ["enroll", "list"]
    assert info["options"][0] == {
        "key": "threshold",
        "label": "Name match threshold",
        "type": "number",
        "default": 0.4,
        "min": 0.2,
        "max": 0.8,
        "step": 0.05,
    }
    assert FacesPipeline().gallery_path.name == "gallery.npz"  # no-arg constructor still works


def test_runs_on_cuda(pipe: FacesPipeline) -> None:
    for model in pipe._app.models.values():
        assert model.session.get_providers()[0] == "CUDAExecutionProvider"


def test_detects_both_faces(pipe: FacesPipeline, image: np.ndarray) -> None:
    anns = pipe.run(image, ctx())
    assert len(anns) >= 2
    for a in anns:
        x1, y1, x2, y2 = a.box
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
        assert 40 < x2 - x1 < 400 and 40 < y2 - y1 < 400  # face-sized
        assert 0.5 <= a.conf <= 1.0
        assert a.label == "unknown" and a.detail is None  # empty gallery
        assert len(a.points) == 5 and all(x1 - 5 <= x <= x2 + 5 and v == 1.0 for x, _, v in a.points)
    assert right_face(anns).box[0] > 850


def test_enroll_recognize_list_delete_persist(pipe: FacesPipeline, image: np.ndarray, data_dir: Path) -> None:
    only_zidane = image[:, 760:].copy()

    res = cmd(pipe, "enroll", only_zidane, name="  Zidane ")
    assert res.ok, res.message
    assert res.message.startswith("Learned Zidane (1 sample).")
    assert (data_dir / "faces" / "gallery.npz").exists()

    anns = pipe.run(image, ctx())
    zidane = right_face(anns)
    others = [a for a in anns if a is not zidane]
    assert zidane.label == "Zidane" and float(zidane.detail) > 0.8
    assert others and all(a.label == "unknown" and float(a.detail) < 0.4 for a in others)

    # Second sample, different capitalization → same person.
    res = cmd(pipe, "enroll", image[:, 640:], name="zidane")
    assert res.ok and "Zidane (2 samples)" in res.message

    res = cmd(pipe, "list", None)
    assert res.ok and res.items == [{"key": "Zidane", "label": "Zidane (2 samples)"}]
    assert res.item_action == {"id": "delete", "label": "Forget", "confirm": True}

    # Fresh instance on the same data dir sees the gallery after load.
    fresh = FacesPipeline(data_dir=data_dir)
    fresh.load(DEVICE)
    try:
        assert cmd(fresh, "list", None).items == [{"key": "Zidane", "label": "Zidane (2 samples)"}]
        assert right_face(fresh.run(image, ctx())).label == "Zidane"
    finally:
        fresh.unload()

    res = cmd(pipe, "delete", None, key="Zidane")
    assert res.ok
    assert cmd(pipe, "list", None).items == []
    assert all(a.label == "unknown" for a in pipe.run(image, ctx()))
    assert not cmd(pipe, "delete", None, key="Zidane").ok


def test_sample_cap(pipe: FacesPipeline, image: np.ndarray) -> None:
    for _ in range(22):
        assert cmd(pipe, "enroll", image[:, 760:], name="Cap").ok
    assert cmd(pipe, "list", None).items == [{"key": "Cap", "label": "Cap (20 samples)"}]
    assert cmd(pipe, "delete", None, key="Cap").ok


def test_enroll_rejections(pipe: FacesPipeline, image: np.ndarray) -> None:
    blank = np.full((480, 640, 3), 128, np.uint8)
    res = cmd(pipe, "enroll", blank, name="Nobody")
    assert not res.ok and res.message == "No face in view"
    assert not cmd(pipe, "enroll", image, name="   ").ok
    assert not cmd(pipe, "enroll", image, name="x" * 41).ok
    assert not cmd(pipe, "enroll", None, name="Zidane").ok
    assert not cmd(pipe, "bogus", image).ok
    assert cmd(pipe, "list", None).items == []


def test_timing(pipe: FacesPipeline, image: np.ndarray) -> None:
    # 960x720 = the app's frame size; the centre crop keeps both faces. Gallery with a few people.
    frame = image[:, 160:1120].copy()
    for name, crop in (("A", image[:, 760:]), ("B", image[:, :760])):
        assert cmd(pipe, "enroll", crop, name=name).ok
    for _ in range(5):
        pipe.run(frame, ctx())
    times = []
    for _ in range(10):
        t = time.perf_counter()
        anns = pipe.run(frame, ctx())
        times.append((time.perf_counter() - t) * 1000)
    median = statistics.median(times)
    print(
        f"\nfaces run(): median {median:.1f} ms over 10 runs on 960x720, {len(anns)} faces, labels "
        f"{sorted(a.label for a in anns)}"
    )
    assert sorted(a.label for a in anns) == ["A", "B"]
    assert median < 50  # target < 25 ms on the 3090; loose bound so a busy GPU doesn't flake the suite
    for name in ("A", "B"):
        assert cmd(pipe, "delete", None, key=name).ok
