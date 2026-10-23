"""Gestures pipeline: real MediaPipe model on MediaPipe's own sample photos (tests/data/*.jpg)."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from mediapipe.tasks.python import BaseOptions
from ultralytics.utils import ASSETS

from glass_vision.pipelines.base import Annotation, RunContext
from glass_vision.pipelines.gestures import GesturesPipeline

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module", params=["cuda:0", "cpu"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[GesturesPipeline]:
    """Every test runs on the GPU delegate (default on the server) and on the CPU fallback."""
    p = GesturesPipeline()
    p.load(request.param)
    expected = BaseOptions.Delegate.GPU if request.param.startswith("cuda") else BaseOptions.Delegate.CPU
    assert p._delegate == expected  # the GPU must not have silently fallen back
    yield p
    p.unload()


def _ctx(**options: float) -> RunContext:
    return RunContext(options={**GesturesPipeline.info.default_options(), **options}, conf=0.35, imgsz=640)


def _image(name: str) -> np.ndarray:
    img = cv2.imread(str(DATA / f"{name}.jpg"))
    assert img is not None, name
    return img


def _assert_hand(ann: Annotation, w: int, h: int) -> None:
    assert ann.points is not None and len(ann.points) == 21
    x1, y1, x2, y2 = ann.box
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    for x, y, vis in ann.points:
        assert 0 <= x <= w and 0 <= y <= h
        assert x1 <= x <= x2 and y1 <= y <= y2
        assert vis == 1.0
    assert 0 < ann.conf <= 1
    assert ann.detail in ("Left", "Right")


def test_info() -> None:
    info = GesturesPipeline.info
    assert info.id == "gestures" and not info.default_enabled and info.interval_ms == 0
    assert info.skeleton is not None and len(info.skeleton) == 21
    assert {i for edge in info.skeleton for i in edge} == set(range(21))
    assert {o.key for o in info.options} == {"num_hands", "min_conf"}
    assert info.actions == []


@pytest.mark.parametrize(
    ("name", "label"),
    [("thumbs_up", "Thumb_Up"), ("victory", "Victory"), ("pointing_up", "Pointing_Up"), ("thumbs_down", "Thumb_Down")],
)
def test_gesture(pipeline: GesturesPipeline, name: str, label: str) -> None:
    img = _image(name)
    anns = pipeline.run(img, _ctx())
    assert [a.label for a in anns] == [label]
    _assert_hand(anns[0], img.shape[1], img.shape[0])


def test_no_hands(pipeline: GesturesPipeline) -> None:
    # Not bus.jpg: the man at x≈260, y≈520 holds a real hand to his chest (the GPU delegate finds it, CPU misses it).
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (720, 960, 3), dtype=np.uint8), (31, 31), 0)
    for img in (np.full((720, 960, 3), 127, np.uint8), noise, cv2.imread(str(ASSETS / "bus.jpg"))[:400]):
        assert pipeline.run(img, _ctx()) == []


def test_num_hands_option(pipeline: GesturesPipeline) -> None:
    # Two hands side by side (this pair is found in any layout; the palm detector misses e.g. victory's
    # smaller hand once the composite is downscaled to its 192 px input).
    left, right = _image("thumbs_up"), _image("pointing_up")
    h = min(left.shape[0], right.shape[0])
    both = np.hstack([left[:h], right[:h]])
    assert sorted(a.label for a in pipeline.run(both, _ctx(num_hands=2))) == ["Pointing_Up", "Thumb_Up"]
    assert len(pipeline.run(both, _ctx(num_hands=1))) == 1
    assert len(pipeline.run(both, _ctx(num_hands=4))) == 2


def test_timing_960x720(pipeline: GesturesPipeline) -> None:
    frame = np.full((720, 960, 3), 127, np.uint8)
    hand = _image("thumbs_up")
    frame[150 : 150 + hand.shape[0], 270 : 270 + hand.shape[1]] = hand
    ctx = _ctx()
    assert [a.label for a in pipeline.run(frame, ctx)] == ["Thumb_Up"]
    times = []
    for _ in range(10):
        t = time.perf_counter()
        pipeline.run(frame, ctx)
        times.append((time.perf_counter() - t) * 1000)
    print(
        f"\ngestures {pipeline._delegate.name} 960x720: median {statistics.median(times):.1f} ms (min {min(times):.1f}, max {max(times):.1f})"
    )
