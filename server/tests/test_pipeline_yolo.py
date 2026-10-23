"""Real-model tests for the ultralytics pipelines (objects, open_vocab, pose) on ultralytics' sample images."""

import time

import cv2
import pytest
from ultralytics.utils import ASSETS

from glass_vision.pipelines.base import RunContext
from glass_vision.pipelines.objects import ObjectsPipeline
from glass_vision.pipelines.open_vocab import OpenVocabPipeline, parse_prompt
from glass_vision.pipelines.pose import PosePipeline

DEVICE = "cuda:0"
BUS = cv2.imread(str(ASSETS / "bus.jpg"))  # 810x1080: 4 people + a bus


def ctx(**options: object) -> RunContext:
    return RunContext(options=dict(options), conf=0.35, imgsz=640)


def labels(anns: list, min_conf: float = 0.35) -> list[str]:
    return sorted(a.label for a in anns if a.conf >= min_conf)


def median_ms(fn, n: int = 10) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[n // 2]


@pytest.fixture(scope="module")
def objects() -> ObjectsPipeline:
    p = ObjectsPipeline()
    p.load(DEVICE)
    return p


def test_objects_finds_people_and_bus(objects: ObjectsPipeline) -> None:
    anns = objects.run(BUS, ctx())
    found = labels(anns)
    assert found.count("person") >= 3 and "bus" in found
    assert min(a.conf for a in anns) >= 0.1  # detector floor for the tracker's low-conf rescue
    h, w = BUS.shape[:2]
    assert all(0 <= a.box[0] < a.box[2] <= w and 0 <= a.box[1] < a.box[3] <= h for a in anns)
    print(f"objects: {median_ms(lambda: objects.run(BUS, ctx())):.1f} ms")


def test_parse_prompt() -> None:
    assert parse_prompt(" Red Mug, keys;glasses\n, , keys ") == ("red mug", "keys", "glasses")
    assert parse_prompt("") == ()


def test_open_vocab_follows_prompt() -> None:
    p = OpenVocabPipeline()
    p.load(DEVICE)
    found = labels(p.run(BUS, ctx(prompt="bus, person")), 0.25)
    assert "bus" in found and found.count("person") >= 3
    # Changing the prompt changes what is looked for; going back reuses cached embeddings.
    assert "person" not in labels(p.run(BUS, ctx(prompt="bus")), 0.25)
    t0 = time.perf_counter()
    p.run(BUS, ctx(prompt="bus, person"))
    cached_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    p.run(BUS, ctx(prompt="bicycle, backpack, shoe"))
    new_prompt_ms = (time.perf_counter() - t0) * 1000
    print(f"open_vocab prompt switch: cached {cached_ms:.0f} ms, new prompt {new_prompt_ms:.0f} ms")
    assert cached_ms < 100 and new_prompt_ms < 1500
    assert p.run(BUS, ctx(prompt=" , ")) == []
    print(f"open_vocab: {median_ms(lambda: p.run(BUS, ctx(prompt='bus, person'))):.1f} ms")


def test_pose_returns_17_keypoints_per_person() -> None:
    p = PosePipeline()
    p.load(DEVICE)
    anns = [a for a in p.run(BUS, ctx(names=False)) if a.conf >= 0.35]
    assert len(anns) >= 3 and all(a.label == "person" for a in anns)  # names=True: see test_pose_names.py
    assert all(a.points is not None and len(a.points) == 17 for a in anns)
    assert max(i for edge in PosePipeline.info.skeleton for i in edge) == 16
    print(f"pose: {median_ms(lambda: p.run(BUS, ctx(names=True))):.1f} ms")
