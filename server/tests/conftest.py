import os

import pytest

# Same environment as ./run, in case pytest is started some other way. Must run before glass_vision is imported.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU: a visible pre-SM-7.5 GPU disables cuDNN process-wide
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# Tests talk to the in-process app without a token, whatever server/config.env says (an env var beats the file,
# and empty = unset). Model/data dirs from config.env still apply.
os.environ["GV_TOKEN"] = ""

_gpu: bool | None = None


def gpu_available() -> bool:
    """A CUDA GPU torch can actually run kernels on (same check as GV_DEVICE=auto)."""
    global _gpu
    if _gpu is None:
        from glass_vision.config import resolve_device

        _gpu = resolve_device("auto") != "cpu"
    return _gpu


def needs_gpu(item: pytest.Item) -> bool:
    """Explicit `@pytest.mark.gpu`, or a test module pinned to CUDA (`DEVICE = "cuda:0"`), or a `cuda…` param."""
    if item.get_closest_marker("gpu"):
        return True
    module_device = getattr(getattr(item, "module", None), "DEVICE", None)
    if isinstance(module_device, str) and module_device.startswith("cuda"):
        return True
    callspec = getattr(item, "callspec", None)
    return callspec is not None and any(isinstance(v, str) and v.startswith("cuda") for v in callspec.params.values())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if needs_gpu(item):
            item.add_marker(pytest.mark.gpu)
            if not gpu_available():
                item.add_marker(skip)


# Hand photos from MediaPipe's own sample bucket, fetched on demand instead of being redistributed in this repo.
SAMPLE_IMAGES_URL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{}.jpg"
SAMPLE_IMAGES = ("thumbs_up", "victory", "pointing_up", "thumbs_down")


def pytest_configure(config: pytest.Config) -> None:
    """Some test modules read these images at import time, so fetch them before collection."""
    import urllib.request
    from pathlib import Path

    data = Path(__file__).parent / "data"
    data.mkdir(exist_ok=True)
    for name in SAMPLE_IMAGES:
        target = data / f"{name}.jpg"
        if not target.exists():
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(SAMPLE_IMAGES_URL.format(name), tmp)
            tmp.rename(target)
