#!/usr/bin/env python3
"""Download every model the server's pipelines need, from its ORIGINAL upstream source, into GV_MODELS_DIR.

    python3 scripts/download_models.py                 # all pipelines
    python3 scripts/download_models.py --only faces,text
    python3 scripts/download_models.py --check         # verify only (exit 1 if anything is missing/corrupt)
    python3 scripts/download_models.py --list          # URLs + licenses, no download

Standard library only (runs before/without the server venv, e.g. in the Docker entrypoint). The target directory is
`--models-dir`, else GV_MODELS_DIR (environment or server/config.env), else server/models.

- Resume-safe: downloads go to `<file>.part` (resumed with an HTTP Range request), are verified, then renamed.
- Verified: every file has a pinned sha256. The hashes were recorded on 2026-10-22 from files downloaded from these
  same official URLs; EasyOCR additionally publishes md5 sums of its weights (easyocr/config.py), checked too.
  A hash mismatch means upstream changed the file (or the download is corrupt): the file is NOT installed.
- Layout matches each pipeline's load code (server/glass_vision/pipelines/*.py). Licenses: docs/MODELS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "server"
ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"
EASYOCR = "https://github.com/JaidedAI/EasyOCR/releases/download"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Member:
    """A file extracted from a zip download."""

    name: str  # path inside the zip
    dest: str  # relative to the models dir
    sha256: str
    md5: str | None = None  # upstream-published checksum, if any


@dataclass(frozen=True)
class Asset:
    pipelines: tuple[str, ...]
    url: str
    license: str
    sha256: str  # of the downloaded file
    size: int
    dest: str | None = None  # relative to the models dir; None = zip that is extracted (see members)
    members: tuple[Member, ...] = field(default_factory=tuple)

    @property
    def outputs(self) -> list[tuple[str, str, str | None]]:
        """(dest, sha256, md5) of what ends up on disk."""
        if self.dest is not None:
            return [(self.dest, self.sha256, None)]
        return [(m.dest, m.sha256, m.md5) for m in self.members]


BUFFALO = "insightface/models/buffalo_l"  # FaceAnalysis(root=<models>/insightface) looks in <root>/models/<name>
ASSETS: list[Asset] = [
    Asset(("objects",), f"{ULTRALYTICS}/yolo11s.pt", "AGPL-3.0",
          "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5", 19313732, "yolo11s.pt"),
    Asset(("pose",), f"{ULTRALYTICS}/yolo11s-pose.pt", "AGPL-3.0",
          "1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba", 20373161, "yolo11s-pose.pt"),
    Asset(("pose",), f"{MEDIAPIPE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "Apache-2.0",
          "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1", 7819105, "mediapipe/hand_landmarker.task"),
    Asset(("open_vocab",), f"{ULTRALYTICS}/yoloe-11s-seg.pt", "AGPL-3.0",
          "8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3", 27803986, "yoloe-11s-seg.pt"),
    Asset(("open_vocab",), f"{ULTRALYTICS}/mobileclip_blt.ts", "Apple ML Research Model TOU (non-commercial research)",
          "a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54", 599764649, "mobileclip_blt.ts"),
    Asset(("faces",), "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip",
          "InsightFace models: non-commercial research only",
          "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", 288621354,
          members=tuple(
              Member(name, f"{BUFFALO}/{name}", sha)
              for name, sha in (
                  ("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
                  ("w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
                  ("1k3d68.onnx", "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc"),
                  ("2d106det.onnx", "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
                  ("genderage.onnx", "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"),
              )
          )),
    Asset(("gestures",), f"{MEDIAPIPE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
          "Apache-2.0", "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482", 8373440,
          "mediapipe/gesture_recognizer.task"),
    Asset(("text",), f"{EASYOCR}/pre-v1.1.6/craft_mlt_25k.zip", "Apache-2.0",
          "", 77251756, members=(Member("craft_mlt_25k.pth", "easyocr/craft_mlt_25k.pth",
                                        "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
                                        "2f8227d2def4037cdb3b34389dcf9ec1"),)),
    Asset(("text",), f"{EASYOCR}/v1.3/english_g2.zip", "Apache-2.0",
          "", 14040947, members=(Member("english_g2.pth", "easyocr/english_g2.pth",
                                        "e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8",
                                        "5864788e1821be9e454ec108d61b887d"),)),
    Asset(("text",), f"{EASYOCR}/v1.3/latin_g2.zip", "Apache-2.0",
          "", 14284936, members=(Member("latin_g2.pth", "easyocr/latin_g2.pth",
                                        "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55",
                                        "469869130aad1a34e8f9086f4262bc59"),)),
]  # fmt: skip
PIPELINES = sorted({p for a in ASSETS for p in a.pipelines})
NON_COMMERCIAL = ("faces", "open_vocab")


def default_models_dir() -> Path:
    sys.path.insert(0, str(SERVER))
    from glass_vision.config import load_settings  # stdlib-only module: no venv needed

    return load_settings().models_dir


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def is_installed(root: Path, asset: Asset, *, deep: bool) -> bool:
    """All outputs present (and, with deep=True, hash-verified)."""
    for dest, sha, _ in asset.outputs:
        path = root / dest
        if not path.is_file():
            return False
        if deep and digest(path, "sha256") != sha:
            print(f"  ! {dest}: sha256 mismatch (corrupt or changed)", file=sys.stderr)
            return False
    return True


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label, self.total, self.tty = label, total, sys.stderr.isatty()
        self.last = 0.0

    def __call__(self, done: int, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - self.last < (0.2 if self.tty else 5.0):
            return
        self.last = now
        pct = f"{100 * done / self.total:5.1f}%" if self.total else ""
        line = f"  {self.label}: {done / 2**20:7.1f} / {self.total / 2**20:.1f} MiB {pct}"
        print(
            ("\r" + line) if self.tty else line,
            end="\n" if final or not self.tty else "",
            file=sys.stderr,
        )


def fetch(url: str, part: Path, size: int, label: str, attempts: int = 4) -> None:
    """Download url → part, resuming a previous partial download."""
    progress = Progress(label, size)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if size and have == size:
            return
        if size and have > size:
            part.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"User-Agent": "glass-vision-download-models/1"})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:  # server ignored the Range: start over
                    have = 0
                with part.open("ab" if have else "wb") as out:
                    done = have
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        progress(done)
            progress(part.stat().st_size, final=True)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == attempts:
                raise RuntimeError(f"download failed after {attempts} attempts: {url}: {exc}") from exc
            print(
                f"\n  {label}: {exc}; retrying ({attempt}/{attempts - 1})",
                file=sys.stderr,
            )
            time.sleep(2 * attempt)


def install(root: Path, asset: Asset) -> None:
    name = asset.url.rsplit("/", 1)[1]
    downloads = root / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    part = downloads / (name + ".part")
    fetch(asset.url, part, asset.size, name)
    if asset.sha256 and digest(part, "sha256") != asset.sha256:
        part.unlink()
        raise RuntimeError(f"{name}: sha256 mismatch — upstream file changed or download corrupt; not installed")

    if asset.dest is not None:
        target = root / asset.dest
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, target)
        return

    with (
        tempfile.TemporaryDirectory(dir=downloads) as tmp,
        zipfile.ZipFile(part) as archive,
    ):
        staged: list[tuple[Path, Path]] = []
        for m in asset.members:
            # Extract by exact member name (never trust paths from the archive).
            out = Path(tmp) / Path(m.dest).name
            with archive.open(m.name) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            if digest(out, "sha256") != m.sha256 or (m.md5 and digest(out, "md5") != m.md5):
                raise RuntimeError(f"{name}:{m.name}: checksum mismatch — not installed")
            staged.append((out, root / m.dest))
        for out, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(out, target)
    part.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, help="default: GV_MODELS_DIR or server/models")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=f"pipeline(s), comma-separated: {PIPELINES}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit 1 if anything is missing or corrupt",
    )
    parser.add_argument("--list", action="store_true", help="print URLs and licenses, download nothing")
    parser.add_argument(
        "--fast-check",
        action="store_true",
        help="with --check: presence only, skip hashing",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout/stderr (progress) lines in order

    wanted = {p.strip() for item in args.only for p in item.split(",") if p.strip()} or set(PIPELINES)
    if unknown := wanted - set(PIPELINES):
        parser.error(f"unknown pipeline(s) {sorted(unknown)}; known: {PIPELINES}")
    assets = [a for a in ASSETS if wanted & set(a.pipelines)]

    if args.list:
        for a in assets:
            print(f"{','.join(a.pipelines):<11} {a.size / 2**20:7.1f} MiB  {a.license:<52} {a.url}")
        return 0

    root = (args.models_dir or default_models_dir()).expanduser().resolve()
    print(f"models dir: {root}")
    missing = [a for a in assets if not is_installed(root, a, deep=not args.fast_check)]
    if args.check:
        for a in missing:
            print(f"  missing/corrupt: {a.url.rsplit('/', 1)[1]} ({','.join(a.pipelines)})")
        print("all models present" if not missing else f"{len(missing)} of {len(assets)} model files need download")
        return 1 if missing else 0

    if any(p in wanted for p in NON_COMMERCIAL):
        print("note: the faces (InsightFace buffalo_l) and open_vocab (Apple MobileCLIP) weights are licensed for "
              "non-commercial research only — see docs/MODELS.md")  # fmt: skip
    total = sum(a.size for a in missing)
    print(f"{len(assets) - len(missing)} of {len(assets)} already present; downloading {total / 2**20:.0f} MiB")
    failures = 0
    for a in missing:
        try:
            install(root, a)
        except (
            OSError,
            RuntimeError,
            KeyError,
            zipfile.BadZipFile,
        ) as exc:  # keep going with the others
            failures += 1
            print(f"  FAILED {exc}", file=sys.stderr)
    downloads = root / ".downloads"
    if downloads.is_dir() and not any(downloads.iterdir()):
        downloads.rmdir()
    print("done" if not failures else f"{failures} download(s) failed (rerun to resume)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
