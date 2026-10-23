#!/usr/bin/env python3
"""Download every model the server's pipelines need, from its ORIGINAL upstream source, into GV_MODELS_DIR.

    python3 scripts/download_models.py                 # all pipelines (OCR: Latin script only)
    python3 scripts/download_models.py --only faces,text
    python3 scripts/download_models.py --only text --ocr-scripts cyrillic,japanese   # + OCR models for more scripts
    python3 scripts/download_models.py --ocr-scripts all                            # + every OCR script (~1.5 GB)
    python3 scripts/download_models.py --only text --translation-models 1.2B        # + the larger translator (4.6 GB)
    python3 scripts/download_models.py --check         # verify only (exit 1 if anything is missing/corrupt)
    python3 scripts/download_models.py --list          # URLs + licenses, no download

Standard library only (runs before/without the server venv, e.g. in the Docker entrypoint). The target directory is
`--models-dir`, else GV_MODELS_DIR (environment or server/config.env), else server/models.

- Resume-safe: downloads go to `<file>.part` (resumed with an HTTP Range request), are verified, then renamed.
- Verified: every file has a pinned sha256. The hashes were recorded on 2026-10-22/23 from files downloaded from these
  same official URLs; EasyOCR additionally publishes md5 sums of its weights (easyocr/config.py), checked too.
  Hugging Face files are fetched at a pinned git revision (`resolve/<commit>/<file>`), never a moving branch.
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
HF = "https://huggingface.co"
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
    option: str | None = None  # opt-in extra (e.g. "ocr:cyrillic"); None = part of its pipelines' default set

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
]  # fmt: skip

# EasyOCR recognition models, one per script group of the text pipeline (`SCRIPT_GROUPS` in pipelines/text.py; the
# keys are the --ocr-scripts names). Latin is the default; the others are opt-in. url, zip size, file, sha256, md5.
OCR_SCRIPTS: dict[str, tuple[str, int, str, str, str]] = {
    "latin": (f"{EASYOCR}/v1.3/latin_g2.zip", 14284936, "latin_g2.pth",
              "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55", "469869130aad1a34e8f9086f4262bc59"),
    "cyrillic": (f"{EASYOCR}/v1.6.1/cyrillic_g2.zip", 14140353, "cyrillic_g2.pth",
                 "48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29", "19f85f43d9128a89ac21b8d6a06973fe"),
    "zh_sim": (f"{EASYOCR}/v1.3/zh_sim_g2.zip", 20288076, "zh_sim_g2.pth",
               "cb678fdef09d651e7763ca551ad790dc89f0b2e3d2a640484330e338fb574c7a", "b601ce7143293387d3ec4f41a66edc07"),
    "zh_tra": (f"{EASYOCR}/pre-v1.1.6/chinese.zip", 209821039, "chinese.pth",
               "19ecc60f4e079eb2cd7f25f3513fc996cf80819fc1f1d0aee24c58402cc4776d", "dfba8e364cd98ed4fed7ad54d71e3965"),
    "japanese": (f"{EASYOCR}/v1.3/japanese_g2.zip", 16065475, "japanese_g2.pth",
                 "afbb32b0485297de5a7c1f59aeecac12c69f32b15cedd3c3d8c2e8529de85af3", "bad5146990ccb1272cb0908440fbe15e"),
    "korean": (f"{EASYOCR}/v1.3/korean_g2.zip", 14910151, "korean_g2.pth",
               "ecdead53c909748aad74bc2f2beeab3e3ad751134db6335e97ea19f879ca407e", "befecf7b1ca2fffb5af814a51443682d"),
    "arabic": (f"{EASYOCR}/pre-v1.1.6/arabic.zip", 199916962, "arabic.pth",
               "2a9afd42c374deb98aed0b53c9b77d75e1d00d4e0501f3b0276c54190c89b1a8", "993074555550e4e06a6077d55ff0449a"),
    "devanagari": (f"{EASYOCR}/pre-v1.1.6/devanagari.zip", 199886824, "devanagari.pth",
                   "1a8d0abb2032c0ec9d06639bc078b7686c10cf3b8193fe42a6382bc6846ae4a3",
                   "db6b1f074fae3070f561675db908ac08"),
    "bengali": (f"{EASYOCR}/v1.1.8/bengali.zip", 199852959, "bengali.pth",
                "28d10d7dab96bf38c1d190d09a65a732130f5f34b28dcbd7e51aba3366146896", "cea9e897e2c0576b62cbb1554997ce1c"),
    "thai": (f"{EASYOCR}/pre-v1.1.6/thai.zip", 199881630, "thai.pth",
             "3ebbb9321b9c7c694e169cd23cc37f3c0bd2a160f5b539602bbef80e663e0a17", "40a06b563a2b3d7897e2d19df20dc709"),
    "telugu": (f"{EASYOCR}/v1.2/telugu.zip", 14108882, "telugu.pth",
               "a73ba1c65f8b3e7b18df3104c1f137970510ce51e3e1d9341506292730fa3c43", "f7576012a3abe593950c47bfa1bd8ddc"),
    "kannada": (f"{EASYOCR}/v1.2/kannada.zip", 14105056, "kannada.pth",
                "3cc308972bccb2c483741e90e87f2941c1b40d84cf548115f48372d288deb0f0", "c240c97e4adb8773b53b3b3dec728627"),
}  # fmt: skip
DEFAULT_OCR_SCRIPTS = ("latin",)
ASSETS += [
    Asset(("text",), url, "Apache-2.0", "", size, members=(Member(name, f"easyocr/{name}", sha, md5),),
          option=None if key in DEFAULT_OCR_SCRIPTS else f"ocr:{key}")
    for key, (url, size, name, sha, md5) in OCR_SCRIPTS.items()
]  # fmt: skip

# Translation for the text pipeline (glass_vision/translate.py `MODELS`): M2M100 at a pinned commit of its original
# repo. --translation-models key → (repo, commit, dir under the models dir, [(file, size, sha256)]). 418M is the
# default; 1.2B (better on one-word signs, 4.6 GB) is opt-in.
_M2M_SHARED = (
    ("generation_config.json", 233, "aed76366507333ddbb8bd49960f23c82fe6446b3319a46a54befdb45324ccf61"),
    ("sentencepiece.bpe.model", 2423393, "d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a"),
    ("special_tokens_map.json", 1140, "c1a4f86c3874d279ae1b2a05162858db5dd6c61665d84223ed886cbcff08fda6"),
    ("vocab.json", 3708092, "b6e77e474aeea8f441363aca7614317c06381f3eacfe10fb9856d5081d1074cc"),
)
TRANSLATION_MODELS: dict[str, tuple[str, str, str, tuple[tuple[str, int, str], ...]]] = {
    "418M": ("facebook/m2m100_418M", "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636", "translation/m2m100_418M", (
        ("config.json", 908, "df0ae43e4e4b0d7e3c97b7f447857a70ef6b6a2aa1f145cedbcc730d95f67134"),
        ("tokenizer_config.json", 298, "a53e6aa83da0b82565ed90c3849056307a9453843322ac5b8439ec4b9497fe48"),
        ("pytorch_model.bin", 1935796948, "d907ea45e4e4b9db163382a6674f6218b3c59566fe06d77f4055c208b4e87ed1"),
        *_M2M_SHARED,
    )),
    "1.2B": ("facebook/m2m100_1.2B", "7b36184180524c1a1bbfa37f120a608046250b98", "translation/m2m100_1.2B", (
        ("config.json", 909, "13b5191c92371b2d587baac6c9a04de018a5264d7179020d379a276769051f95"),
        ("tokenizer_config.json", 271, "1566a6cfa4f541a55594c9d5e090f530812d5de7c94882ea3af156962d9933ae"),
        ("pytorch_model.bin", 4958230644, "a58ef8f42362ef12adeddc600b3425f1e2bbd019cfa6aae6b0051e2e3e055cd4"),
        *_M2M_SHARED,
    )),
}  # fmt: skip
DEFAULT_TRANSLATION_MODELS = ("418M",)
ASSETS += [
    Asset(("text",), f"{HF}/{repo}/resolve/{commit}/{name}", "MIT", sha, size, f"{folder}/{name}",
          option=None if key in DEFAULT_TRANSLATION_MODELS else f"translation:{key}")
    for key, (repo, commit, folder, files) in TRANSLATION_MODELS.items()
    for name, size, sha in files
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
    part = downloads / ((asset.dest or name).replace("/", "_") + ".part")  # unique: several repos have config.json
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
    parser.add_argument(
        "--ocr-scripts",
        action="append",
        default=[],
        help=f"text pipeline: OCR models for more scripts, comma-separated or 'all' (default: latin): {list(OCR_SCRIPTS)}",
    )
    parser.add_argument(
        "--translation-models",
        action="append",
        default=[],
        help=f"text pipeline: extra translation models, comma-separated or 'all' (default: 418M): "
        f"{list(TRANSLATION_MODELS)}",
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
    scripts = {k.strip() for item in args.ocr_scripts for k in item.split(",") if k.strip()}
    if "all" in scripts:
        scripts = set(OCR_SCRIPTS)
    if unknown := scripts - set(OCR_SCRIPTS):
        parser.error(f"unknown OCR script(s) {sorted(unknown)}; known: {list(OCR_SCRIPTS)}")
    translation = {k.strip() for item in args.translation_models for k in item.split(",") if k.strip()}
    if "all" in translation:
        translation = set(TRANSLATION_MODELS)
    if unknown := translation - set(TRANSLATION_MODELS):
        parser.error(f"unknown translation model(s) {sorted(unknown)}; known: {list(TRANSLATION_MODELS)}")
    options = {f"ocr:{k}" for k in scripts} | {f"translation:{k}" for k in translation}
    assets = [a for a in ASSETS if wanted & set(a.pipelines) and (a.option is None or a.option in options)]

    if args.list:
        for a in assets:
            what = ",".join(a.pipelines) + (f" ({a.option})" if a.option else "")
            print(f"{what:<22} {a.size / 2**20:7.1f} MiB  {a.license:<52} {a.url}")
        return 0

    root = (args.models_dir or default_models_dir()).expanduser().resolve()
    print(f"models dir: {root}")
    missing = [a for a in assets if not is_installed(root, a, deep=not args.fast_check)]
    if args.check:
        for a in missing:
            print(f"  missing/corrupt: {a.dest or a.url.rsplit('/', 1)[1]} ({','.join(a.pipelines)})")
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
