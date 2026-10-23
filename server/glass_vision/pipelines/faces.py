"""Faces: InsightFace buffalo_l (SCRFD-10G detector + ArcFace R50 recognizer) naming people the user taught it.

Fast pipeline (runs on every frame). Detection and recognition both run on onnxruntime's CUDA EP; all faces of a
frame are aligned and embedded in one batched ArcFace call. The gallery (name → up to `MAX_SAMPLES` L2-normalized
512-d embeddings) lives in memory as numpy arrays and on disk at `DATA_DIR/faces/gallery.npz`, rewritten atomically
on every change. A face's score for a person is the best cosine similarity over that person's samples.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import (
    DATA_DIR,
    MODELS_DIR,
    ActionSpec,
    Annotation,
    CommandContext,
    CommandResult,
    OptionSpec,
    Pipeline,
    PipelineInfo,
    RunContext,
)

log = logging.getLogger(__name__)

INSIGHTFACE_ROOT = MODELS_DIR / "insightface"  # buffalo_l is downloaded to <root>/models/buffalo_l on first load
MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)
EMBEDDING_DIM = 512
MAX_SAMPLES = 20  # per person; the oldest sample is dropped beyond this
MAX_NAME_LEN = 40
UNKNOWN = "unknown"

# ArcFace's canonical 112x112 landmark positions (insightface.utils.face_align.arcface_dst).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32
)


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def _align(image: np.ndarray, kps: np.ndarray, size: int) -> np.ndarray:
    """112x112 ArcFace crop. Same least-squares similarity fit (Umeyama) as insightface's `face_align.norm_crop`,
    which goes through skimage's deprecated `SimilarityTransform.estimate` (warns, removed in skimage 2.2)."""
    src, dst = kps.astype(np.float64), ARCFACE_DST.astype(np.float64) * (size / 112.0)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    u, s, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    d = np.array([1.0, -1.0 if np.linalg.det(u) * np.linalg.det(vt) < 0 else 1.0])  # no reflections
    rot = u @ np.diag(d) @ vt
    scale = (s @ d) / src_c.var(axis=0).sum()
    m = np.hstack([scale * rot, (dst_mean - scale * rot @ src_mean)[:, None]])
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)


def _samples_label(n: int) -> str:
    return f"{n} sample" if n == 1 else f"{n} samples"


class FacesPipeline(Pipeline):
    info = PipelineInfo(
        id="faces",
        name="Faces",
        description="Detects faces and names the people you taught it (Learn face).",
        default_enabled=False,
        uses_global_conf=False,
        interval_ms=0,
        options=[
            OptionSpec("threshold", "Name match threshold", "number", 0.4, min=0.2, max=0.8, step=0.05),
        ],
        actions=[
            ActionSpec("enroll", "Learn face", [OptionSpec("name", "Name", "text", "")]),
            ActionSpec("list", "Known faces"),
        ],
        skeleton=None,  # the 5 landmarks (eyes, nose, mouth corners) are drawn as plain points
    )

    def __init__(self, data_dir: Path | None = None) -> None:
        self.gallery_path = (data_dir if data_dir is not None else DATA_DIR) / "faces" / "gallery.npz"
        self._app: Any = None  # insightface.app.FaceAnalysis
        self._gallery: dict[str, np.ndarray] = {}  # name → (k, 512) float32, rows L2-normalized, oldest first
        # Flattened gallery for run(): all samples stacked person by person, the person index of every row, and
        # the first row of each person (np.maximum.reduceat offsets). Rebuilt by _rebuild_index on every change.
        self._names: list[str] = []
        self._matrix = np.zeros((0, EMBEDDING_DIM), np.float32)
        self._owner = np.zeros(0, np.intp)
        self._starts = np.zeros(0, np.intp)

    # --- lifecycle -------------------------------------------------------------------------------------------

    def load(self, device: str) -> None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis  # heavy import: only when the pipeline is actually used

        ctx_id = int(device.split(":", 1)[1] or 0) if device.startswith("cuda") else -1
        if ctx_id >= 0 and "CUDAExecutionProvider" not in ort.get_available_providers():
            # CPU-only onnxruntime installed (CPU install, macOS, or insightface's onnxruntime overwrote -gpu).
            log.warning("faces: onnxruntime has no CUDA provider %s, using CPU", ort.get_available_providers())
            ctx_id = -1
        providers: list[Any] = ["CPUExecutionProvider"]
        if ctx_id >= 0:
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": ctx_id}))
        INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
        app = FaceAnalysis(
            name=MODEL_PACK,
            root=str(INSIGHTFACE_ROOT),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
        used = {task: model.session.get_providers()[0] for task, model in app.models.items()}
        if ctx_id >= 0 and any(p != "CUDAExecutionProvider" for p in used.values()):
            log.warning("faces: onnxruntime fell back off CUDA (%s) — is the server started via ./run?", used)
        else:
            log.info("faces: %s loaded, providers %s", MODEL_PACK, used)
        self._app = app
        self._load_gallery()
        self.run(np.zeros((480, 640, 3), np.uint8), RunContext(self.info.default_options(), 0.0, 640))  # warm up

    def unload(self) -> None:
        self._app = None  # gallery stays in memory and on disk

    # --- inference -------------------------------------------------------------------------------------------

    def _faces(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """→ boxes (n, 4) float, det scores (n,), kps (n, 5, 2) — no embeddings."""
        if self._app is None:
            raise RuntimeError("faces pipeline not loaded")
        dets, kpss = self._app.det_model.detect(image, max_num=0, metric="default")
        if dets.shape[0] == 0 or kpss is None:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        return dets[:, :4], dets[:, 4], kpss

    def _embed(self, image: np.ndarray, kpss: np.ndarray) -> np.ndarray:
        """Align every face and run ArcFace once on the whole batch → (n, 512) normalized."""
        rec = self._app.models["recognition"]
        crops = [_align(image, kps, rec.input_size[0]) for kps in kpss]
        return _normalize(rec.get_feat(crops))

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        boxes, scores, kpss = self._faces(image)
        if len(boxes) == 0:
            return []
        h, w = image.shape[:2]
        threshold = float(ctx.options["threshold"])
        best_name: list[str | None] = [None] * len(boxes)
        best_sim = np.zeros(len(boxes), np.float32)
        if self._names:
            sims = self._embed(image, kpss) @ self._matrix.T  # (faces, samples)
            # Rows are grouped per person (see _rebuild_index): max over each person's contiguous column block.
            per_person = np.maximum.reduceat(sims, self._starts, axis=1)  # (faces, people)
            best = per_person.argmax(axis=1)
            best_sim = per_person[np.arange(len(boxes)), best]
            best_name = [self._names[i] for i in best]

        out: list[Annotation] = []
        for i, (box, kps) in enumerate(zip(boxes, kpss, strict=True)):
            x1, y1, x2, y2 = (float(v) for v in box)
            name = best_name[i]
            out.append(
                Annotation(
                    box=(max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                    conf=float(scores[i]),
                    label=name if name is not None and best_sim[i] >= threshold else UNKNOWN,
                    detail=f"{best_sim[i]:.2f}" if name is not None else None,
                    points=[(float(x), float(y), 1.0) for x, y in kps],
                )
            )
        return out

    # --- commands --------------------------------------------------------------------------------------------

    def command(self, action: str, args: dict[str, Any], ctx: CommandContext) -> CommandResult:
        if action == "enroll":
            return self._enroll(args.get("name"), ctx.frame)
        if action == "list":
            return self._list()
        if action == "delete":
            return self._delete(args.get("key"))
        return super().command(action, args, ctx)

    def _enroll(self, raw_name: Any, frame: np.ndarray | None) -> CommandResult:
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            return CommandResult(ok=False, message="Type a name first")
        if len(name) > MAX_NAME_LEN:
            return CommandResult(ok=False, message=f"Name too long (max {MAX_NAME_LEN} characters)")
        if self._app is None:
            return CommandResult(ok=False, message="Faces is still loading, try again in a moment")
        if frame is None:
            return CommandResult(ok=False, message="No camera frame yet")
        boxes, _, kpss = self._faces(frame)
        if len(boxes) == 0:
            return CommandResult(ok=False, message="No face in view")
        largest = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        embedding = self._embed(frame, kpss[largest : largest + 1])

        # Same person typed with different capitalization → same entry.
        name = next((known for known in self._gallery if known.casefold() == name.casefold()), name)
        samples = np.concatenate([self._gallery.get(name, np.zeros((0, EMBEDDING_DIM), np.float32)), embedding])
        self._gallery[name] = samples[-MAX_SAMPLES:]
        self._save_gallery()
        n = len(self._gallery[name])
        hint = " Add more from other angles for better matching." if n < MAX_SAMPLES else ""
        return CommandResult(ok=True, message=f"Learned {name} ({_samples_label(n)}).{hint}")

    def _list(self) -> CommandResult:
        items = [{"key": name, "label": f"{name} ({_samples_label(len(s))})"} for name, s in self._gallery.items()]
        count = len(items)
        message = "No known faces yet" if not count else f"{count} known face{'' if count == 1 else 's'}"
        return CommandResult(
            ok=True,
            message=message,
            items=items,
            item_action={"id": "delete", "label": "Forget", "confirm": True},
        )

    def _delete(self, key: Any) -> CommandResult:
        if not isinstance(key, str) or key not in self._gallery:
            return CommandResult(ok=False, message=f"Unknown face {key!r}")
        del self._gallery[key]
        self._save_gallery()
        return CommandResult(ok=True, message=f"Forgot {key}")

    # --- gallery persistence ---------------------------------------------------------------------------------

    def _load_gallery(self) -> None:
        self._gallery = {}
        if self.gallery_path.exists():
            with np.load(self.gallery_path, allow_pickle=False) as data:
                names, embeddings = data["names"], _normalize(data["embeddings"])
            for name in dict.fromkeys(names.tolist()):  # keeps first-seen order
                self._gallery[name] = embeddings[names == name][-MAX_SAMPLES:]
            log.info("faces: gallery %s: %d people", self.gallery_path, len(self._gallery))
        self._rebuild_index()

    def _save_gallery(self) -> None:
        self._rebuild_index()
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.gallery_path.with_name(self.gallery_path.name + ".tmp")
        with open(tmp, "wb") as f:  # file object: np.savez must not append its own ".npz" to the tmp name
            np.savez(f, names=np.array([self._names[i] for i in self._owner], dtype=str), embeddings=self._matrix)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.gallery_path)

    def _rebuild_index(self) -> None:
        self._names = list(self._gallery)
        counts = [len(self._gallery[n]) for n in self._names]
        if self._names:
            self._matrix = np.concatenate([self._gallery[n] for n in self._names]).astype(np.float32)
        else:
            self._matrix = np.zeros((0, EMBEDDING_DIM), np.float32)
        self._owner = np.repeat(np.arange(len(self._names)), counts)
        self._starts = np.cumsum([0, *counts[:-1]]).astype(np.intp)  # first column of each person
