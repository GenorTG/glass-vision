"""PP-OCR (PaddleOCR) text detection, recognition and orientation models from their ONNX exports — no PaddlePaddle.

Models: PaddlePaddle's official ONNX exports on Hugging Face (`PaddlePaddle/<model>_onnx`, Apache-2.0), downloaded by
`scripts/download_models.py` into `MODELS_DIR/ppocr/<model>/{inference.onnx,inference.yml}`. The `.yml` next to each
graph carries its preprocessing and, for recognizers, the character dictionary. The graphs run on torch
(`onnx_torch.py` says why not ONNX Runtime).

The pre/post-processing is adapted from PaddleX 3.x (Apache-2.0, Copyright (c) 2024 PaddlePaddle Authors;
https://github.com/PaddlePaddle/PaddleX): `paddlex/inference/models/text_detection/processors.py`
(DetResizeForTest, NormalizeImage, DBPostProcess quad mode), `text_recognition/processors.py` (OCRReisizeNormImg,
CTCLabelDecode), `pipelines/components/common/crop_image_regions.py` (CropByPolys quad: min-area-rect crop with a
perspective warp, tall crops turned 90°) and the image-classification preprocessing of the orientation models.
Kept faithful on purpose so results match PaddleOCR's (`docs/research/bench/`); deviations are commented.

Channel order, as in PaddleX: detector and recognizers read BGR (their `.yml` say `img_mode: BGR`), the orientation
classifiers RGB (PaddleX's classifier reader converts).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from .onnx_torch import OnnxTorch

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# Text detection thresholds of PaddleX's OCR pipeline (paddlex/configs/pipelines/OCR.yaml), which PaddleOCR 3.x uses
# for PP-OCRv6_medium_det: they override the model .yml's (0.2 / 0.45 / 1.4).
DET_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP_RATIO = 1.5
DET_MIN_SIZE = 3
REC_HEIGHT = 48
REC_MIN_WIDTH = 320  # PaddleX pads every crop to at least 48x320
REC_MAX_WIDTH = 3200
REC_BATCH = 32
CLS_BATCH = 32
TALL_RATIO = 1.5  # a crop at least this much taller than wide is vertical text: rotated 90° (PaddleX rule)


def torch_device(device: str) -> torch.device:
    return torch.device(device)


def load_graph(model_dir: Path, device: str, *, half: bool = False) -> OnnxTorch:
    return OnnxTorch(model_dir / "inference.onnx", device, half=half)


def read_config(model_dir: Path) -> dict[str, Any]:
    with (model_dir / "inference.yml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- geometry ---------------------------------------------------------------------------------------------------------
def order_quad(points: np.ndarray) -> np.ndarray:
    """4 points → [top-left, top-right, bottom-right, bottom-left] (PaddleX `get_mini_boxes` ordering)."""
    pts = sorted(points.tolist(), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    return order_quad(cv2.boxPoints(rect)), min(rect[1])


def crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Min-area rectangle around `quad`, perspective-warped upright (PaddleX `get_minarea_rect_crop`). No 90° turn
    here: `Line.turned` decides that (TALL_RATIO)."""
    box, _ = mini_box(quad.astype(np.float32))
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    width, height = max(width, 1), max(height, 1)
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    m = cv2.getPerspectiveTransform(box, target)
    return cv2.warpPerspective(image, m, (width, height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)


# --- detection --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    quad: np.ndarray  # 4x2 float32, pixels of the image passed to `detect`, PaddleX corner order
    score: float


class TextDetector:
    """PP-OCRv6 det (DB): line quads. fp16 on CUDA: 130 → 72 ms at 2560x1920 on the 3090, probability maps within
    0.0015 of fp32 (the recognizers gain nothing from it and stay fp32)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.name = model_dir.name
        self.graph = load_graph(model_dir, device, half=True)
        post = read_config(model_dir).get("PostProcess", {})
        self.max_candidates = int(post.get("max_candidates", 1000))

    def detect(self, image: np.ndarray, max_side: int) -> list[Detection]:
        """`image` BGR; detection runs on it downscaled to `max_side` (multiple of 32), quads come back in `image`'s
        pixels. Deviation from PaddleOCR: it detects at full resolution up to 4000 px (`limit_type: min`)."""
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        rh = max(round(h * scale / 32) * 32, 32)
        rw = max(round(w * scale / 32) * 32, 32)
        resized = image if (rh, rw) == (h, w) else cv2.resize(image, (rw, rh))
        with torch.inference_mode():  # normalization on the device: on a busy CPU it cost as much as the model
            t = torch.from_numpy(np.ascontiguousarray(resized)).to(self.graph.device).float().div_(255.0)
            mean = torch.tensor(IMAGENET_MEAN, device=t.device)
            std = torch.tensor(IMAGENET_STD, device=t.device)
            pred = self.graph.run_tensor(((t - mean) / std).permute(2, 0, 1).unsqueeze(0).contiguous())[0][0, 0]
            bitmap = (pred > DET_THRESH).to(torch.uint8).mul_(255).cpu().numpy()
            scores = pred.float().cpu().numpy()
        return self._boxes(scores, bitmap, w, h)

    def _boxes(self, pred: np.ndarray, bitmap: np.ndarray, dest_w: int, dest_h: int) -> list[Detection]:
        """DBPostProcess.boxes_from_bitmap (quad, score_mode fast, no dilation)."""
        height, width = pred.shape
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for contour in contours[: self.max_candidates]:
            points, side = mini_box(contour)
            if side < DET_MIN_SIZE:
                continue
            score = _box_score_fast(pred, points)
            if score < DET_BOX_THRESH:
                continue
            expanded = _unclip(points, DET_UNCLIP_RATIO)
            if expanded is None:
                continue
            box, side = mini_box(expanded.reshape(-1, 1, 2).astype(np.float32))
            if side < DET_MIN_SIZE + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] * dest_w / width), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] * dest_h / height), 0, dest_h)
            out.append(Detection(box.astype(np.float32), float(score)))
        return out


def _box_score_fast(pred: np.ndarray, box: np.ndarray) -> float:
    h, w = pred.shape
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    shifted = box - np.array([xmin, ymin], np.float32)
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    import pyclipper

    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    if length <= 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(area * ratio / length)
    if len(expanded) != 1:  # PaddleX keeps only single-polygon results
        return None
    return np.array(expanded[0], np.float32)


# --- recognition ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reading:
    text: str
    conf: float  # mean probability of the decoded characters (PaddleX); 0 when nothing was decoded
    chars: int  # decoded characters, incl. spaces


class RecBatches:
    """Line crops resized and batched once (on the device), shared by every recognizer that reads them."""

    def __init__(self, crops: list[np.ndarray], device: torch.device) -> None:
        self.n = len(crops)
        self.chunks: list[tuple[list[int], torch.Tensor]] = []
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(crops[i].shape[0], 1))
        for start in range(0, len(order), REC_BATCH):
            chunk = order[start : start + REC_BATCH]
            self.chunks.append((chunk, _rec_batch([crops[i] for i in chunk], device)))


class TextRecognizer:
    """A PP-OCR CTC recognizer (v6 medium, or a v5 mobile script model)."""

    def __init__(self, model_dir: Path, device: str, *, bidi: bool = False) -> None:
        self.name = model_dir.name
        self.graph = load_graph(model_dir, device)
        config = read_config(model_dir)
        characters = [str(c) for c in config["PostProcess"]["character_dict"]]
        self.characters = ["", *characters, " "]  # CTC blank first, space last (use_space_char)
        self.charset = frozenset(characters) | {" "}
        self.bidi = bidi  # Arabic models emit visual order; PaddleX turns it into logical order with python-bidi

    def prepare(self, crops: list[np.ndarray]) -> RecBatches:
        return RecBatches(crops, self.graph.device)

    def recognize(self, crops: list[np.ndarray] | RecBatches) -> list[Reading]:
        """Upright BGR line crops (or batches prepared from them) → readings. Batched by aspect ratio like PaddleX
        (similar widths share a batch)."""
        batches = crops if isinstance(crops, RecBatches) else self.prepare(crops)
        out: list[Reading | None] = [None] * batches.n
        for chunk, batch in batches.chunks:
            with torch.inference_mode():
                probs = self.graph.run_tensor(batch)[0]
                best, indices = probs.max(dim=-1)  # on the device: v6 has 18 710 classes per time step
            for i, reading in zip(chunk, self._decode(indices.cpu().numpy(), best.float().cpu().numpy()), strict=True):
                out[i] = reading
        return [r for r in out if r is not None]

    def _decode(self, indices: np.ndarray, best: np.ndarray) -> list[Reading]:
        readings = []
        for idx, prob in zip(indices, best, strict=True):
            keep = np.ones(len(idx), bool)
            keep[1:] = idx[1:] != idx[:-1]
            keep &= idx != 0
            text = "".join(self.characters[i] for i in idx[keep])
            if self.bidi and text:
                from bidi import get_display

                text = get_display(text)
            conf = float(prob[keep].mean()) if keep.any() else 0.0
            readings.append(Reading(text, conf, len(text)))
        return readings


def _rec_batch(crops: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """OCRReisizeNormImg + ToBatch: height 48, width by aspect ratio (min 320, max 3200), (x/255 - 0.5)/0.5, padded
    with zeros to the batch's widest crop. Deviation: the batch width is rounded up to a multiple of 64 so the
    recognizer sees fewer distinct input shapes (zero padding is what PaddleX pads with anyway)."""
    widths = []
    for crop in crops:
        h, w = crop.shape[:2]
        widths.append(min(REC_MAX_WIDTH, max(1, math.ceil(REC_HEIGHT * w / max(h, 1)))))
    batch_w = min(REC_MAX_WIDTH, -(-max(REC_MIN_WIDTH, *widths) // 64) * 64)
    canvas = np.zeros((len(crops), REC_HEIGHT, batch_w, 3), np.uint8)
    for k, (crop, w) in enumerate(zip(crops, widths, strict=True)):
        canvas[k, :, :w] = cv2.resize(crop, (w, REC_HEIGHT))
    with torch.inference_mode():
        x = torch.from_numpy(canvas).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        inside = torch.arange(batch_w, device=device)[None, :] < torch.tensor(widths, device=device)[:, None]
        return (x * inside[:, None, None, :]).contiguous()  # padding = 0 after normalization, as PaddleX


# --- orientation ------------------------------------------------------------------------------------------------------
class TextlineOrientation:
    """PP-LCNet_x1_0_textline_ori: is an (upright-looking) line crop upside down? → (180 or 0, score)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.graph = load_graph(model_dir, device)

    def classify(self, crops: list[np.ndarray]) -> list[tuple[int, float]]:
        """Batches are padded to CLS_BATCH crops: one input shape (each new shape costs a cuDNN plan)."""
        out: list[tuple[int, float]] = []
        for start in range(0, len(crops), CLS_BATCH):
            chunk = crops[start : start + CLS_BATCH]
            batch = np.zeros((CLS_BATCH, 3, 80, 160), np.float32)
            batch[: len(chunk)] = [_cls_input(cv2.resize(c, (160, 80))) for c in chunk]
            probs = self.graph.run(batch)[0][: len(chunk)]
            out += [(180 if p.argmax() == 1 else 0, float(p.max())) for p in probs]
        return out


class DocOrientation:
    """PP-LCNet_x1_0_doc_ori: how far the whole image is turned → (0/90/180/270, score). The label is the
    counter-clockwise angle the content is rotated by (rotating the image clockwise by it makes it upright)."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.graph = load_graph(model_dir, device)
        self.labels = [int(v) for v in read_config(model_dir)["PostProcess"]["Topk"]["label_list"]]

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        h, w = image.shape[:2]
        scale = 256 / min(h, w)  # ResizeImage resize_short 256, then CropImage 224 (centre)
        resized = cv2.resize(image, (max(224, round(w * scale)), max(224, round(h * scale))))
        y0, x0 = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
        x = _cls_input(resized[y0 : y0 + 224, x0 : x0 + 224])[None]
        probs = self.graph.run(x)[0][0]
        return self.labels[int(probs.argmax())], float(probs.max())


def _cls_input(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
