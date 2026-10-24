"""RapidOCR 3.x (ONNX Runtime CUDA EP), PP-OCRv5 server det + rec. argv: images out.json"""

import json
import sys
import time
from pathlib import Path

import cv2
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

images, out = Path(sys.argv[1]), Path(sys.argv[2])
gpu = len(sys.argv) < 4 or sys.argv[3] != "cpu"


def make(lang_rec):
    return RapidOCR(params={
        "Global.max_side_len": 2560,
        "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.CH,
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER,
        "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": lang_rec,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.SERVER if lang_rec == LangRec.CH else ModelType.MOBILE,
        "EngineConfig.onnxruntime.use_cuda": gpu,
        "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
    })


engines = {"ch": make(LangRec.CH)}
res = {}
for f in sorted(images.glob("*.jpg")):
    lang = "cyrillic" if "ru" in f.stem.split("_") or f.stem.startswith("Moscow") else "ch"
    if lang not in engines:
        engines[lang] = make(LangRec.CYRILLIC)
    eng = engines[lang]
    img = cv2.imread(str(f))
    eng(img)
    times = []
    for _ in range(2):
        t = time.perf_counter()
        r = eng(img)
        times.append((time.perf_counter() - t) * 1000)
    lines = []
    if r.boxes is not None:
        for box, text, score in zip(r.boxes, r.txts, r.scores):
            xs, ys = box[:, 0], box[:, 1]
            lines.append({"text": text, "conf": round(float(score), 3),
                          "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]})
    res[f.stem] = {"ms": round(min(times), 1), "lines": lines}
    print(f.stem, round(min(times)), " | ".join(x["text"] for x in lines)[:150], flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
