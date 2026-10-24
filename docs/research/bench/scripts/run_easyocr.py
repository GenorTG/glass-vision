"""Glass Vision's own server OCR (EasyOCR, auto script) via TextPipeline.read_still. Run with server/run."""

import json
import sys
import time
from pathlib import Path

import cv2
import torch

from glass_vision.pipelines.text import AUTO_SCRIPTS, TextPipeline

images, out = Path(sys.argv[1]), Path(sys.argv[2])
p = TextPipeline()
t = time.perf_counter()
p.load("cuda")
print("load s", round(time.perf_counter() - t, 1))
res = {}
for f in sorted(images.glob("*.jpg")):
    img = cv2.imread(str(f))
    r = p.read_still(img, AUTO_SCRIPTS)  # cold (loads any script recognizer)
    times = []
    for _ in range(2):
        r = p.read_still(img, AUTO_SCRIPTS)
        times.append(r["ms"]["ocr"])
    w, h = r["w"], r["h"]
    res[f.stem] = {
        "ms": min(times),
        "script": r["script"],
        "lines": [
            {"text": ln["text"], "conf": ln["conf"], "block": ln["block"],
             "box": [ln["box"][0] * w, ln["box"][1] * h, ln["box"][2] * w, ln["box"][3] * h]}
            for ln in r["lines"]
        ],
    }
    print(f.stem, r["script"], min(times), " | ".join(ln["text"] for ln in r["lines"])[:160])
res["_vram_max_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
