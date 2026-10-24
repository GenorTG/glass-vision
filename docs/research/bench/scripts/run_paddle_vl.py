"""PaddleOCR-VL (native Paddle backend) on whole photos, no layout: prompt 'spotting' (text + quads) and 'ocr'.
argv: images out.json version"""

import json
import sys
import time
from pathlib import Path

import cv2
import paddle
from paddleocr import PaddleOCRVL

images, out, version = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
vl = PaddleOCRVL(pipeline_version=version, use_layout_detection=False, use_doc_orientation_classify=False,
                 use_doc_unwarping=False)
res = {}
for f in sorted(images.glob("*.jpg")):
    if only and f.stem not in only:
        continue
    img = cv2.imread(str(f))
    entry = {}
    for prompt in ("spotting", "ocr"):
        t = time.perf_counter()
        try:
            r = list(vl.predict(img, prompt_label=prompt))[0]
        except Exception as exc:  # noqa: BLE001
            entry[prompt] = {"error": repr(exc)}
            print(f.stem, prompt, "ERROR", exc, flush=True)
            continue
        ms = (time.perf_counter() - t) * 1000
        j = r.json["res"]
        sp = j.get("spotting_res") or {}
        blocks = [b.get("block_content", "") for b in j.get("parsing_res_list", [])]
        entry[prompt] = {"ms": round(ms), "texts": sp.get("rec_texts"), "blocks": blocks}
        shown = sp.get("rec_texts") if prompt == "spotting" else blocks
        print(f.stem, prompt, round(ms), " | ".join(map(str, shown or []))[:200].replace("\n", " ⏎ "), flush=True)
    # scoring view: spotting lines (fallback: ocr text lines)
    sp = entry.get("spotting", {})
    lines = sp.get("texts") or "\n".join(entry.get("ocr", {}).get("blocks") or []).split("\n")
    res[f.stem] = {"ms": sp.get("ms", entry.get("ocr", {}).get("ms", 0)), "raw": entry,
                   "lines": [{"text": t, "conf": 1.0, "box": [0, 0, 0, 0]} for t in lines if t and t.strip()]}
res["_vram_max_alloc_mb"] = paddle.device.cuda.max_memory_allocated() / 2**20
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
