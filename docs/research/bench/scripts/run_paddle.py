"""PaddleOCR 3.x pipelines on the bench images. argv: images_dir out.json config"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

images, out, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
keys = json.load(open(Path(__file__).with_name("keys.json")))
LANG_FOR = {k: v["lang"] for k, v in keys.items()}
LANG_FOR.update({"render_ru": "ru", "render_pl_paragraph": "pl", "render_pl_tilt15": "pl"})

common = dict(use_doc_orientation_classify=True, use_doc_unwarping=False, use_textline_orientation=True)
if cfg == "v6":  # default PaddleOCR(): PP-OCRv6_medium det+rec (zh/en/ja + Latin langs)
    make = lambda lang: PaddleOCR(**common)  # noqa: E731
elif cfg == "v6_noori":
    make = lambda lang: PaddleOCR(  # noqa: E731
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
elif cfg == "v6_tl":  # textline 0/180 classifier only, no whole-image orientation classifier
    make = lambda lang: PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,  # noqa: E731
                                  use_textline_orientation=True)
elif cfg == "v6_unwarp":
    make = lambda lang: PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True,  # noqa: E731
                                  use_textline_orientation=True)
elif cfg == "v5server":  # PP-OCRv5 server det + server rec (zh/zh-Hant/en/ja)
    make = lambda lang: PaddleOCR(ocr_version="PP-OCRv5", lang="ch", **common)  # noqa: E731
elif cfg == "v5lang":  # per-image language → script-specific PP-OCRv5 rec (oracle language choice)
    make = lambda lang: PaddleOCR(ocr_version="PP-OCRv5", lang=lang, use_doc_orientation_classify=False,
                                  use_doc_unwarping=False, use_textline_orientation=True)  # noqa: E731
elif cfg == "v6det_v5rec":  # PP-OCRv6 line detector + script-specific recognizer where v6 has no script
    REC = {"ru": "cyrillic_PP-OCRv5_mobile_rec", "ar": "arabic_PP-OCRv5_mobile_rec"}
    make = lambda lang: PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",  # noqa: E731
                                  text_recognition_model_name=REC.get(lang, "PP-OCRv6_medium_rec"),
                                  use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=True)
else:
    raise SystemExit(cfg)

cache: dict = {}
res = {}
for f in sorted(images.glob("*.jpg")):
    stem = f.stem
    lang = LANG_FOR.get(stem, "de" if stem.startswith("render_de") else "ch")
    if cfg == "v5lang" and lang in ("chinese_cht", "japan"):
        lang = "ch"
    key = lang if cfg == "v5lang" else ("ru" if lang == "ru" else "ar" if lang == "ar" else "one") if cfg == "v6det_v5rec" else "one"
    if key not in cache:
        t = time.perf_counter()
        cache[key] = make(lang)
        print("init", key, round(time.perf_counter() - t, 1), "s")
    ocr = cache[key]
    img = cv2.imread(str(f))
    ocr.predict(img)  # warm
    times = []
    for _ in range(2):
        t = time.perf_counter()
        r = ocr.predict(img)[0]
        times.append((time.perf_counter() - t) * 1000)
    j = r.json["res"]
    angle = (j.get("doc_preprocessor_res") or {}).get("angle")
    lines = []
    for text, score, box in zip(j["rec_texts"], j["rec_scores"], j["rec_boxes"]):
        if text.strip():
            lines.append({"text": text, "conf": round(float(score), 3), "box": [float(v) for v in box]})
    tl_angles = j.get("textline_orientation_angles")
    res[stem] = {"ms": round(min(times), 1), "doc_angle": angle, "lang": lang, "lines": lines,
                 "textline_180": int(sum(1 for a in (tl_angles or []) if a == 1))}
    print(stem, lang, angle, round(min(times)), " | ".join(x["text"] for x in lines)[:160], flush=True)
res["_vram_max_alloc_mb"] = paddle.device.cuda.max_memory_allocated() / 2**20
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
