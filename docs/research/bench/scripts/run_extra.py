"""(1) doc orientation classifier label+score per image; (2) enhancement variants with PP-OCRv6 (textline ori only)."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import DocImgOrientationClassification, PaddleOCR
from rapidfuzz import fuzz

images, out = Path(sys.argv[1]), Path(sys.argv[2])
keys = json.load(open(Path(__file__).with_name("keys.json")))
res = {"doc_ori": {}, "enhance": {}}

clf = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
for f in sorted(images.glob("*.jpg")):
    r = clf.predict(str(f))[0].json["res"]
    res["doc_ori"][f.stem] = {"label": r["label_names"][0], "score": round(float(r["scores"][0]), 3)}
    print("ori", f.stem, res["doc_ori"][f.stem])


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def up2(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


VARIANTS = {"orig": lambda x: x, "clahe": clahe, "sharpen": sharpen, "clahe+sharpen": lambda x: sharpen(clahe(x)),
            "up2x": up2}
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True,
                text_det_limit_type="max", text_det_limit_side_len=5120)
norm = lambda s: "".join(s.casefold().split())  # noqa: E731
for stem, k in keys.items():
    if k["lang"] in ("ru", "ar"):
        continue
    img = cv2.imread(str(images / f"{stem}.jpg"))
    row = {}
    for name, fn in VARIANTS.items():
        x = fn(img)
        ocr.predict(x)
        t = time.perf_counter()
        j = ocr.predict(x)[0].json["res"]
        ms = (time.perf_counter() - t) * 1000
        text = norm("".join(j["rec_texts"]))
        hits = sum(fuzz.partial_ratio(norm(q), text) >= 90 for q in k["keys"])
        row[name] = {"hits": hits, "of": len(k["keys"]), "ms": round(ms), "n_lines": len(j["rec_texts"])}
    res["enhance"][stem] = row
    print("enh", stem, {n: f'{v["hits"]}/{v["of"]} {v["ms"]}ms' for n, v in row.items()}, flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
