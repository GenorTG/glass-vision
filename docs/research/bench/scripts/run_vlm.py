"""VLM OCR via transformers on bench images. argv: images out.json model_key [only,stems]"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

images, out, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
MODELS = {
    "qwen3vl2b": ("Qwen/Qwen3-VL-2B-Instruct", 1280 * 32 * 32,
                  "Read all the text in this image exactly as written, in its original language and script. Do not "
                  "translate or explain. Output one text line per line, in reading order; separate different "
                  "signs/paragraphs with an empty line."),
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", 1280 * 32 * 32, None),
    "pocrvl15": ("PaddlePaddle/PaddleOCR-VL-1.5", 2048 * 28 * 28, "Spotting:"),
    "pocrvl15ocr": ("PaddlePaddle/PaddleOCR-VL-1.5", 1280 * 28 * 28, "OCR:"),
}
repo, max_pixels, prompt = MODELS[key]
if prompt is None:
    prompt = MODELS["qwen3vl2b"][2]
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
processor = AutoProcessor.from_pretrained(repo)
res = {}
for f in sorted(images.glob("*.jpg")):
    if only and f.stem not in only:
        continue
    image = Image.open(f).convert("RGB")
    if key == "pocrvl15" and max(image.size) < 1500:
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    times = []
    for _ in range(2 if f.stem.startswith("render_de_rot") is False else 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        text = processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    ntok = int(gen.shape[-1] - inputs["input_ids"].shape[-1])
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    res[f.stem] = {"ms": round(min(times)), "tokens": ntok, "raw": text,
                   "lines": [{"text": ln, "conf": 1.0, "box": [0, 0, 0, 0]} for ln in lines]}
    print(f.stem, round(min(times)), ntok, text[:220].replace("\n", " ⏎ "), flush=True)
res["_vram_max_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
