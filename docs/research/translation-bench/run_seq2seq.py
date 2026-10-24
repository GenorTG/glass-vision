"""Seq2seq translators on the test set: M2M100 (as the app runs it), MADLAD-400, Opus-MT.

usage: python -I run_seq2seq.py {m2m418,m2m12,madlad3b,opus} [--mode line|block]
Outputs results/<name>.jsonl ({id, hyp, ms}) + .meta.json (load time, VRAM, disk). Every item is translated on its
own (one block = one call, its lines batched), after 5 warm-up calls, like the app does per sign.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from common import MODELS, gpu_mem_of, is_translatable, load_testset, normalize, segments, write_results  # noqa: E402

GV_MODELS = Path("/home/genorbox1/projects/glass-vision/server/models/translation")  # read only
NUM_BEAMS = 4  # the app's setting
DEV = "cuda"


def dir_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


class M2M:
    def __init__(self, path: Path):
        from transformers import AutoTokenizer, M2M100ForConditionalGeneration

        self.tok = AutoTokenizer.from_pretrained(path)
        self.net = M2M100ForConditionalGeneration.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
        self.disk = dir_size(path)

    def supports(self, src, tgt):
        return True

    def __call__(self, texts, src, tgt):
        self.tok.src_lang = src
        batch = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
        with torch.inference_mode():
            out = self.net.generate(**batch, forced_bos_token_id=self.tok.get_lang_id(tgt), num_beams=NUM_BEAMS,
                                    max_length=min(256, 2 * batch.input_ids.shape[1] + 10))
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]


class Madlad:
    def __init__(self, path: Path):
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tok = AutoTokenizer.from_pretrained(path)
        self.net = T5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
        self.disk = dir_size(path)

    def supports(self, src, tgt):
        return True

    def __call__(self, texts, src, tgt):
        batch = self.tok([f"<2{tgt}> {t}" for t in texts], return_tensors="pt", padding=True).to(DEV)
        with torch.inference_mode():
            out = self.net.generate(**batch, num_beams=NUM_BEAMS, max_new_tokens=min(256, 2 * batch.input_ids.shape[1] + 10))
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]


# Opus-MT repos per pair (Helsinki-NLP/opus-mt-*); en-zlw needs a target token. Missing pairs = not covered.
OPUS = {f"{s}-en": f"opus-mt-{s}-en" for s in "de fr es it ru uk ja zh ko ar tr nl cs pl".split()}
OPUS.update({p: f"opus-mt-{p}" for p in "de-pl fr-pl es-pl uk-pl ja-pl ar-pl pl-de en-de fr-de ja-de en-es pl-es de-es".split()})
OPUS["en-pl"] = "opus-mt-en-zlw"
OPUS_PREFIX = {"en-pl": ">>pol<< "}


class Opus:
    def __init__(self, root: Path):
        from huggingface_hub import snapshot_download

        self.root, self.models, self.disk = root, {}, 0.0
        for pair, repo in OPUS.items():
            p = snapshot_download(f"Helsinki-NLP/{repo}", local_dir=root / repo,
                                  allow_patterns=["*.json", "*.spm", "*.model", "pytorch_model.bin", "*.safetensors", "vocab*", "*.txt"])
            self.disk += dir_size(Path(p))
        self.loaded: dict = {}

    def supports(self, src, tgt):
        return f"{src}-{tgt}" in OPUS

    def load_all(self):
        from transformers import MarianMTModel, MarianTokenizer

        for pair, repo in OPUS.items():
            path = self.root / repo
            self.loaded[pair] = (MarianTokenizer.from_pretrained(path),
                                 MarianMTModel.from_pretrained(path, dtype=torch.float16).to(DEV).eval())

    def __call__(self, texts, src, tgt):
        pair = f"{src}-{tgt}"
        tok, net = self.loaded[pair]
        batch = tok([OPUS_PREFIX.get(pair, "") + t for t in texts], return_tensors="pt", padding=True).to(DEV)
        with torch.inference_mode():
            out = net.generate(**batch, num_beams=NUM_BEAMS, max_new_tokens=min(256, 2 * batch.input_ids.shape[1] + 10))
        return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]


def translate_item(model, item, mode):
    segs = segments(item, mode)
    texts = [normalize(s) for s in segs]
    todo = [i for i, t in enumerate(texts) if is_translatable(t)]
    out = list(segs)
    if todo:
        for i, t in zip(todo, model([texts[i] for i in todo], item["src"], item["tgt"])):
            out[i] = t
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["m2m418", "m2m12", "madlad3b", "opus"])
    ap.add_argument("--mode", default="block", choices=["line", "block"])
    a = ap.parse_args()
    items = load_testset()
    torch.cuda.init()
    base = gpu_mem_of(os.getpid())
    t0 = time.perf_counter()
    if a.system == "m2m418":
        model = M2M(GV_MODELS / "m2m100_418M")
    elif a.system == "m2m12":
        model = M2M(GV_MODELS / "m2m100_1.2B")
    elif a.system == "madlad3b":
        model = Madlad(MODELS / "madlad400-3b-mt")
    else:
        model = Opus(MODELS / "opus-mt")
        t0 = time.perf_counter()  # don't count the downloads
        model.load_all()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    covered = [it for it in items if model.supports(it["src"], it["tgt"])]
    for it in covered[:5]:  # warm-up
        translate_item(model, it, a.mode)
    rows = []
    for it in items:
        if not model.supports(it["src"], it["tgt"]):
            rows.append({"id": it["id"], "hyp": None, "ms": None})
            continue
        torch.cuda.synchronize()
        t = time.perf_counter()
        hyp = translate_item(model, it, a.mode)
        torch.cuda.synchronize()
        rows.append({"id": it["id"], "hyp": hyp, "ms": round((time.perf_counter() - t) * 1000, 1)})
    name = {"m2m418": "m2m100-418M", "m2m12": "m2m100-1.2B", "madlad3b": "madlad400-3b", "opus": "opus-mt"}[a.system]
    name += f"-{a.mode}"
    meta = {"system": name, "load_s": round(load_s, 1), "vram_mib": gpu_mem_of(os.getpid()) - base,
            "vram_mib_process": gpu_mem_of(os.getpid()), "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "disk_gb": round(model.disk, 2), "covered": len(covered), "items": len(items), "num_beams": NUM_BEAMS,
            "mode": a.mode, "precision": "bf16" if a.system == "madlad3b" else "fp16", "runtime": "transformers"}
    write_results(name, rows, meta)
    print(meta)


if __name__ == "__main__":
    main()
