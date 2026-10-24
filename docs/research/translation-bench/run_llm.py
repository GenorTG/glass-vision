"""LLM translators on the test set, served by llama.cpp's llama-server (CUDA, all layers on the GPU).

usage: python -I run_llm.py <model-key> [--variant plain|sign] [--port 8791]
Starts its own llama-server (GPU 0, own port), measures cold load (spawn → /health ok), VRAM of that process, then
sends every block as one request (sequential, after warm-up) and finally 8 blocks concurrently (a photo with several
blocks). Greedy decoding everywhere (temperature 0), thinking off.

Variants: plain = the model card's own translation prompt; sign = the same plus what Glass Vision knows about the text
(OCR from a photo, an image-caption-like hint, neighbouring blocks, "translate as such a sign is written"); ctx = the
model card's prompt plus only a one-line hint (what the photo shows + neighbouring blocks), no rules; signauto = sign
without naming the source language (what an app without language detection would send).
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, NAMES, NAMES_ZH, WORK, gpu_mem_of, load_testset, write_results  # noqa: E402

LLAMA_SERVER = "/home/genorbox1/projects/local-agentic-inference-provider/llama.cpp/build/bin/llama-server"  # a25c986
LLAMA_SERVER_OLD = "/home/genorbox1/llama-cpp-server/build/bin/llama-server"  # 3ce7da2 (2026-07-22)

# key → (gguf path under MODELS, prompt family, label)
SPECS = {
    "hymt2-1.8b": ("Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf", "hymt", "Hy-MT2-1.8B Q8_0"),
    "hymt2-7b": ("Hy-MT2-7B-GGUF/Hy-MT2-7B-Q4_K_M.gguf", "hymt", "Hy-MT2-7B Q4_K_M"),
    "index-2b": ("Index-Translate-2B-GGUF/Index-Translate-2B.Q8_0.gguf", "index", "Index-Translate-2B Q8_0"),
    "index-9b": ("Index-Translate-9B-GGUF/Index-Translate-9B.Q4_K_M.gguf", "index", "Index-Translate-9B Q4_K_M"),
    "qwen35-4b": ("Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf", "generic", "Qwen3.5-4B Q8_0"),
    "gemma4-e2b": ("gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf", "generic", "Gemma 4 E2B Q8_0"),
    "gemma4-e4b": ("gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf", "generic", "Gemma 4 E4B Q4_K_M"),
    "translategemma-4b": ("translategemma-4b-it-GGUF/translategemma-4b-it.Q8_0.gguf", "tgemma", "TranslateGemma-4B Q8_0"),
}


# llama.cpp a25c986: Gemma 4 E4B with flash attention + 4 slots dies with "CUDA error: an illegal memory access" on
# the first request (works with -fa off, with -np 1, and with build 3ce7da2).
# Also with -fa off it crashed at random later in the run, so E4B runs on build 3ce7da2 (stable on this host).
FLASH_ATTN = {"gemma4-e4b": "off"}
OLD_BUILD = {"gemma4-e4b"}


def context_lines(item: dict) -> list[str]:
    out = []
    if item.get("ctx"):
        out.append(f"Photo: {item['ctx']}.")
    if item.get("nb"):
        out.append("Other text on the same photo (for context only, do not translate it): " + " | ".join(item["nb"]))
    return out


SIGN_RULES = (
    "The text was read by OCR from a phone photo of a sign, label, menu, notice, screen or document. Translate it the "
    "way such a text is normally written in {tgt} (a door sign \"Ausgang\" is \"Exit\", not \"Output\"; \"Drücken\" on "
    "a door is \"Push\"). Keep the line breaks (one output line per input line), numbers, prices, times, codes and "
    "proper names. The OCR may contain small errors: translate what was meant. Output only the translation."
)


DEFAULT_CTX = "a sign, label, menu or notice"


def short_hint(item: dict) -> str:
    hint = f"Text read by OCR from a phone photo ({item.get('ctx', DEFAULT_CTX)})."
    if item.get("nb"):
        hint += " Other text on the same photo: " + " | ".join(item["nb"]) + "."
    return hint


def generic_messages(item: dict, variant: str) -> list[dict]:
    src, tgt = NAMES[item["src"]], NAMES[item["tgt"]]
    if variant == "ctx":
        return [{"role": "user", "content": f"Translate the following text from {src} to {tgt}. {short_hint(item)} "
                 f"Output only the translation.\n\n{item['text']}"}]
    if variant == "plain":
        return [{"role": "user", "content": f"Translate the following text from {src} to {tgt}. Output only the "
                 f"translation.\n\n{item['text']}"}]
    head = f"Translate into {tgt}." if variant == "signauto" else f"Translate from {src} to {tgt}."
    parts = [head, SIGN_RULES.format(tgt=tgt), *context_lines(item), "", "Text:", item["text"]]
    return [{"role": "user", "content": "\n".join(parts)}]


def hymt_messages(item: dict, variant: str) -> list[dict]:
    tgt = NAMES[item["tgt"]]
    if variant == "ctx":  # model card "Structured Data 2" (background information) with a one-line background
        return [{"role": "user", "content": f"[Background Information]\n{short_hint(item)}\n\nPlease translate the "
                 f"following text into {tgt}, taking the provided background information into consideration.\n\n"
                 f"[Source Text]\n{item['text']}"}]
    if variant == "plain":  # model card "Default Translation" (English prompt)
        return [{"role": "user", "content": f"Translate the following text into {tgt}. Note that you should only "
                 f"output the translated result without any additional explanation:\n\n{item['text']}"}]
    background = [SIGN_RULES.format(tgt=tgt), *context_lines(item)]
    # model card "Structured Data 2" (background information) template
    return [{"role": "user", "content": "[Background Information]\n" + "\n".join(background) + "\n\nPlease translate "
             f"the following text into {tgt}, taking the provided background information into consideration.\n\n"
             f"[Source Text]\n{item['text']}"}]


def index_messages(item: dict, variant: str) -> list[dict]:
    src, tgt = NAMES_ZH[item["src"]], NAMES_ZH[item["tgt"]]
    if variant == "ctx":  # model card instTrans format, genre = sign text, one context constraint
        return [{"role": "user", "content": f"请将以下{src}标识文字翻译成{tgt}，并且严格遵循所有约束要求。\n【源文】\n"
                 f"{item['text']}\n【约束要求】\n1. 【注意】背景：{short_hint(item)}\n只输出译文，不要有任何额外说明。"}]
    if variant == "plain":  # model card standard prompt
        return [{"role": "user", "content": f"请将以下{src}文本翻译为{tgt}，直接输出翻译结果，不要进行任何解释。\n{item['text']}"}]
    rules = [
        "【注意】源文是用手机拍照后OCR识别出的标识、标签、菜单、告示、屏幕或文件上的文字，可能有少量识别错误，请按原意翻译。",
        f"【注意】按照这类文字在{tgt}中的惯用写法翻译（例如德语门上的“Ausgang”应译为“Exit”，“Drücken”应译为“Push”）。",
        "【硬性要求】保留换行（每行输入对应一行输出），保留数字、价格、时间、代码和专有名词。",
    ]
    if item.get("ctx"):
        rules.append(f"【注意】照片内容：{item['ctx']}。")
    if item.get("nb"):
        rules.append("【注意】同一照片上的其他文字（仅供参考，不要翻译）：" + " | ".join(item["nb"]))
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    src = "" if variant == "signauto" else src  # model card: source "auto" omits the source-language name
    return [{"role": "user", "content": f"请将以下{src}标识文字翻译成{tgt}，并且严格遵循所有约束要求。\n【源文】\n"
             f"{item['text']}\n【约束要求】\n{numbered}\n只输出译文，不要有任何额外说明。"}]


def tgemma_prompt(item: dict) -> str:
    """TranslateGemma's own chat template (from the GGUF's tokenizer.chat_template), rendered by hand: it takes
    source_lang_code / target_lang_code fields that an OpenAI chat request can't carry."""
    s, t = item["src"], item["tgt"]
    src, tgt = NAMES[s], NAMES[t]
    return (f"<start_of_turn>user\nYou are a professional {src} ({s}) to {tgt} ({t}) translator. Your goal is to "
            f"accurately convey the meaning and nuances of the original {src} text while adhering to {tgt} grammar, "
            f"vocabulary, and cultural sensitivities.\nProduce only the {tgt} translation, without any additional "
            f"explanations or commentary. Please translate the following {src} text into {tgt}:\n\n\n"
            f"{item['text'].strip()}<end_of_turn>\n<start_of_turn>model\n")


def post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    return THINK.sub("", text).strip()


class Server:
    def __init__(self, gguf: Path, port: int, log: Path, jinja: bool = True, fa: str = "on", old: bool = False, np: int = 4):
        self.url = f"http://127.0.0.1:{port}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        cmd = [LLAMA_SERVER_OLD if old else LLAMA_SERVER, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "999",
               "-c", str(2048 * np), "-np", str(np), "-fa", fa, "--fit", "off", "--no-webui", "--reasoning", "off",
               "--jinja" if jinja else "--no-jinja", "--cache-ram", "0"]
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited, see {log}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.1)
        self.load_s = time.perf_counter() - self.t0

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(SPECS))
    ap.add_argument("--variant", default="plain", choices=["plain", "sign", "ctx", "signauto"])
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--np", type=int, default=4, help="server slots (the concurrent photo test always sends 8)")
    a = ap.parse_args()
    rel, family, label = SPECS[a.model]
    gguf = MODELS / rel
    items = load_testset()
    (WORK / "logs").mkdir(exist_ok=True)
    # TranslateGemma's template only takes its own content format: raw /completion with the prompt rendered here
    srv = Server(gguf, a.port, WORK / "logs" / f"llama-{a.model}-{a.variant}.log", jinja=family != "tgemma",
                 fa=FLASH_ATTN.get(a.model, "on") if a.np > 1 else "on", old=a.model in OLD_BUILD and a.np > 1, np=a.np)
    try:
        def run(item):
            if family == "tgemma":
                body = {"prompt": tgemma_prompt(item), "temperature": 0, "n_predict": 256, "cache_prompt": False}
                t = time.perf_counter()
                r = post(srv.url + "/completion", body)
                return clean(r["content"]), (time.perf_counter() - t) * 1000, r.get("tokens_predicted")
            msgs = {"generic": generic_messages, "hymt": hymt_messages, "index": index_messages}[family](item, a.variant)
            body = {"messages": msgs, "temperature": 0, "max_tokens": 256, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if family == "index":  # instTrans prompts: the 2B goes on to translate the constraint list
                body["stop"] = ["\n【"]
            t = time.perf_counter()
            r = post(srv.url + "/v1/chat/completions", body)
            return (clean(r["choices"][0]["message"]["content"] or ""), (time.perf_counter() - t) * 1000,
                    r.get("usage", {}).get("completion_tokens"))

        for it in items[:5]:
            run(it)
        rows = []
        for it in items:
            hyp, ms, ntok = run(it)
            rows.append({"id": it["id"], "hyp": hyp, "ms": round(ms, 1), "tokens": ntok})
        # a photo with 8 blocks, sent concurrently (4 server slots)
        multi = [it for it in items if it["cat"] in ("word", "short", "menu", "doc", "warn")][:8]
        walls = []
        for _ in range(3):
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(8) as ex:
                list(ex.map(run, multi))
            walls.append(round((time.perf_counter() - t) * 1000, 1))
        name = f"{a.model}-{a.variant}" + (f"-np{a.np}" if a.np != 4 else "")
        meta = {"system": name, "label": label, "gguf": rel, "disk_gb": round(gguf.stat().st_size / 1e9, 2),
                "load_s": round(srv.load_s, 1), "vram_mib_process": gpu_mem_of(srv.proc.pid),
                "server_args": f"-ngl 999 -c {2048 * a.np} -np {a.np} -fa {FLASH_ATTN.get(a.model, 'on') if a.np > 1 else 'on'} --reasoning off",
                "runtime": "llama.cpp " + ("3ce7da2" if a.model in OLD_BUILD and a.np > 1 else "a25c986"),
                "variant": a.variant, "photo8_concurrent_ms": walls, "items": len(items)}
        write_results(name, rows, meta)
        print(meta)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
