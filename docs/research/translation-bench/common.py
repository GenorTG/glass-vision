"""Shared helpers for the translation benchmark (test set, language names, text normalisation as the app does it)."""

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset.jsonl"
RESULTS = HERE / "results"
WORK = Path("/home/genorbox1/.openclaw/workspace/.tmp/translation-bench")
MODELS = WORK / "models"

NAMES = {
    "ar": "Arabic", "cs": "Czech", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "nl": "Dutch", "pl": "Polish", "ru": "Russian",
    "tr": "Turkish", "uk": "Ukrainian", "zh": "Chinese",
}
NAMES_ZH = {
    "ar": "阿拉伯语", "cs": "捷克语", "de": "德语", "en": "英语", "es": "西班牙语", "fr": "法语", "it": "意大利语",
    "ja": "日语", "ko": "韩语", "nl": "荷兰语", "pl": "波兰语", "ru": "俄语", "tr": "土耳其语", "uk": "乌克兰语",
    "zh": "中文",
}


def load_testset() -> list[dict]:
    return [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]


def letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def normalize(text: str) -> str:
    """Same as server/glass_vision/translate.py: collapse whitespace, ALL-CAPS → sentence case."""
    text = " ".join(text.split())
    if letter_count(text) >= 3 and text == text.upper() and text != text.lower():
        text = text[:1] + text[1:].lower()
    return text


_URL = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\w+|^[\w.-]+\.[a-z]{2,6}(/\S*)?$", re.IGNORECASE)


def is_translatable(text: str) -> bool:
    """Same as server/glass_vision/translate.py: numbers / codes / URLs stay as read."""
    letters = letter_count(text)
    if letters < 2 or sum(ch.isdigit() for ch in text) >= letters or _URL.search(text):
        return False
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    return not all(any(ch.isdigit() for ch in w) for w in words)


def segments(item: dict, mode: str) -> list[str]:
    """What a seq2seq model gets: 'line' = every line on its own (today's app); 'block' = prose joined into one
    segment, list blocks (menus, timetables, UI) still per line."""
    lines = item["text"].split("\n")
    if mode == "block" and item["kind"] == "prose":
        return [" ".join(lines)]
    return lines


def gpu_mem_of(pid: int) -> int:
    """MiB used on the GPU by `pid` (nvidia-smi), 0 if none."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        p, mem = (x.strip() for x in line.split(","))
        if p == str(pid):
            return int(mem)
    return 0


def write_results(name: str, rows: list[dict], meta: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
