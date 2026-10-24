"""Language-coverage probe: does a model actually write each of the app's 43 languages (server/glass_vision/translate.py
LANGUAGES)? Three English notices → every language, each model's plain prompt; the output language is checked with
lingua (wrong language / copy of the source = off-target). Languages lingua can't identify are only checked for "not
English / not empty".

usage: python -I run_coverage.py <model-key|m2m418|m2m12|madlad3b> ...   → results/coverage.json (merged)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import RESULTS, WORK  # noqa: E402

APP = {  # server/glass_vision/translate.py LANGUAGES (code → English name, Chinese name for Index-Translate)
    "ar": ("Arabic", "阿拉伯语"), "be": ("Belarusian", "白俄罗斯语"), "bn": ("Bengali", "孟加拉语"),
    "bg": ("Bulgarian", "保加利亚语"), "zh": ("Chinese", "中文"), "hr": ("Croatian", "克罗地亚语"),
    "cs": ("Czech", "捷克语"), "da": ("Danish", "丹麦语"), "nl": ("Dutch", "荷兰语"), "et": ("Estonian", "爱沙尼亚语"),
    "fi": ("Finnish", "芬兰语"), "fr": ("French", "法语"), "de": ("German", "德语"), "el": ("Greek", "希腊语"),
    "he": ("Hebrew", "希伯来语"), "hi": ("Hindi", "印地语"), "hu": ("Hungarian", "匈牙利语"),
    "id": ("Indonesian", "印尼语"), "it": ("Italian", "意大利语"), "ja": ("Japanese", "日语"), "ko": ("Korean", "韩语"),
    "lv": ("Latvian", "拉脱维亚语"), "lt": ("Lithuanian", "立陶宛语"), "mr": ("Marathi", "马拉地语"),
    "ne": ("Nepali", "尼泊尔语"), "no": ("Norwegian", "挪威语"), "fa": ("Persian", "波斯语"), "pl": ("Polish", "波兰语"),
    "pt": ("Portuguese", "葡萄牙语"), "ro": ("Romanian", "罗马尼亚语"), "ru": ("Russian", "俄语"),
    "sr": ("Serbian", "塞尔维亚语"), "sk": ("Slovak", "斯洛伐克语"), "sl": ("Slovenian", "斯洛文尼亚语"),
    "es": ("Spanish", "西班牙语"), "sv": ("Swedish", "瑞典语"), "ta": ("Tamil", "泰米尔语"), "th": ("Thai", "泰语"),
    "tr": ("Turkish", "土耳其语"), "uk": ("Ukrainian", "乌克兰语"), "ur": ("Urdu", "乌尔都语"),
    "vi": ("Vietnamese", "越南语"),
}
TEXTS = [
    "Emergency exit. Keep this door closed at all times.",
    "Please do not touch the exhibits.",
    "The shop is closed on Sundays and public holidays.",
]


def detector():
    from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

    langs, to_code = [], {}
    for code in [*APP, "en"]:
        isos = ["NB", "NN"] if code == "no" else [code.upper()]
        for iso in isos:
            if hasattr(IsoCode639_1, iso):
                lang = Language.from_iso_code_639_1(getattr(IsoCode639_1, iso))
                langs.append(lang)
                to_code[lang] = code
    det = LanguageDetectorBuilder.from_languages(*langs).build()
    return det, to_code


def main():
    # names for the prompts of run_llm.py / run_seq2seq.py
    common.NAMES.update({c: n for c, (n, _) in APP.items()})
    common.NAMES_ZH.update({c: z for c, (_, z) in APP.items()})
    import run_llm

    det, to_code = detector()
    out_path = RESULTS / "coverage.json"
    cov = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key in sys.argv[1:]:
        items = [{"src": "en", "tgt": c, "text": t, "kind": "prose"} for c in APP if c != "en" for t in TEXTS]
        if key in run_llm.SPECS:
            rel, family, _ = run_llm.SPECS[key]
            srv = run_llm.Server(common.MODELS / rel, 8792, WORK / "logs" / f"coverage-{key}.log", jinja=family != "tgemma")
            try:
                hyps = []
                for it in items:
                    if family == "tgemma":
                        r = run_llm.post(srv.url + "/completion", {"prompt": run_llm.tgemma_prompt(it), "temperature": 0,
                                                                     "n_predict": 128})
                        hyps.append(run_llm.clean(r["content"]))
                    else:
                        msgs = {"generic": run_llm.generic_messages, "hymt": run_llm.hymt_messages,
                                "index": run_llm.index_messages}[family](it, "plain")
                        r = run_llm.post(srv.url + "/v1/chat/completions", {"messages": msgs, "temperature": 0,
                                         "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}})
                        hyps.append(run_llm.clean(r["choices"][0]["message"]["content"] or ""))
            finally:
                srv.stop()
        else:
            import torch

            import run_seq2seq as s2s

            model = {"m2m418": lambda: s2s.M2M(s2s.GV_MODELS / "m2m100_418M"),
                     "m2m12": lambda: s2s.M2M(s2s.GV_MODELS / "m2m100_1.2B"),
                     "madlad3b": lambda: s2s.Madlad(common.MODELS / "madlad400-3b-mt")}[key]()
            hyps = [model([it["text"]], "en", it["tgt"])[0]
                    for it in items]
            del model
            torch.cuda.empty_cache()
        per_lang: dict[str, dict] = {}
        for it, hyp in zip(items, hyps):
            got = det.detect_language_of(hyp) if hyp.strip() else None
            got_code = to_code.get(got) if got is not None else None
            checkable = any(to_code[lang] == it["tgt"] for lang in to_code)
            ok = (got_code == it["tgt"]) if checkable else (bool(hyp.strip()) and got_code != "en")
            d = per_lang.setdefault(it["tgt"], {"ok": 0, "n": 0, "checked_by_lingua": checkable, "samples": []})
            d["ok"] += int(ok)
            d["n"] += 1
            if not ok:
                d["samples"].append({"hyp": hyp[:120], "detected": got_code})
        weak = sorted(c for c, d in per_lang.items() if d["ok"] < d["n"])
        cov[key] = {"off_target_languages": weak, "ok": sum(d["ok"] for d in per_lang.values()),
                    "n": sum(d["n"] for d in per_lang.values()), "per_lang": per_lang}
        print(key, cov[key]["ok"], "/", cov[key]["n"], "weak:", weak)
        out_path.write_text(json.dumps(cov, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
