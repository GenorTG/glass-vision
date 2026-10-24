"""Blind manual review of the top systems on 40 items.

usage: python -I review.py make SYS1 SYS2 SYS3   → results/review_sheet.md (outputs shuffled per item as A/B/C,
                                                  identical outputs merged) + results/review_key.json
       python -I review.py score                 → reads results/review_ratings.json {item id: {"A": 2, ...}},
                                                  prints/writes per-system adequacy (results/review_scores.json)
Rating: 2 = correct and natural for this kind of text (what the sign would say), 1 = understandable but literal /
partly wrong / wrong register, 0 = wrong, misleading, untranslated, or extra text.
The 40 items: a fixed seeded sample stratified by category (signs weigh most, as in real use).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_testset  # noqa: E402

PLAN = {("word",): 14, ("short", "caps"): 8, ("noise", "ff", "multi", "warn"): 6, ("menu", "time", "ui", "codes"): 6,
        ("doc", "label"): 6}


def sample(items):
    rng = random.Random(20261024)
    out = []
    for cats, n in PLAN.items():
        pool = [it for it in items if it["cat"] in cats]
        out += rng.sample(pool, n)
    return out


def make(systems):
    items = load_testset()
    hyps = {s: {r["id"]: r["hyp"] for r in map(json.loads, (RESULTS / f"{s}.jsonl").read_text().splitlines())}
            for s in systems}
    rng = random.Random(7)
    key, lines = {}, ["# Blind review sheet (rate each output 0/1/2, see review.py)", ""]
    for it in sample(items):
        outs: dict[str, list[str]] = {}
        for s in systems:
            outs.setdefault((hyps[s][it["id"]] or "").strip(), []).append(s)
        texts = list(outs)
        rng.shuffle(texts)
        labels = "ABCDE"
        key[it["id"]] = {labels[k]: outs[t] for k, t in enumerate(texts)}
        lines += [f"## {it['id']} ({it['cat']}, {it['src']}→{it['tgt']}){' — ' + it['ctx'] if it.get('ctx') else ''}",
                  "```", it["text"], "```", "refs: " + " ‖ ".join(r.replace("\n", " / ") for r in it["refs"])]
        lines += [f"- **{labels[k]}**: {t.replace(chr(10), ' / ')}" for k, t in enumerate(texts)]
        lines.append("")
    (RESULTS / "review_sheet.md").write_text("\n".join(lines))
    (RESULTS / "review_key.json").write_text(json.dumps({"systems": systems, "key": key}, indent=1) + "\n")
    print(f"{len(key)} items → results/review_sheet.md")


def score():
    k = json.loads((RESULTS / "review_key.json").read_text())
    ratings = json.loads((RESULTS / "review_ratings.json").read_text())
    tot = {s: [] for s in k["systems"]}
    for iid, labels in k["key"].items():
        for label, systems in labels.items():
            for s in systems:
                tot[s].append(ratings[iid][label])
    out = {s: {"mean_0_2": round(sum(v) / len(v), 3), "correct_2": sum(x == 2 for x in v), "wrong_0": sum(x == 0 for x in v),
               "n": len(v)} for s, v in tot.items()}
    (RESULTS / "review_scores.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    make(sys.argv[2:]) if sys.argv[1] == "make" else score()
