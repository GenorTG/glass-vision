"""Scores every results/<system>.jsonl against testset.jsonl.

usage: python -I score.py [--no-comet]   (venv with unbabel-comet + sacrebleu; COMET checkpoint under WORK/models)
Writes results/scores.json and results/scores.md.

Metrics (per item, then averaged; a missing hypothesis = pair not covered by that system, left out and counted):
- chrF++ (sacrebleu, word_order=2), sentence level, best over the item's references.
- COMET = Unbabel/wmt22-comet-da (Apache-2.0), best over references.
- term = key-term hit rate on items with `accept` (signs: did "Ausgang" become "Exit"?).
- lines = list blocks (menus, timetables, UI) whose output has as many lines as the input.
- latency p50/p95 over items that went to the model (warm, one block per call).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, WORK, load_testset, percentile  # noqa: E402

GROUPS = {
    "word": {"word"},
    "sign": {"short", "caps", "noise", "ff", "multi", "warn"},
    "list": {"menu", "time", "ui", "codes"},
    "prose": {"doc", "label"},
}


def flat(text: str) -> str:
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-comet", action="store_true")
    a = ap.parse_args()
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    items = {it["id"]: it for it in load_testset()}
    systems = sorted(p.stem for p in RESULTS.glob("*.jsonl"))
    hyps = {s: {r["id"]: r for r in map(json.loads, (RESULTS / f"{s}.jsonl").read_text().splitlines())} for s in systems}

    comet_scores: dict[tuple[str, str], float] = {}
    if not a.no_comet:
        from comet import load_from_checkpoint

        model = load_from_checkpoint(str(WORK / "models/wmt22-comet-da/checkpoints/model.ckpt"))
        triples, keys = [], []
        for s in systems:
            for iid, r in hyps[s].items():
                if r["hyp"] is None:
                    continue
                for k, ref in enumerate(items[iid]["refs"]):
                    triples.append({"src": flat(items[iid]["text"]), "mt": flat(r["hyp"]), "ref": flat(ref)})
                    keys.append((s, iid))
        out = model.predict(triples, batch_size=128, gpus=1, progress_bar=False)
        for key, score in zip(keys, out.scores):
            comet_scores[key] = max(comet_scores.get(key, -1.0), float(score))

    table = {}
    for s in systems:
        meta_path = RESULTS / f"{s}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        per = []
        for iid, it in items.items():
            r = hyps[s].get(iid)
            if r is None or r["hyp"] is None:
                continue
            hyp = flat(r["hyp"])
            c = max(chrf.sentence_score(hyp, [flat(ref)]).score for ref in it["refs"])
            row = {"id": iid, "cat": it["cat"], "tgt": it["tgt"], "chrf": c, "comet": comet_scores.get((s, iid)),
                   "ms": r.get("ms")}
            if "accept" in it:
                row["term"] = any(t in r["hyp"].lower() for t in it["accept"])
            if it["kind"] == "list":
                row["lines"] = len(r["hyp"].strip().split("\n")) == len(it["text"].split("\n"))
            per.append(row)

        def agg(rows):
            def mean(key):
                xs = [x[key] for x in rows if x.get(key) is not None]
                return round(sum(xs) / len(xs), 4) if xs else None

            return {"n": len(rows), "chrf": mean("chrf"), "comet": mean("comet"), "term": mean("term"),
                    "lines": mean("lines")}

        ms = [x["ms"] for x in per if x.get("ms")]
        entry = {"meta": meta, "covered": len(per), "all": agg(per),
                 "groups": {g: agg([x for x in per if x["cat"] in cats]) for g, cats in GROUPS.items()},
                 "targets": {t: agg([x for x in per if x["tgt"] == t]) for t in ("en", "pl", "de", "es")},
                 "p50_ms": round(percentile(ms, 0.5), 1) if ms else None,
                 "p95_ms": round(percentile(ms, 0.95), 1) if ms else None,
                 "per_item": per}
        table[s] = entry
    (RESULTS / "scores.json").write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")

    def f(x, k=1, pct=False):
        if x is None:
            return "–"
        return f"{100 * x:.{k}f}" if pct else f"{x:.{k}f}"

    lines = ["| system | covered | COMET | chrF++ | term hit | word COMET | sign COMET | list COMET | prose COMET "
             "| →pl COMET | list lines kept | p50 ms | p95 ms | load s | VRAM MiB | disk GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(table, key=lambda s: -(table[s]["all"]["comet"] or table[s]["all"]["chrf"] / 100))
    for s in order:
        e, m = table[s], table[s]["meta"]
        g = e["groups"]
        lines.append(
            f"| {s} | {e['covered']}/{len(items)} | {f(e['all']['comet'], 3)} | {f(e['all']['chrf'])} | "
            f"{f(e['all']['term'], 0, True)}% | {f(g['word']['comet'], 3)} | {f(g['sign']['comet'], 3)} | "
            f"{f(g['list']['comet'], 3)} | {f(g['prose']['comet'], 3)} | {f(e['targets']['pl']['comet'], 3)} | "
            f"{f(g['list']['lines'], 0, True)}% | {f(e['p50_ms'], 0)} | {f(e['p95_ms'], 0)} | {m.get('load_s', '–')} | "
            f"{m.get('vram_mib_process', '–')} | {m.get('disk_gb', '–')} |")
    (RESULTS / "scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
