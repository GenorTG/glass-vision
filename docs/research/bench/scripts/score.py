"""Score bench results: key-phrase recall (photos), line accuracy + fragmentation (rendered signs), latency."""
import json, re, sys, unicodedata
from pathlib import Path
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

R = Path(sys.argv[1]); S = Path(__file__).parent
keys = json.load(open(S / "keys.json")); gt = json.load(open(S.parent / "images/render_gt.json"))
def norm(s): s = re.sub(r"<\|LOC_\d+\|>", "", s); return "".join(unicodedata.normalize("NFKC", s).casefold().split())
runs = {p.stem: json.load(open(p)) for p in sorted(R.glob("*.json")) if p.stem != "summary"}
summary = {}
print("| run | photo key recall (≥90 fuzzy) | photo mean fuzzy | rendered lines exact (CER≤5%) | boxes per GT line | rendered rotations ok | median ms photos | VRAM MB |")
print("|---|---|---|---|---|---|---|---|")
for name, run in runs.items():
    hits = tot = 0; fz = []; per = {}
    for stem, k in keys.items():
        r = run.get(stem);
        if r is None: continue
        text = norm("".join(l["text"] for l in r["lines"]))
        sc = [fuzz.partial_ratio(norm(x), text) if text else 0 for x in k["keys"]]
        hits += sum(s >= 90 for s in sc); tot += len(sc); fz += sc
        per[stem] = round(sum(s >= 90 for s in sc) / len(sc), 2)
    ok = n = 0; boxes = gtl = 0; rot_ok = 0
    for stem, lines in gt.items():
        r = run.get(stem)
        if r is None: continue
        outs = [norm(l["text"]) for l in r["lines"]]
        boxes += len([o for o in outs if o]); gtl += len(lines)
        good = sum(any(Levenshtein.normalized_distance(norm(g), o) <= 0.05 for o in outs) for g in lines)
        ok += good; n += len(lines)
        if "rot" in stem or "persp" in stem: rot_ok += good == len(lines)
        per[stem] = f"{good}/{len(lines)}"
    ms = sorted(run[s]["ms"] for s in keys if s in run)
    summary[name] = per
    print(f"| {name} | {hits}/{tot} ({hits/tot:.0%}) | {sum(fz)/len(fz):.0f} | {ok}/{n} | {boxes/gtl:.2f} | {rot_ok}/4 | {ms[len(ms)//2]:.0f} | {run.get('_vram_max_alloc_mb', 0):.0f} |")
print()
stems = list(keys) + list(gt)
print("| image | " + " | ".join(runs) + " |"); print("|---|" + "---|" * len(runs))
for s in stems:
    print(f"| {s[:28]} | " + " | ".join(f"{summary[n].get(s,'')} / {runs[n].get(s,{}).get('ms','')}" for n in runs) + " |")
