# Translation benchmark: sign / menu / notice texts (Glass Vision OCR step 2)

Results and recommendation: [`../translation-bench.md`](../translation-bench.md).

## Files
| file | what |
|---|---|
| `build_testset.py` | the test set as readable Python (source of truth) → `testset.jsonl` |
| `testset.jsonl` | 259 items: `id, src, tgt, cat, kind, text, refs[], accept[], ctx, nb[]` (format in the script's docstring) |
| `common.py` | shared helpers; `normalize` / `is_translatable` are copies of `server/glass_vision/translate.py` |
| `run_seq2seq.py` | M2M100 418M / 1.2B (as the app runs them: fp16, 4 beams, ALL-CAPS sentence-casing, numbers/codes kept), MADLAD-400-3B (bf16), Opus-MT (one model per pair), on transformers |
| `run_llm.py` | LLM translators on llama.cpp's `llama-server` (CUDA, all layers on GPU, greedy, thinking off), prompt variants `plain` / `ctx` / `sign` |
| `run_coverage.py` | coverage probe: 3 English notices → each of the app's 43 languages, output language checked with lingua |
| `score.py` | chrF++ (sacrebleu) + COMET (`Unbabel/wmt22-comet-da`) + key-term hit + line-structure check → `results/scores.{json,md}` |
| `review.py` | blind manual review sheet for the top systems (shuffled, system names hidden) + the unblinding key |
| `results/` | raw outputs (`<system>.jsonl`: id, hyp, ms), run metadata (`.meta.json`: load time, VRAM, disk, runtime), scores, coverage, review |

## Test set (259 items)
- **Sources** (15): de 33, pl 25, fr 21, ja 21, ru 20, zh 19, es 17, uk 14, it 13, ko 13, nl 13, cs 13, en 13, ar 12, tr 12.
- **Targets**: en 180, pl 66, de 8, es 5.
- **Categories**: one-word signs 100, short signs 58, ALL CAPS 16, documents/notice prose 17, menus 16, warnings 10,
  OCR-noise variants 9, false friends / context-dependent 9, timetables / opening hours 8, multi-line signs 5,
  labels 5, UI screens 4, codes 2. One-word signs dominate on purpose: they are what the camera sees most and what
  M2M100 gets wrong (AGENTS.md).
- **ctx** = what an image caption or the app could say about the photo ("sign above a door", "sticker on a glass door",
  "restaurant menu"); **nb** = other blocks on the same photo. They are given to models only in the `ctx` / `sign`
  prompt variants.
- **OCR-noise items** imitate real recogniser errors: digit/letter swaps (`S0RTIE`, `Wyjśc1e`, `VIETAT0`), `l`/`I`
  for `I`/`l` (`PALENlA`), a wrong final letter (`AUSGANC`), Latin look-alikes inside Cyrillic (`BЫXOД` with Latin
  B and X), look-alike CJK characters (`出囗` with 囗 for 口, `禁止吸姻` with 姻 for 烟).
- **False friends / context** (`Gift` de = poison, nl = donation; it `Caldo` on a tap = hot; `Frei`/`Libre` on a
  toilet lock = vacant; `Kasse`/`Caja`/`Касса` in a shop vs a station) test whether `ctx` helps.

## How the references were made
1. Written by hand by the benchmark author (Claude, an LLM working as a multilingual translator) as the wording a
   professional would put on the target-language sign: the established sign convention, not a literal gloss
   ("Ausgang" → "Exit", "Ziehen" → "Pull", "非常口" → "Emergency exit", "Mind the gap" → "Uwaga na szczelinę").
   Prices, times, codes and proper names are kept as written in the source (e.g. "14,90 €", "￥38").
2. One-word and short signs carry **several acceptable references** (Exit / Way out, Toilet / Restroom / WC) and a
   lower-case **`accept`** list of key terms. The automatic "term hit" counts an output as adequate if any of them
   appears in it. Metrics take the best score over the references.
3. Cross-checks: sign conventions for ja/zh/ko/ar/tr were checked against standard bilingual signage (非常口 /
   安全出口 / 비상구 / مخرج طوارئ / ACİL ÇIKIŞ = Emergency exit, 押す/引く, 推/拉, 미세요/당기세요, İtiniz/Çekiniz =
   Push/Pull). After the first runs, every item where all strong systems disagreed with the reference was re-read;
   no reference was changed to match a system's wording.
4. Limits: one author, no second human translator; the ar/ko/ja references are the least certain (mostly standard
   signs, so the risk is low). chrF++ is harsh on one-word items (one different word = low score), which is why the
   term hit and the blind manual review matter more for signs than the averages.

## Reproduce
Isolated environment only (never glass-vision's `server/.venv`): `uv venv --python 3.12 venv` with torch,
transformers, sentencepiece, sacrebleu, huggingface_hub, gguf, lingua-language-detector; a second venv with
`unbabel-comet` + `setuptools<81` for COMET. GGUFs from the repos listed in `results/*.meta.json` (`gguf`), llama.cpp
`llama-server` built with CUDA (commit a25c986 here). GPU 0 only (`CUDA_VISIBLE_DEVICES=0`).

```bash
python build_testset.py
python -I run_seq2seq.py m2m418 --mode line        # also m2m12, madlad3b, opus; --mode block
python -I run_llm.py index-9b --variant ctx        # keys in run_llm.SPECS; variants plain|ctx|sign
python -I run_coverage.py index-9b gemma4-e4b m2m418
python -I score.py                                 # in the COMET venv
python -I review.py                                # blind review sheet
```
