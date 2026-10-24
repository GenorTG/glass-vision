# Server OCR engines for Glass Vision: survey, benchmark, recommendation

Status: research note, 2026-10-24. Nothing in the server has changed. The benchmark ran in an isolated venv, which has since been deleted. Raw results and scripts are in `~/.openclaw/workspace/.tmp/ocr-bench/results/` on genorbox1.

Scope: server-side "Read text" (`POST /text/read`), which reads a high-resolution phone still of signs, labels, screens, menus and documents in many scripts and then translates it. This note does not cover live per-frame OCR. It also does not cover the phone's ML Kit and Tesseract path.

The owner's complaints:
1. One sign comes back as many small word objects.
2. A rotated snapshot is not straightened.
3. The image is not enhanced before OCR.

---

## 1. Recommendation (ranked)

| # | Stage | Choice | Why |
|---|---|---|---|
| 1 | **Text detector** | **PP-OCRv6_medium_det** (Apache-2.0, official ONNX `PaddlePaddle/PP-OCRv6_medium_det_onnx`), run on the ONNX Runtime CUDA EP that the server venv already has | Returns **whole lines**: 0.93–1.0 boxes per ground-truth line vs 1.70 for our EasyOCR/CRAFT and 2.0–2.6 for PP-OCRv5 det. It finds 5× more lines than EasyOCR on a dense menu (375 vs 74) and handles perspective and 15° tilt. The vendor reports +4.6 Hmean over PP-OCRv5_server. |
| 2 | **Recognizer, pass 1** | **PP-OCRv6_medium_rec** (34.5M parameters; one model for zh-Hans, zh-Hant, ja, en and 46 Latin-script languages) | Best text accuracy in our test. It also removes the "try 12 EasyOCR readers per image" auto mode, which took **10–33 s** on menus and screens. |
| 3 | **Recognizer, pass 2 (script fallback)** | Script-specific **PP-OCRv5 mobile recs**: `cyrillic`, `eslav`, `arabic`, `devanagari`, `korean`, `th`, `el`, `te`, `ta` (Apache-2.0, ONNX, about 2–8M parameters each), run **per line** on the crops pass 1 read badly | v6 has no Cyrillic, Arabic or Indic. Running v6 det + cyrillic rec turned `APATCKA` into `АРБАТСКАЯ` and read the rendered Russian sign 3/3. Running small recs on a few crops costs milliseconds; EasyOCR re-reads the whole image per script. |
| 4 | **Orientation** | EXIF first (done today) → PaddleOCR's per-crop rule (a crop with h/w ≥ 1.5 is rotated 90°) → **PP-LCNet_x1_0_textline_ori** (0/180 per line), always on → whole-image **PP-LCNet_x1_0_doc_ori** only when its score is **≥ 0.85**, and used mainly to fix reading order | Without the whole-image classifier, PP-OCRv6 with the textline classifier read the sign at 90°, 180° and 270° 3/3. The ungated doc classifier rotated two real photos the wrong way (Tokyo → 180° at 0.48, Piwna → 90° at 0.74), and their recall dropped. Correct rotations scored 0.92. |
| 5 | **Lines → blocks / paragraphs** | Keep `text_layout.reading_order`, but feed it **line quads** from (1). Optionally add **PP-DocLayoutV3** (Apache-2.0, ONNX, regions + reading order, multi-point boxes) when a still has many lines (menu, document, screen) | Once the detector returns lines, the geometric grouping we already have works. The "many small objects" problem starts at CRAFT word boxes; see §2. |
| 6 | **Image enhancement** | **No global enhancement.** Read at full detector resolution, and add an optional *crop-and-upscale* second pass for tiny or low-confidence regions. Use UVDoc unwarping only in a future "document" mode | CLAHE, unsharp mask, CLAHE+sharpen and 2× upscale changed key-phrase hits by at most ±1 on 8 photos. 2× upscale cost 3–4× the time and lost a hit on bus and Tokyo. Recognition already rectifies each quad's perspective. |
| 7 | **Optional VLM second pass** | **Qwen3-VL-4B / Qwen3.5-4B** (Apache-2.0) behind vLLM in a separate, memory-capped process. Run it on demand ("Read carefully") or only on low-confidence or stylised lines. For a future document mode, use **PaddleOCR-VL-1.5/1.6** (0.9B, Apache-2.0) or **GLM-OCR** (0.9B, MIT) | Qwen3-VL-2B read what no classical engine could (the wrought-iron `DOM KSIĄŻKI`, every key of the Tokyo sign, the handwritten HK menu). It also **looped** on the dense menu (`15 ⏎ 16 ⏎ 17 …` up to 1024 tokens) and invented Arabic diacritics. Treat it as a second opinion tied to detector boxes, never as the primary engine. |
| 8 | **Translation** | Keep M2M100 (MIT) as the fallback. Evaluate **MADLAD-400-3B-MT** (Apache-2.0, 400+ languages) and an **LLM translator with sign context**: Gemma 4 E4B (Apache-2.0), Qwen3.5-4B (Apache-2.0) or TranslateGemma-4B (Gemma terms). **Do not use NLLB-200** (CC-BY-NC-4.0), Tower+ (CC-BY-NC-SA), HunyuanOCR or Hunyuan-MT (their licence excludes the EU, UK and South Korea) | One-word signs are M2M100's known weakness (AGENTS.md). An LLM that sees the whole block, and optionally the image, fixes the context ("Ausgang" on a door means Exit). See §6. |

**Integration path (no PaddlePaddle in the server venv):** PaddlePaddle publishes official `*_onnx` and `*_safetensors` repos for PP-OCRv6 det/rec (medium/small/tiny), every PP-OCRv5 script rec, both orientation classifiers, UVDoc and PP-DocLayoutV3 ([HF org](https://huggingface.co/PaddlePaddle)).
- The server venv already has `onnxruntime-gpu` 1.31.0 with a working CUDA EP (insightface uses it).
- The pre- and post-processing is about 300 lines: DB box decoding, quad crop + perspective warp, CTC decoding with the model's character dict, the textline classifier, and the h/w ≥ 1.5 rotate rule. Port it from PaddleX/RapidOCR (both Apache-2.0). RapidOCR 3.10 does not ship v6 yet, but it accepts custom model paths.
- That removes the torch-cuDNN/EasyOCR gotchas listed in AGENTS.md: `DISABLE_ADDMM_CUDA_LT`, the SM 7.5 cuDNN check, and the fp16 CRAFT reimplementation.
- Measured in the bench: v6 det + rec + textline classifier peaked at **3.4–3.7 GB** VRAM (PaddlePaddle allocator). Our EasyOCR with all script readers peaked at 3.8 GB.
- Before switching, check ONNX parity against the Paddle outputs recorded in the bench JSONs.

## 2. Why today's server output is word-level (EasyOCR/CRAFT)

`text.py` reimplements EasyOCR's `detect`:
1. CRAFT produces a character-region map and an affinity (link) map.
2. `getDetBoxes(text_threshold=0.7, link_threshold=0.4, low_text=0.4)` turns their connected components into boxes. In practice these are words, or word groups where letters touch.
3. `group_text_box` merges them into lines only if every condition holds:
   - slope < `slope_ths` 0.1
   - y-centres within `ycenter_ths` 0.5 × height
   - heights within `height_ths` 0.5
   - horizontal gap < `width_ths` 0.5 × height

   Any box tilted more than about 6° becomes a separate "free" quad and is never merged.
4. The recognizer then reads each box on its own. `paragraph=True` (`x_ths`/`y_ths`) would only merge axis-aligned boxes into paragraph strings, and it loses the per-line boxes.

So a sign photographed at an angle, or with wide letter spacing, splits into word objects. Measured: the rendered Polish paragraph tilted 15° gave **23 boxes in 5 blocks** for 6 lines, with **0/6** lines read whole. Untilted, it gave 10 boxes and 5/6 lines. Tuning `width_ths`, `slope_ths` or `link_threshold` is a band-aid. A detector trained on line-level labels (DB-style PP-OCR) fixes the cause. EasyOCR's last release was 1.7.2 (2024-09-24), and its roadmap items (swappable detector/recognizer, handwriting) are still open ([repo](https://github.com/JaidedAI/EasyOCR)).

EasyOCR has no rotation handling for whole images. `rotation_info` only tries extra rotated crops and is not used by us. The rendered sign at 90°, 180° and 270° came back as Telugu, Kannada and mirrored-Latin garbage, because auto mode picked whichever recognizer was most confident about nonsense.

## 3. Benchmark (genorbox1, RTX 3090 shared with the live service)

**Set.**
- 10 real photos:
  - `bus.jpg`
  - 9 Wikimedia Commons photos in Latin (de/pl/en), CJK (zh-Hant handwritten menu, Japanese station sign, zh-Hant exit sign), Cyrillic and Arabic. Licences are below.
- 8 rendered signs, all on a noisy wall with a light gradient:
  - German 3-line sign
  - the same sign rotated 90°, 180° and 270°, and perspective-skewed
  - Polish 6-line paragraph, straight and tilted 15°
  - Russian 3-line sign

**Metrics.**
- *Key recall*: the share of 67 hand-checked key phrases found with a fuzzy partial match of at least 90, after case-folding and removing whitespace.
- *Lines exact*: rendered ground-truth lines read by **one** box with CER ≤ 5%.
- *Boxes per GT line*: the fragmentation measure.
- Latency: the best of 2 warm runs, end-to-end inside the engine, with the image at ≤ 2560 px long side, as in `read_still`.

The GPU is shared and the default Paddle inference path was used (no TensorRT/HPI), so latencies are indicative only. Repeated runs varied up to about 2×.

| run | photo key recall | mean fuzzy | rendered lines exact | boxes / GT line | rotated/skewed signs fully read | median ms (photos) | peak VRAM |
|---|---|---|---|---|---|---|---|
| **Glass Vision today** (EasyOCR, auto script, `read_still`) | 31/67 (46%) | 72 | 10/30 | 1.70 | 0/4 | 802 (max 32 589) | 3.8 GB |
| PaddleOCR 3.7 PP-OCRv5 server det+rec | 36/67 (54%) | 74 | 5/30 | 2.33 | 0/4 | 817 | 5.9 GB |
| PP-OCRv5 server det + per-language rec (oracle language) | 40/67 (60%) | 81 | 11/30 | 2.03 | 2/4 | 713 | 7.0 GB |
| PP-OCRv6 medium, doc_ori + textline_ori (PaddleOCR default) | 43/67 (64%) | 80 | 27/30 | 0.93 | 4/4 | 550 | 3.4 GB |
| PP-OCRv6 medium, no orientation models | 48/67 (72%) | 86 | 21/30 | 0.93 | 2/4 | 516 | 3.3 GB |
| PP-OCRv6 medium, textline_ori only | 48/67 (72%) | 86 | 27/30 | 0.93 | 4/4 | 451 | 3.4 GB |
| **PP-OCRv6 det + v6 rec, cyrillic/arabic v5 rec where needed, textline_ori** | **51/67 (76%)** | **91** | **30/30** | **1.00** | **4/4** | **434** | 3.7 GB |
| RapidOCR 3.10, PP-OCRv5 server, **CPU** (ORT CUDA EP did not register in the bench venv) | 47/67 (70%) | 87 | 9/30 | 2.60 | 2/4 | 861 | — |
| Qwen3-VL-2B-Instruct (transformers, bf16, ≤1.3 MP, plain prompt) | 44/67 (66%) | 85 | 30/30 | 1.00 | 4/4 | 1931 (max 34 130) | 4.4 GB |
| PaddleOCR-VL-1.5 `Spotting:` (transformers; 6 images only) | 7/12 | 77 | 6/6 | 1.00 | 1/1 tested | 7317 | 2.0 GB |

Per-image observations:
- **Rotation.** PP-OCRv6 with the textline classifier read 90°, 180° and 270° correctly without the whole-image classifier. At 180° and 270° the line order came out reversed, because reading order was computed in the rotated frame; the gated doc classifier or a direction vote fixes the order. Without the textline classifier, 180° and 270° failed.
- **Doc orientation classifier on scene photos.** Correct rotations of the rendered sign scored 0.921–0.926. The two misfires scored 0.479 (Tokyo → 180°) and 0.737 (Piwna → 90°). Correct "0°" answers on cluttered photos scored 0.36–0.76. A gate at ≥ 0.85 separates them on this set.
- **Hard cases.** On `DOM KSIĄŻKI` (wrought iron, viewed from below), every classical engine scored 0; only Qwen3-VL-2B read it. On the Arabic street sign, the best classical result was v6 det + arabic rec at 2/5 keys, reading the postcode and `الرقم البريدي` partly. The large calligraphic line was missed by every engine; the VLM transliterated it with invented diacritics.
- **Dense content.**
  - Del Taco menu: v6 found 375 lines vs EasyOCR's 74.
  - HK handwritten menu: v6 read 9/14 keys in 0.8 s; EasyOCR read 5/14 in 10.1 s; Qwen3-VL-2B read 11/14 in 18 s.
  - Laptop code screenshot: v6 13/14 keys in 0.55 s vs EasyOCR 6/14 in 10.2 s.
  - EasyOCR's multi-second times come from auto mode trying every installed script, with Chinese decoding on the CPU.
- **PP-OCRv5 server det** splits lines into words almost as much as CRAFT (2.0–2.3 boxes per line). The line-level behaviour is new in PP-OCRv6.
- **Enhancement experiment** (v6 + textline_ori; key hits for orig / CLAHE / sharpen / CLAHE+sharpen / 2× upscale):
  - bus: 1/1/1/1/0 of 1
  - Tokyo: 7/7/7/7/6 of 8
  - Del Taco: 14/14/13/14/13 of 19
  - HK menu: 9/10/9/9/10 of 14
  - laptop: 13/14/14/12/14 of 14
  - Emergency exit, Taiwan exit: unchanged
  - Piwna: 0 in every variant

  No variant is consistently better, and 2× upscale costs 3–4× the time.
- **VLM speed.** Plain transformers generation (Qwen3-VL-2B at about 30 tok/s; PaddleOCR-VL in transformers at about 5 tok/s with 1.6 MP spotting input) is far too slow for dense pages. Production use needs vLLM or SGLang. PaddleOCR-VL-1.6 on the **native Paddle** backend ran CPU-bound at 0% GPU for more than 4 min on one image and was aborted; use the vLLM/FastDeploy server or the llama.cpp backend that PaddleOCR 3.4.1 added.

Test images (Wikimedia Commons; used for local evaluation only, not redistributed):

| file | licence | author |
|---|---|---|
| [Emergency exit sign 01](https://commons.wikimedia.org/wiki/File:Emergency_exit_sign_01.jpg) | CC BY-SA 4.0 | Kritzolina |
| [Taiwan exit sign](https://commons.wikimedia.org/wiki/File:Taiwan_exit_sign_white_backgrounded.jpg) | Public domain | Oakw |
| [HK paper menus](https://commons.wikimedia.org/wiki/File:Paper_menus_and_white_board_menu_in_the_hk_style_restaurants.jpg) | CC BY-SA 4.0 | Peachyeung316 |
| [Tokyo Station sign (Yamanote Line)](https://commons.wikimedia.org/wiki/File:Tokyo_Station_Sign_(Yamanote_Line).jpg) | CC BY-SA 3.0 | そらみみ |
| [Moscow Arbatskaya sign](https://commons.wikimedia.org/wiki/File:Moscow,_Arbatskaya_(APL)_metro_station_-_sign_(43315108441).jpg) | CC BY 2.0 | Andrey Filippov |
| [Sidi El Wasti St. sign](https://commons.wikimedia.org/wiki/File:Arabic-English_street_sign_%22Sidi_El_Wasti_St.%22_in_Alexandria.jpg) | CC BY 4.0 | Marsupium |
| [Del Taco drive-thru menu](https://commons.wikimedia.org/wiki/File:Drive_thru_menu_board_at_Del_Taco_in_Albany,_Oregon.jpg) | CC BY 4.0 | Rick Obst |
| [Macro laptop coding](https://commons.wikimedia.org/wiki/File:Macro_laptop_coding_(Unsplash).jpg) | CC0 | Marc Mueller |
| [Piwna Street, Warsaw 05](https://commons.wikimedia.org/wiki/File:Piwna_Street,_Warsaw_05.jpg) | CC BY-SA 4.0 | Simon Burchell |

## 4. Engine comparison

"Scene" means signs and photos; "docs" means pages and scans. Licences were checked on the Hugging Face API or the GitHub repo on 2026-10-24. The repo is AGPL-3.0, and Apache-2.0 and MIT weights and code are compatible with it.

| engine (latest) | kind | scene text | documents | scripts | output | orientation | 24 GB GPU cost | licence (code / weights) | maturity | verdict for GV |
|---|---|---|---|---|---|---|---|---|---|---|
| **PaddleOCR 3.7 · PP-OCRv6** (2026-06) | det + CTC rec | **best measured here** | good (PP-StructureV3 on top) | v6: zh/zh-Hant/ja/en + 46 Latin; v5 recs: Cyrillic, East Slavic, Arabic, Devanagari, Korean, Thai, Greek, Telugu, Tamil (109 languages in total). **No Bengali**; Kannada only via PP-OCRv3 | **line quads** + text + score (+ word boxes) | doc_ori (4-way), textline_ori (0/180), UVDoc unwarp | 3–4 GB; 0.2–0.5 s per sign, about 2 s for a dense menu (Paddle, no TRT) | Apache-2.0 / Apache-2.0 | very active (monthly releases) | **adopt (det + rec + orientation)** |
| PaddleOCR · PP-StructureV3 / PP-DocLayoutV3 | layout + OCR pipeline | n/a (layout for docs) | strong | as above | regions with 25 classes, reading order, multi-point boxes | via doc_ori | +1–2 GB | Apache-2.0 | active | optional for menu/doc mode |
| PaddleOCR-VL 1.5 / 1.6 (0.9B) | doc VLM (NaViT + ERNIE-0.3B) | spotting prompt; weak on stylised scene text in our test | **SOTA**: OmniDocBench v1.5 94.5 (1.5), v1.6 96.33 (1.6) | 109 languages | markdown / blocks / spotting quads | Real5 robustness set (skew, warp, screen photo) | 2 GB in bf16; needs vLLM, FastDeploy or llama.cpp to be fast | Apache-2.0 | active | document mode, later |
| RapidOCR 3.10 | ONNX/OpenVINO wrapper of PP-OCR models | same as the PP-OCR model used | same | PP-OCRv5 set (no v6 models yet) | lines (word boxes optional) | cls 0/180 | ORT/OpenVINO; CPU ~0.3–0.9 s per sign here | Apache-2.0 | active | reference code for the ONNX port |
| EasyOCR 1.7.2 (today) | CRAFT + CRNN | weak (word splits, no rotation) | weak | 80+ languages, one reader per script | word/line boxes | none | 3.8 GB; seconds in auto mode | Apache-2.0 | dormant since 2024-09 | **replace** |
| docTR 1.1 (Mindee) | DBNet/FAST + CRNN/PARSeq/VIPTR | ok (Latin) | good for Latin docs | mostly Latin vocabularies | Page/Block/Line/Word tree | rotated boxes, `straighten_pages` | small | Apache-2.0 | active, small team | Latin docs only; no gain over PP-OCRv6 |
| Surya OCR 2 (0.65B) / Marker | line det + VLM OCR / layout | ok | strong (olmOCR-bench 83.3) | 90+ languages (Arabic 72.7%) | lines, blocks, layout, reading order | none documented | VLM-scale | Apache-2.0 / **modified OpenRAIL-M: free for research, personal use and startups under $5M revenue** | active | licence and size not worth it |
| Chandra OCR 2 (Datalab, ~4–9B) | doc VLM | — | olmOCR-bench 85.8 | 40+ languages | markdown/html/json with layout | — | 9B-class | Apache / **OpenRAIL, same revenue cap** | active | no (licence, size) |
| MMOCR (OpenMMLab) | toolbox | — | — | — | — | — | — | Apache-2.0 | effectively unmaintained since 2023 | no |
| Qwen3-VL 2B/4B/8B (2025-10) / Qwen3.5 0.8–9B (2026-03) | general VLM | **strong on hard scene text** (blur, tilt, low light; 32 OCR languages) | good; Qwen3-VL-235B scores 74.9 rec / 38.3 det on the PP-OCRv6 benchmark | 32+ | free text; can emit boxes | robust | 2B: 4.4 GB bf16; 4B about 9–10 GB bf16, about 5 GB in FP8/AWQ; vLLM needed | Apache-2.0 | very active | **optional second pass** |
| GOT-OCR2.0 (0.7B) | OCR VLM | moderate | ok (olmOCR-bench 48.3) | mainly zh/en | text / formatted | — | small | Apache-2.0 | stale since 2024 | no |
| olmOCR-2 (7B, Qwen2.5-VL) | doc VLM | not targeted | olmOCR-bench 82.4 | mostly Latin | markdown | rotation handling in the pipeline | 7B (about 16 GB in bf16, FP8 available) | Apache-2.0 | active (AI2) | no (documents only, big) |
| dots.ocr → dots.mocr (3B, 2026-03) | doc VLM | has a scene-spotting prompt | olmOCR-bench 83.9 | "any script" | layout JSON + text | — | 3B, vLLM | code MIT; v1 weights MIT; mocr/1.5 weights under the "dots.ocr License Agreement" (MIT + attribution and data-governance terms) | active | candidate for document mode; check the licence |
| DeepSeek-OCR (3B, MIT) / DeepSeek-OCR 2 (2026-01, Apache-2.0) | doc VLM with optical compression | not targeted | OmniDocBench v1.5 87.4 → 91.1 | ~100 | markdown, grounding | — | 3.3B, vLLM | MIT / Apache-2.0 | active | no (documents, big) |
| MinerU 2.5 / MinerU 4.0 pipeline | doc parsing (1.2B VLM) | not targeted | strong | ~109 (uses PP-OCR) | markdown/JSON | — | ≥ 8 GB | MinerU2.5 weights **AGPL-3.0**; MinerU repo now under the "MinerU Open Source License" (Apache-2.0 + conditions) | active | not needed |
| Nanonets-OCR2-3B | doc VLM (Qwen2.5-VL-3B fine-tune) | — | good | multi | markdown | — | 3B | **no licence stated on the HF card**; the base model is Qwen2.5-VL-3B (Qwen Research licence) | active | no (licence unclear) |
| Florence-2 (0.23/0.77B) | small VLM, `<OCR_WITH_REGION>` | fair on Latin, weak elsewhere | fair | mainly English | words/lines + quads | — | under 2 GB | MIT | stale since 2024 | no |
| InternVL3.5 (1–241B) | general VLM | good | good | multi | free text | — | 8B ≈ 17 GB in bf16 | Apache-2.0 | active | Qwen covers this role |
| GLM-OCR (0.9B, 2026-02) | doc VLM | not targeted (weak on vertical CJK per a review) | OmniDocBench v1.5 94.6 | multi | markdown | — | ~2 GB | code Apache-2.0 / weights MIT | active | document mode candidate |
| LightOnOCR-2-1B (2026-01) | doc VLM | — | olmOCR-bench ~83 | Latin-strong; degraded on CJK/Arabic | markdown (+bbox variant) | — | 1B | Apache-2.0 | active | Latin documents only |
| HunyuanOCR (1B) / HunyuanOCR-1.5 | OCR VLM with spotting + photo translation | strong (vendor) | strong | multi | spotting quads, translation | — | 1B | **Tencent Hunyuan Community License: "does not apply in the European Union, UK and South Korea"** | active | **excluded (owner in the EU)** |
| Mistral OCR 3 (`mistral-ocr-2512`) | API | — | strong | multi | markdown + annotations | — | n/a; $2 per 1000 pages ($1 batch) | proprietary | — | reference only (privacy, cost) |

## 5. Per-engine notes and sources

- **PaddleOCR 3.x** ([repo](https://github.com/PaddlePaddle/PaddleOCR), [releases](https://github.com/PaddlePaddle/PaddleOCR/releases), [PP-OCRv6 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html), [3.0 report](https://arxiv.org/abs/2507.05595))
  - Releases:
    - 3.1 (2025-06): 37-language PP-OCRv5 rec
    - 3.2 (2025-08): English, Thai and Greek recs
    - 3.3 (2025-10): PaddleOCR-VL and 109-language PP-OCRv5 recs (~2M parameters)
    - 3.4 (2026-01): PaddleOCR-VL-1.5, irregular boxes, spotting, seal recognition
    - 3.4.1: llama.cpp backend
    - 3.5 (2026-04): transformers backend for 20 models
    - 3.6 (2026-05): PaddleOCR-VL-1.6
    - 3.7 (2026-06-11): PP-OCRv6 tiny/small/medium (1.5M / 7.7M / 34.5M parameters)
  - PP-OCRv6 vendor benchmark (16 categories):
    - medium: det Hmean 86.2, rec 83.2
    - v5_server: 81.6 / 78.1
    - Qwen3-VL-235B: 38.3 / 74.9
    - GPT-5.5: 45.6 / 64.2

    A100 latency: 0.29 s/image for medium. These are vendor numbers, but our independent mini-benchmark agrees on the direction.
  - `PaddleOCR()` defaults in 3.7: `lang=None` gives PP-OCRv6_medium det+rec; `lang="ru"|"ar"|…` gives PP-OCRv5_server_det + a script mobile rec (`paddleocr/_pipelines/ocr.py` `_get_ocr_model_names`). The best combination we measured (v6 det + v5 script rec) has to be configured explicitly with `text_detection_model_name` and `text_recognition_model_name`.
  - Pipeline steps: doc_ori → UVDoc → det → crop (h/w ≥ 1.5 rotated 90°) → textline_ori → rec. `return_word_box=True` adds word boxes inside lines, which helps tap-to-select.
  - Official `*_onnx` and `*_safetensors` model repos exist for v6, the v5 script recs, the orientation classifiers, UVDoc and PP-DocLayoutV2/V3 ([HF org](https://huggingface.co/PaddlePaddle)).
  - PaddleOCR-VL: [1.5 paper](https://arxiv.org/abs/2601.21957) and [card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) report OmniDocBench v1.5 94.5 and Real5-OmniDocBench 92.05; prompts `OCR:`, `Spotting:`, `Table Recognition:`, … 1.6 reports OmniDocBench v1.6 96.33.
  - Install note: `paddleocr`/`paddlex` pin `opencv-contrib-python` 4.10, which clashes with an `opencv-python(-headless)` cv2, the same class of problem as the mediapipe gotcha. The GPU wheel comes from Paddle's own index (`paddlepaddle-gpu==3.2.1`, cu126). The isolated venv was 6.8 GB. Another reason to use ONNX in the server.
- **RapidOCR** ([repo](https://github.com/RapidAI/RapidOCR)): Apache-2.0. ONNX Runtime, OpenVINO, Paddle or torch backends. Ships PP-OCRv4/v5 models (no v6 in 3.10.0) and accepts custom model paths. With onnxruntime-gpu 1.31 in a fresh venv, the CUDA EP did not register ("No registered plugin EP device"), so our numbers are CPU. The server venv's onnxruntime 1.31.0 does list `CUDAExecutionProvider`; confirm that it actually runs a PP-OCR graph on the GPU before relying on it.
- **EasyOCR** ([repo](https://github.com/JaidedAI/EasyOCR)): 1.7.2 (2024-09). See §2.
- **docTR** ([repo](https://github.com/mindee/doctr)): Apache-2.0, v1.1.
  - Detection: DBNet / LinkNet / FAST. Recognition: CRNN / SAR / MASTER / ViTSTR / PARSeq / VIPTR. Layout: LW-DETR.
  - Output tree: Page → Block → Line → Word. Rotated boxes and page straightening are supported.
  - Vocabularies are Latin-centric, so it doesn't help our multi-script need.
- **Surya OCR 2 / Marker / Chandra** ([surya](https://github.com/datalab-to/surya)): Surya OCR 2 is a 650M-parameter VLM, 90+ languages, 87.2% pass rate on its own 91-language set, olmOCR-bench 83.3. Code is Apache-2.0. Weights use a "modified AI Pubs Open Rail-M license … free for research, personal use, and startups under $5M funding/revenue". Chandra's weights are under OpenRAIL as well.
- **MMOCR** ([repo](https://github.com/open-mmlab/mmocr)): no meaningful releases since 2023. Skip.
- **Qwen3-VL / Qwen3.5** ([Qwen3-VL-8B card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models)):
  - Qwen3-VL: Apache-2.0. The card claims OCR in "32 languages (up from 19)" that is "robust in low light, blur, and tilt".
  - Qwen3.5 0.8B/2B/4B/9B (2026-03) are natively multimodal and Apache-2.0. Qwen3.5-4B is a newer drop-in for the 4B slot; not benchmarked here.
  - Our run showed that a plain "transcribe" prompt makes the 2B model loop on dense, repetitive menus. Use `repetition_penalty`, cap `max_new_tokens` per crop, and prefer crop-level prompts.
- **GOT-OCR2.0** ([card](https://huggingface.co/stepfun-ai/GOT-OCR2_0)): Apache-2.0, 0.7B, in transformers. Superseded (olmOCR-bench 48.3 per Surya's table).
- **olmOCR-2** ([paper](https://arxiv.org/abs/2510.19817), [card](https://huggingface.co/allenai/olmOCR-2-7B-1025)): Apache-2.0. Qwen2.5-VL-7B fine-tuned with GRPO unit-test rewards. olmOCR-bench 82.4. PDF-oriented.
- **dots.ocr / dots.mocr** ([repo](https://github.com/rednote-hilab/dots.ocr)): code MIT. dots.ocr-1.5 was renamed dots.mocr on 2026-03-19; it has 3B parameters, olmOCR-bench 83.9, and a scene-spotting prompt. Weights for v1.5 and later are under the "dots.ocr LICENSE AGREEMENT" (MIT-based with supplementary terms). Recommended serving is vLLM 0.11 or later.
- **DeepSeek-OCR / OCR 2** ([v1 card](https://huggingface.co/deepseek-ai/DeepSeek-OCR), [v2 card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)): v1 is MIT (3.3B, 2025-10). v2 is Apache-2.0 (2026-01-27) and scores OmniDocBench v1.5 91.09. Built for document compression.
- **MinerU** ([repo](https://github.com/opendatalab/MinerU)): the 4.0 pipeline's repo licence is now the "MinerU Open Source License" (Apache-2.0 + conditions). The [MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) weights are AGPL-3.0. It is a document parser.
- **Nanonets-OCR2-3B** ([card](https://huggingface.co/nanonets/Nanonets-OCR2-3B)): no licence on the HF card. A fine-tune of Qwen2.5-VL-3B, whose base licence is research-only. Avoid.
- **Florence-2** ([card](https://huggingface.co/microsoft/Florence-2-large)): MIT, 0.77B, with `<OCR_WITH_REGION>`. English-centric and stale.
- **InternVL3.5** ([card](https://huggingface.co/OpenGVLab/InternVL3_5-8B)): Apache-2.0. A capable general VLM, but Qwen3-VL/3.5 have better OCR claims at the same size.
- **GLM-OCR** ([paper](https://arxiv.org/abs/2603.10910), [card](https://huggingface.co/zai-org/GLM-OCR)): weights MIT, code Apache-2.0. 0.9B parameters (CogViT 0.4B + GLM 0.5B), OmniDocBench v1.5 94.6.
- **LightOnOCR-2-1B** ([paper](https://arxiv.org/abs/2601.14251), [card](https://huggingface.co/lightonai/LightOnOCR-2-1B)): Apache-2.0. Strong on Latin documents.
- **HunyuanOCR** ([report](https://arxiv.org/abs/2511.19575), [card](https://huggingface.co/tencent/HunyuanOCR)): 1B. Its spotting benchmark covers street view, screens, ads and video frames, and it does photo translation end to end. The LICENSE text says: "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA". **Unusable here.**
- **Mistral OCR 3** ([announcement](https://mistral.ai/news/mistral-ocr-3)): API `mistral-ocr-2512`, $2 per 1000 pages ($1 with batch). Self-hosting is enterprise-only.
- **Benchmarks:**
  - [OCRBench v2](https://arxiv.org/abs/2501.00321) ([board](https://99franklin.github.io/ocrbench_v2/)): general LMMs; not comparable to pipeline OCR.
  - [OmniDocBench](https://github.com/opendatalab/OmniDocBench): documents.
  - olmOCR-bench: PDFs.
  - None of them measures "phone photo of a sign in script X". The PP-OCRv6 benchmark and HunyuanOCR's spotting set come closest. Our own small set, with rendered rotations and Commons photos, is the most relevant evidence and should become a regression test (below).

## 6. Translation (alternatives to M2M100)

| model | size | languages | licence (HF API, 2026-10-24) | fit |
|---|---|---|---|---|
| M2M100 418M / 1.2B (today) | 0.4 / 1.2B | 100 | MIT | keep as fallback |
| NLLB-200 distilled 600M / 1.3B | 0.6 / 1.3B | 200 | **CC-BY-NC-4.0** | **no**: non-commercial, conflicts with AGPL users' freedoms |
| MADLAD-400-MT 3B / 7B / 10B | 3B+ | 419 | Apache-2.0 | best permissive seq2seq coverage; 3B ≈ 6 GB in bf16, about 3 GB in int8 |
| Opus-MT (Helsinki-NLP) | ~75M per pair | per pair | Apache-2.0 / CC-BY-4.0 (varies per repo) | fast; needs a model per pair; weak on short context |
| TranslateGemma 4B / 12B / 27B (2026-01) | 4–27B | 55 | Gemma Terms of Use (gated, use restrictions) | strong; Google reports the 12B beating the Gemma 3 27B base on MetricX |
| Gemma 4 E2B / E4B / 26B-A4B / 31B (2026-04) | 2–31B | 140+ | **Apache-2.0** | general LLM translator with image input (E4B is about 8B parameters including embeddings) |
| Qwen3.5 2B / 4B / 9B (2026-03) | 2–9B | many (count not verified here) | **Apache-2.0** | same role; also covers the VLM second pass (one model for both) |
| EuroLLM-9B-Instruct | 9B | EU languages + more | Apache-2.0 | EU-centric |
| Seed-X-PPO-7B (ByteDance) | 7.5B | 28 | OpenMDW (permissive) | strong MT, narrower coverage |
| Hunyuan-MT-7B | 8B | 33 | Tencent licence, **EU/UK/KR excluded** | **no** |
| Tower-Plus-9B (Unbabel) | 9B | ~24 | **CC-BY-NC-SA-4.0** | **no** |

Recommendation:
- Keep M2M100 as the fast, permissive default.
- Add an "LLM translate" option that sends the **whole block** (all lines of one sign) plus a hint ("short public sign / menu item / UI label"), and optionally the crop image, to Qwen3.5-4B or Gemma 4 E4B served by vLLM.
- Run the same model as the OCR second pass so only one extra model is resident (about 5 GB in FP8/AWQ). That fits next to OCR (about 4 GB) and the live pipelines on the 3090.
- Benchmark the candidates on our one-word sign cases ("Ausgang", "ВХОД ВОСПРЕЩЁН", …) before switching.

## 7. Proposed `read_still` v2 pipeline and next steps

1. Decode JPEG with EXIF. Downscale to ≤ 2560 px for the detector, but crop recognition patches from the **full-resolution** original. This is cheap and recovers small text.
2. PP-OCRv6_medium_det → line quads. PP-LCNet_x1_0_doc_ori in parallel: if its score is ≥ 0.85 and the label is not 0, rotate and re-run detection. Otherwise vote a dominant direction from the quads.
3. Crop each quad with a perspective warp (rotate 90° if h/w ≥ 1.5) → textline_ori (0/180) → PP-OCRv6_medium_rec, batched.
4. Lines with low confidence, or whose letters are mostly outside v6's character set, go through the installed PP-OCRv5 script recs. Keep the best per line, reusing the `_auto_score` native-script idea. The response's `script` becomes per line, which is a protocol addition; make it optional so old clients keep working (AGENTS.md: never break installed apps).
5. Run `reading_order` on quads, in the de-rotated frame. Optionally use PP-DocLayoutV3 regions as block seeds when there are more than ~40 lines.
6. Optional: VLM second pass on the remaining low-confidence crops, or on request.
7. Translation per block, as in §6.

Before replacing EasyOCR:
- Turn this bench into `tests/test_text_read_quality.py`: rendered rotations, tilt and Cyrillic, plus a few Commons photos stored with attribution, or regenerate them as synthetic signs. Assert line counts and key recall.
- Check ONNX vs Paddle parity.
- Measure ORT CUDA EP latency (with and without the TensorRT EP) in the real server process alongside YOLO and InsightFace.
- Decide what to do with Bengali (no PaddleOCR rec): keep EasyOCR's `bengali.pth` as a legacy fallback, or drop it.
- Live "Mark text areas" can move to PP-OCRv6_small_det or tiny_det (7.7M / 1.5M parameters) for lower per-frame cost; it was not measured here.
