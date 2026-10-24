# OCR research — production-grade "Read text" (step 1, 2026-10-24)

Synthesis of four reports in `docs/research/` (each claim there has its source URL):
- `ocr-diagnosis.md`: why our current output is fragmented, unrotated and unprocessed (file:line).
- `ocr-commercial.md`: how Google Lens/Translate, Apple Live Text, Microsoft Lens/Azure Read, Adobe, CamScanner,
  Dropbox, ABBYY, DeepL, Papago and Yandex do it (papers, docs, patents).
- `ocr-server-engines.md`: self-hostable engines (PaddleOCR 3.x/PP-OCRv6, RapidOCR, docTR, Surya, MMOCR, EasyOCR, VLM
  OCR: Qwen3-VL, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, olmOCR, GLM-OCR…) + translation models, **with a benchmark on
  this server** (`docs/research/bench/`: scripts, ground truth, raw JSON, image sources/licences).
- `ocr-phone-pipeline.md`: CameraX capture, orientation, preprocessing, on-device engines, grouping, result UX, and
  20 concrete defects in our Kotlin code.

## 1. What the best products do (the patterns that repeat)
1. **Lines are the unit, blocks are what the user sees.** Google, Azure, ABBYY, Yandex, ML Kit and Apple's newest
   API all return lines inside blocks/paragraphs. Google's CVPR 2022 paper: line-level detection gives far better
   paragraph grouping than word-level (53.6 vs 39.1 PQ on HierText).
2. **Translate whole blocks**, de-hyphenated, then reflow into the block (Lens, Yandex). Per-line translation breaks
   sentences and is exactly M2M100's one-word-sign failure.
3. **Orientation is an explicit, reported step** (ABBYY 0/90/180/270, Azure angle incl. fine skew, Yandex per page
   and line, Dropbox a dedicated net). The main risk is rotating already-upright images, so it is confidence-gated.
4. **Capture quality is gated before OCR**: focus on the text, motion/blur checks, brightness → torch hint, best frame
   or auto-retake (ABBYY quality score, Apple smudge detection, Amazon sharpness frame selection, Scanbot states).
5. **Read at native resolution where the text is**: detect on a downscale, recognize line crops from the full-res
   image (ML Kit wants 16–24 px per character, Azure ≥12 px text height); tap-a-region to read.
6. **No binarization / "magic" filters ahead of neural recognizers.** Those filters are for how scanned pages look.
   Our bench confirms it: CLAHE, sharpening and 2× upscale moved results by ≤1 key phrase and upscale cost 3–4×.
7. **Documents are a separate mode** (Microsoft's scene-vs-document split). On Android the free, on-device ML Kit
   Document Scanner gives edge detection, perspective, auto-rotate, shadow/stain removal and auto-capture.
8. **Live view is stabilized** (ABBYY Tentative → Stable, Apple stable item IDs, Translate 2019's anti-flicker work).
9. **Best translation display = in place**: erase the original text and render the translation in the block
   (Word Lens, Lens, Papago, Yandex, DeepL); the 2019 Lens did it with simple colour sampling.

## 2. Benchmark on this server (RTX 3090, 10 real photos + 8 rendered signs incl. 90/180/270°, skew, tilt, Cyrillic)
| stack | photo key-phrase recall | rendered lines exact | boxes per real line | rotated/skewed signs | median latency |
|---|---|---|---|---|---|
| **EasyOCR (today)** | 46% | 10/30 | 1.70 (word fragments) | 0/4 | 10–33 s on menus/screens (12-script auto) |
| **PP-OCRv6 det+rec + PP-OCRv5 Cyrillic/Arabic rec + textline orientation** | **76%** | **30/30** | **1.00** | **4/4** | **~0.43 s**, 3.7 GB VRAM |
| Qwen3-VL-2B (VLM) | best on stylised/handwritten signs | — | — | — | slow (~30 tok/s unbatched); looped on a dense menu, invented diacritics |

Whole-image orientation classifier rotated 2 real photos wrongly (scores 0.48/0.74 vs 0.92 for correct ones) → use it
only at ≥0.85; per-line 0/180 classifier + "tall crop → 90°" rule read every rotated sign.

## 3. Recommended architecture

### Server (`/text/read`)
1. Decode with EXIF. **PP-OCRv6_medium_det** (official ONNX, Apache-2.0) on ≤2560 px → line quads.
2. Orientation: per-crop h/w ≥ 1.5 → 90°, **PP-LCNet textline_ori** (0/180) always; **doc_ori** only at score ≥ 0.85.
   Report the applied `angle`.
3. Perspective-warped crops **from the full-res still** → **PP-OCRv6_medium_rec** (zh/ja/en + 46 Latin languages),
   batched; low-confidence or out-of-charset lines → small **PP-OCRv5 script recs** (Cyrillic, East-Slavic, Arabic,
   Devanagari, Korean, Thai, Greek, Telugu, Tamil). Bengali: keep EasyOCR's model as legacy fallback or drop.
4. Group the *lines* with our existing `text_layout` (now fed real lines, in the de-rotated frame); optional
   **PP-DocLayoutV3** regions when a still has many lines (menus, documents, screens).
5. Translate **per block** (joined, de-hyphenated); per line only for list-like blocks (menus, timetables).
6. Optional second opinion: **Qwen3-VL-4B / Qwen3.5-4B** (Apache-2.0) in a memory-capped vLLM process, only for
   low-confidence/stylised lines or a "Read carefully" button, always tied to detector boxes.
All of it runs on the server's existing `onnxruntime-gpu`, with no PaddlePaddle. It removes EasyOCR and its torch
gotchas (cuBLASLt flag, SM 7.5 check, fp16 CRAFT reimplementation) and the 10–33 s multi-script auto mode.

### Phone
1. **Capture**: `OrientationEventListener` → `ImageCapture.targetRotation` (the rotation bug); `MAXIMIZE_QUALITY` +
   `startFocusAndMetering` on the live text marks (CameraX 1.3.4 never triggers AF before a still in
   MINIMIZE_LATENCY); gyro "hold still" gate; freeze the preview, then capture; Laplacian blur check → one auto-retake;
   torch / zoom controls.
2. **Server mode**: upload the camera JPEG as is + EXIF orientation (no decode/rotate/re-encode of 12 MP on the phone).
3. **Phone mode**: decode at ½ + region-decode full-res crops for small text; ML Kit v2 keeps quads, angle and
   language; orientation check from the median line angle; **PP-OCRv5 mobile via ONNX Runtime** (downloadable packs)
   for Cyrillic/Arabic/Greek/Thai/Tamil/Telugu, including its detector for the scripts ML Kit can't even box; Tesseract
   only for the long tail (Hebrew, Georgian, Armenian, Bengali), with rectified, polarity-normalized crops.
4. **Grouping**: deskew → stitch row pieces → link lines into blocks (height ratio ~1.35 + alignment) → XY-cut reading
   order (RTL-aware) → classify prose vs list blocks.

### Protocol / UI
- Response gains `blocks: [{box/quad, line_ids, text, lang, translations}]`, `angle`, per-line `script`; per-line
  `translations` stay for installed apps (AGENTS.md: never break installed apps).
- Live marks and the result thumbnail outline **blocks**; the sheet shows one card per block (tap a block on the
  frozen still ↔ its card), reflowed paragraphs, copy/share, data detectors (phone, URL, e-mail, address, date).
- Later: in-place translated overlay on the still with a "Show original" toggle; "Scan document" via ML Kit Document
  Scanner.

## 4. Step-2 plan (ordered; each phase verified on a real-photo set, not only rendered text)
| phase | what | why first | effort |
|---|---|---|---|
| **P0** | Eval set + harness: the bench's real photos (Commons, attributed) + rendered rotations/tilt/scripts + phone-shot photos from Genor; metrics = key-phrase recall, boxes per line, rotation success, latency; server test + Android instrumented test | every later phase must prove it beats the baseline | S |
| **P1** | Phone capture fixes (rotation listener, focus/metering, MAXIMIZE_QUALITY, freeze → capture, gyro + blur gate, JPEG pass-through with EXIF) | fixes the rotation complaint and most blur; no new deps | S–M |
| **P2** | Blocks as the product: protocol `blocks` + `angle`, per-block translation, block cards + block outlines (live + thumbnail), phone row stitching/deskew/RTL | fixes the "many small objects" complaint end to end | M |
| **P3** | Server engine swap to PP-OCRv6 ONNX + script recs + orientation classifiers + full-res crops; parity check vs the bench; retire EasyOCR | +30 points recall, 4/4 rotations, ~0.4 s instead of 10–33 s | M–L |
| **P4** | Phone engine: quads/angle in `OcrLine`, orientation check, adaptive scale + full-res crops, PP-OCRv5 mobile (ORT) packs, Tesseract hygiene, CameraX 1.6 | parity of the on-device path | M–L |
| **P5** | Extras: Document mode (ML Kit Document Scanner), data detectors, live stability states, in-place overlay, LLM/MADLAD translation eval, optional VLM "Read carefully" | polish and reach | M each |

Licences checked: PaddleOCR models/code Apache-2.0, Qwen3-VL / Qwen3.5 Apache-2.0, MADLAD-400 Apache-2.0, M2M100 MIT.
**Excluded**: NLLB-200 and Tower+ (non-commercial), Hunyuan OCR/MT (licence excludes the EU), Surya/Chandra weights
(revenue-capped), Nanonets-OCR2 (no licence).

## 5. Open decisions for Genor
1. **VLM second pass** (Qwen3-VL-4B via vLLM, ~8–10 GB VRAM while loaded on the shared 3090): on demand only, or not
   at all? It is the only thing that reads stylised/handwritten signs, but it can hallucinate.
2. **Translation upgrade**: try an LLM translator with whole-sign context (Gemma 4 E4B / Qwen3.5-4B, Apache-2.0) and
   MADLAD-400-3B against M2M100 1.2B in P5, or keep M2M100?
3. **In-place translated overlay** (Lens-style) vs the card sheet as the main result view.
