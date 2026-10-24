# How commercial products do camera OCR + translation

Research for making Glass Vision's Read text production grade. The owner's three complaints: (1) text that is one
block visually comes back as many small word-level objects, (2) the still isn't auto-rotated upright, (3) there's no
image processing to improve OCR. Sources are public: official docs, engineering blogs, papers, patents, press.
**[inference]** marks a conclusion of mine that no source states outright. Researched 2026-10-24.

---

## 0. TL;DR: the patterns that repeat across vendors

1. **The text line is the atomic unit; paragraphs/blocks are an explicit second level.** Google's own research shows
   that line-level detection beats word-level detection for layout (paragraph PQ 53.60 vs 39.11 on HierText). Azure
   returns `blocks → lines → words`. Google Cloud Vision returns `pages → blocks → paragraphs → words → symbols`. ML
   Kit returns `TextBlock → Line → Element → Symbol`. Apple's newest API returns `paragraphs` of `lines`. Nobody
   presents words as the user-facing object; words exist only for selection and highlight geometry.
2. **Orientation is a separate, explicit step, and its result is reported.** Examples: ABBYY
   `isTextOrientationDetectionEnabled` (default on, returns 0/90/180/270); Azure `angle` per page (e.g. 2.1243°, so
   fine skew too); Yandex `rotate` per page plus `orientation` per line; Dropbox runs a dedicated Inception-ResNet
   orientation net before OCR; the ML Kit Document Scanner does "automatic rotation detection to show documents
   upright"; ML Kit `Line.getAngle()` reports per-line rotation.
3. **Translation runs on whole sentences/blocks, not line by line**, and is then reflowed into the block's geometry
   (Lens: "translates whole sentences", "redistributes the translation into lines of similar length").
4. **Capture is freeze-frame-then-interact.** Apple Live Text shows a viewfinder indicator → tap → frozen frame with
   selectable text. Google Translate: "Tap the shutter to freeze"; Scan mode = still + "manually highlight text".
   Lens Go: instant capture from a constant hi-res stream on good phones, tap-to-capture on weak ones.
5. **Quality gates come before OCR.** Examples: ABBYY `assessQualityForOcr` (per-region 0–100); Apple
   `DetectLensSmudgeRequest`; Amazon's patented edge-width sharpness classifier for frame selection; Scanbot statuses
   `OK_BUT_BAD_ANGLES`, `OK_BUT_TOO_DARK`, `ERROR_TOO_DARK`; Google Translate's low-light flash hint.
6. **Heavy preprocessing (edges, perspective, illumination, shadow removal) is a *document* path**, separate from the
   scene-text path: Office Lens/Whiteboard It, Adobe Scan, ML Kit Document Scanner, VisionKit Document Camera,
   CamScanner, Dropbox. For signs and scene text, vendors feed the (cropped, ROI-limited, right-resolution) image to a
   neural recognizer trained on dirty synthetic data. Nobody publicly binarizes for scene-text OCR. **[inference]**
   Binarizing before a neural recognizer usually hurts.
7. **Live mode stabilises results over frames.** ABBYY RTR merges recognition across frames and reports a stability
   status (Tentative → TentativelyStable → Stable). Apple DataScanner gives items stable IDs with
   added/updated/removed callbacks. Google Translate's 2019 update was largely "less flicker".
8. **The right resolution matters more than filters.** ML Kit wants ≥16×16 px per character (no gain above 24×24).
   Azure's minimum text height is 12 px at 1024×768. Apple says to raise camera resolution for fine print, use a
   region of interest to crop away noise, and set `minimumTextHeight` to skip tiny text.

---

## 1. Google: Lens, Translate camera, Cloud Vision, ML Kit

### 1.1 Lens reading pipeline (Google Research, 2019, "Lens in Google Go")
Source: https://research.google/blog/giving-lens-new-reading-capabilities-in-google-go/
- **Capture**: CameraX. On higher-end phones, "capture instantaneous" from a constant high-resolution frame stream.
  On low-end phones, one frame is processed on tap "to produce a single, on-demand high-resolution image". The image
  is **scaled down and sent to the Lens server**: OCR is server-side here.
- **Detection**: a region proposal network finds **character-level** boxes, which are merged into lines. Word
  assembly uses a Hough transform (text assumed on parallel lines), then "Text Flow" (shortest path through a graph
  of boxes) for curved text.
- **Recognition**: separable CNNs plus a quantized LSTM, trained on reCAPTCHA and scanned Google Books.
- **Correction**: a context language model fixes OCR errors, and Knowledge Graph cues stop it from "correcting"
  proper nouns ("Cise is beauti640" → "life is beautiful").
- **Layout / reading order**: "CNNs detect coherent blocks such as columns or regions of consistent style or color".
  Within a block, alignment, language and paragraph geometry decide reading order. Geometry is modelled as
  **homographic distortion**, not axis-aligned boxes.
- **Translation display**: NMT on **whole sentences**. The translation is reflowed into lines of similar length with
  a matched font size. Colours come from classifying pixels into text and background (background assumed to be the
  majority, with different luminance) and averaging each class.
- **TTS**: per-word timing, so words highlight in sync with speech.

### 1.2 Translate "Word Lens" instant camera (2015, fully on-device)
Source: https://research.google/blog/how-google-translate-squeezes-deep-learning-onto-a-phone/
- Pipeline: heuristic (non-DNN) grouping of similar-coloured nearby blobs into lines → tiny CNN per **letter**
  (trained on letters and non-letters) → **fuzzy dictionary lookup** that tolerates misreads ("5" vs "S") →
  translation → **erase the original text using surrounding colours and redraw the translation in the original
  foreground colour**.
- Training data is synthetic with "dirt": reflections, smudges, highlights, rotation. Too much warping made "$" read
  as "S", so augmentation needed tuning.
- Constraints: low-end phones run ~50× slower than a laptop, so the code uses hand-tuned SIMD and cache-sized matmuls,
  runs in real time while the camera moves, and works offline.

### 1.3 Translate camera, 2019 revamp
Sources: https://9to5google.com/2019/07/10/google-translate-camera-revamp/ ,
https://www.neowin.net/news/google-translates-camera-translation-gets-support-for-more-languages/
- Three modes: **Instant** (live overlay), **Scan** (still image; the user highlights text), **Import** (gallery).
- Source "Detect language" auto-detects. NMT came to the camera (55–85 % fewer errors on some pairs). Works offline
  with downloaded packs, better online. The headline UX fix was **"more stable, less flickery"** overlays.
- Current help page (https://support.google.com/translate/answer/6142483): shutter = **freeze**; tap a word and drag
  anchors to select; the translation appears in a draggable card; "Show original text" toggle; Copy; Listen; a
  low-light **flash** button; small, unclear or stylized text called out as a known failure case.

### 1.4 Lens AR translate (2022): inpainting instead of boxes
Source: https://techcrunch.com/2022/09/28/google-turns-to-machine-learning-to-advance-translation-of-text-out-in-the-real-world/
- The original text is removed and the pixels rebuilt with GANs (the Magic Eraser tech), then the translation is drawn
  in place. The source doesn't say where it runs or how fast.

### 1.5 Google OCR research (production-adjacent papers by the Lens/Cloud OCR team: Bissacco, Fujii, Raptis…)
- **Unified Detector** (CVPR 2022), https://arxiv.org/html/2203.15143v2 :
  - One MaX-DeepLab model with 384 object queries gives one **mask per text line**, plus an N×N **affinity matrix**
    over lines. Paragraphs = threshold the affinity at 0.5, then union-find.
  - Ablation: **line-based detection, layout PQ 53.60 vs word-based 39.11**. That is the quantitative answer to
    "don't output words".
  - Against two-stage grouping (GCN on line boxes 50.10; Mask-RCNN + cluster 51.67) it wins at about half the
    compute. Google's own commercial GCP API scored 46.33 layout PQ on the same set.
  - Dataset **HierText**: word/line/paragraph annotations, 11,639 images, ~104 words per image
    (https://www.research.google/blog/announcing-the-icdar-2023-competition-on-hierarchical-text-detection-and-recognition/).
    Definitions worth copying:
    - **Line**: "space-separated clusters of words that are logically connected in one direction, and aligned in
      spatial proximity".
    - **Paragraph**: lines that "share the same semantic topic and are geometrically coherent".
  - Code: TF Model Garden `unified_detector`. Data: github.com/google-research-datasets/hiertext.
- **Hierarchical Text Spotter** (WACV 2024), https://arxiv.org/abs/2310.17674 : Bezier-polygon line detector plus
  the affinity matrix for paragraphs. A **line** recognizer is then split into characters and words, i.e. recognition
  also happens at line level and words are derived.
- **Rethinking Text Line Recognition Models** (2021), https://arxiv.org/abs/2104.07787 : the best line recognizer is a
  self-attention encoder + **CTC** decoder + external LM. It handles arbitrary line length; Transformer decoders
  degrade on long lines.
- **Sequence-to-label script ID** (ICDAR 2017), https://arxiv.org/abs/1708.04671 : a per-**line** script classifier
  over 232 languages / 30 scripts cut script-ID errors 16 % and the resulting CER 33 % versus counting per-character
  script codes. Google picks the script per line, then runs that script's recognizer.

### 1.6 Cloud Vision output model
Sources: https://docs.cloud.google.com/vision/docs/fulltext-annotations ,
https://docs.cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageResponse
- `Pages → Blocks → Paragraphs → Words → Symbols`. A block is a "logical element", with types TEXT, TABLE, PICTURE,
  RULER, BARCODE.
- Each level has `confidence` and `detectedLanguages`. Each symbol has a `detectedBreak` (SPACE, SURE_SPACE,
  EOL_SURE_SPACE = line-wrap, HYPHEN = end-of-line hyphen not in the text, LINE_BREAK = ends a paragraph). This is how
  a client rebuilds paragraph text with correct joins and de-hyphenation.
- **Rotation lives in the polygon**: vertices are always listed top-left, top-right, bottom-right, bottom-left
  *relative to the text*, so an upside-down word has its "top-left" at the bottom-right of the image. The orientation
  is recoverable from vertex order.

### 1.7 ML Kit Text Recognition v2 (what the phone backend already uses)
Sources: https://developers.google.com/ml-kit/vision/text-recognition/v2/android ,
https://developers.google.com/android/reference/com/google/mlkit/vision/text/Text.Line
- `Text → TextBlock → Line → Element → Symbol`. `Line` has **`getAngle()`** (−180…180, clockwise positive),
  `getConfidence()`, `getRecognizedLanguage()` (BCP-47) and `getCornerPoints()` (clockwise from top-left;
  perspective-skewed).
- Input guidance: ≥16×16 px per character, no gain above 24×24; bad focus is the main accuracy killer (prompt a
  recapture); pass the correct `rotationDegrees`. Live tips: throttle, `STRATEGY_KEEP_ONLY_LATEST`, draw once per
  result.
- **[inference]** The median `getAngle()` of confident lines estimates both the 90/180/270 orientation and the fine
  skew of a still at zero extra cost.

### 1.8 ML Kit Document Scanner (`GmsDocumentScanning`)
Sources: https://developers.google.com/ml-kit/vision/doc-scanner ,
https://android-developers.googleblog.com/2024/02/ml-kit-document-scanner-api.html
- A complete Google-provided flow, delivered via Play services, all on-device, **no camera permission needed**, used
  by Google Drive and Pixel Camera. Features:
  - "automatic capture with document detection"
  - "accurate edge detection"
  - **"automatic rotation detection to show documents upright"**
  - crop/rotate/reorder
  - filters: grayscale, auto-enhance
  - `SCANNER_MODE_FULL`: ML cleaning that erases stains and fingers
- Returns JPEG/PDF URIs. API 21+.
- **[inference]** This is a drop-in "Document" mode for Glass Vision: launch it, then feed the JPEG to the existing
  `/text/read` or `LocalTextReader`.

---

## 2. Apple: Vision, VisionKit, Live Text

- **VNRecognizeTextRequest** (WWDC19 session 234: https://developer.apple.com/videos/play/wwdc19/234/ ; reference:
  https://developer.apple.com/tutorials/data/documentation/vision/vnrecognizetextrequest.md):
  - **Fast path**: character detection plus a small ML model, one character at a time. "Meant for real-time", less
    memory.
  - **Accurate path**: a neural net that finds **strings and full lines**, then recognizes words and sentences. It
    has "much broader support" for **rotated or perspectively misaligned text**. Default; for asynchronous use.
  - Both paths are followed by **language correction**. Apple warns that it costs time and turns codes into words
    ("C001" → "cool"), so turn it off for serials.
  - `customWords` adds domain vocabulary ("HI11" fixed in the demo). `automaticallyDetectsLanguage` picks
    recognition model and LM. `recognitionLanguages` is in priority order.
  - `minimumTextHeight` (fraction of image height): smaller text is ignored and **the input is downscaled, so runtime
    drops**.
  - `regionOfInterest` removes surrounding noise and speeds things up.
  - Advice: raise capture resolution for fine print, lower it for billboards.
  - Results are **one observation per line** with `topCandidates(n)`. Word boxes come only on demand via
    `boundingBox(for: range)`. Apple recommends indexing several candidates for recall.
- **RecognizeDocumentsRequest** (WWDC25 session 272: https://developer.apple.com/videos/play/wwdc2025/272/):
  - Returns a `DocumentObservation` with text/tables/lists/barcodes, views as `transcript`, `lines` and
    **`paragraphs`**: "Lines can be grouped into paragraphs, giving a more natural view of how the text would be read.
    If a line is not part of a paragraph grouping, it's considered to be its own paragraph." Words are unsupported
    for CJK and Thai. 26 languages.
  - Built-in **DataDetection**: phones, emails, URLs, dates, money, units, flight and tracking numbers.
  - Apple explicitly frames the older line-only API as "losing important structural information": tables came back as
    separate per-cell objects.
- **DetectLensSmudgeRequest** (same session): 0–1 smudge score, threshold 0.9 in the example. Motion blur, long
  exposure and fog also score high, so it is a usable "image too blurry/smudged" gate with a retake prompt.
- **DataScannerViewController** (WWDC22 session 10025: https://developer.apple.com/videos/play/wwdc2022/10025/):
  - `qualityLevel` fast/balanced/accurate ("fast… for large, easily-legible text like signs").
  - Region of interest; **guidance labels** at the top of the screen; system or custom highlights with high-frame-rate
    tracking.
  - Items have **stable IDs over their lifetime** (didAdd/didUpdate/didRemove; update also fires when the
    transcription changes). Items are returned in **natural reading order**.
  - **`capturePhoto()` gives a high-quality still**: the same live-marks-then-still split as Glass Vision.
  - A12+/Neural Engine only.
- **VisionKit Document Camera** (WWDC19): "finds the document, crops it out", perspective-corrects and "evenly lights"
  the image, multi-page. Apple's recommended pre-step for documents.
- **Live Text UX** (https://9to5mac.com/2021/09/21/how-iphone-live-text-ocr-works-in-ios-15/ ,
  https://macrumors.com/guide/live-text-ios-16):
  - On-device (A12+). A yellow bracket appears when text is noticed; tapping the Live Text button freezes the frame
    and highlights all text; actions are Copy / Select All / Look Up / **Translate** / Share, with call/message for
    detected numbers.
  - iOS 16: the camera can be paused to get translations **overlaid** on the text, and Live Text works on paused video.

---

## 3. Microsoft: Office/Microsoft Lens, Azure Read, Document Intelligence

- **Office Lens / "Whiteboard It"** (MSR: https://www.microsoft.com/en-us/research/?p=284906):
  - Steps: edge detection to find board/page boundaries (manual border fallback), **perspective correction**, white
    balancing to a uniform background, colour saturation of pen strokes, glare reduction.
  - Inherited OneNote "AIE" steps: **automatic rotation, straightening, cropping, sharpening, shadow removal**, plus
    tone, exposure, contrast and denoise.
  - Blur measurement advised by MSR (Joshi, Nowozin).
  - Image cleanup ran on the phone; **OCR ran in the cloud** (OneNote).
  - Microsoft Lens Document mode tries to **auto-capture** when the frame is stable and falls back to a manual
    shutter (https://www1.wne.edu/information-technology/teaching-learning/doc/ms365/ms_lens.pdf).
- **Azure Vision Image Analysis 4.0 Read** ("OCR for images": signs, labels, posters; synchronous, "near real-time"):
  `readResult.blocks[].lines[].words[]`, each with a `boundingPolygon` (a 4-point quad, so rotation and perspective
  survive) and per-word confidence. https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-ocr
- **Read v3.2**: per page **`angle`** (the sample shows 2.1243°, i.e. fine skew, not only quadrants) and lines with
  `appearance.style` handwriting/print. `readingOrder=natural` for Latin, otherwise left-to-right. Passing a language
  is optional, and forcing a wrong one gives "incomplete and incorrect text". Mixed languages are supported by default.
  https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-read-api
- **Document Intelligence Read**: "runs at a higher resolution than Azure Vision Read". It adds **`paragraphs`**
  (spans into one `content` string) and per-page `angle`. Detected languages, handwriting styles, and searchable PDF
  output. Text height ≥12 px at 1024×768 (≈8 pt at 150 DPI); images 50…10,000 px.
  https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/read
- Product split: Microsoft runs **two Read editions**, a fast synchronous one for in-the-wild images and a
  higher-resolution asynchronous one for documents
  (https://learn.microsoft.com/azure/ai-services/computer-vision/overview-ocr). That matches a "sign" vs "document"
  mode.

---

## 4. Adobe Scan, CamScanner, Dropbox (document scanners)

- **Adobe Scan** (https://www.adobe.com/sensei/document-cloud-artificial-intelligence.html):
  - "Adobe Sensei automatically detects the boundaries to auto-crop and algorithmically correct document perspective".
  - "auto-cleans and removes shadows to improve the appearance and readability", text "crisper".
  - Auto-capture plus cloud OCR. No public engineering detail.
- **CamScanner** (https://dev.camscanner.com/): automatic border detection, "smartly removes the background", five
  enhancement modes, offline. No public algorithm detail.
- **Dropbox doc scanner** (the best public engineering write-up of a commercial scanner):
  - Document detection (https://dropbox.tech/machine-learning/fast-and-accurate-document-detection-for-scanning):
    - **Canny failed**: it amplified the text inside the page and missed the page edges. They switched to an **ML
      per-pixel edge-probability** map.
    - Hough lines (polar form) → corner candidates filtered by angle → enumerate quads → score = sum of edge
      probability along the perimeter.
    - Budget ≤100 ms/frame on CPU; achieved 8–10 fps on iOS.
  - OCR (https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning):
    - MSER word detector tuned for recall; CNN + BiLSTM + **CTC** word recognizer; >10 M synthetic words; a
      "Wordinator" splits multi-word boxes.
    - **A separate orientation predictor (Inception-ResNet-v2) rotates 90°/180° images upright before detection.**
      The hard part was *not* rotating already-upright images, so false rotations need a high-confidence gate.
    - CTC confidence drives lexicon correction. Inference moved to CPU at equal or lower cost.

---

## 5. ABBYY: TextGrabber, Mobile Capture SDK, Real-Time Recognition SDK

- **RTR SDK, live text**:
  - Recognizes several frames in parallel and **merges results to eliminate random errors**
    (https://support.abbyy.com/hc/en-us/community/posts/360009367759-RTR-OCR-ACCURACY).
  - No per-character confidence; instead a **result stability status** for the whole text. The Mobile Capture docs
    list Tentative (one frame), TentativelyStable (stable for 2 frames) and Stable (≥3 frames)
    (https://help.abbyy.com/en-us/mobilecapturesdk/1/cordova_help/cordova-result-status).
  - The app acts when the status reaches the level it needs.
- **Mobile Capture SDK core API**
  (http://help.abbyy.com/en-us/mobilecapturesdk/1/reactnative_help/reactnative-abbyymobilecapture/):
  `detectDocumentBoundary` (quad), `cropImage` (perspective), `rotateImage`, `assessQualityForOcr`, `recognizeText`.
  - **`recognizeText` has `isTextOrientationDetectionEnabled` (default true) and returns `orientation` ∈ {0, 90, 180,
    270}** and `textBlocks[]` → `textLines[]` → `charInfo[]`
    (http://help.abbyy.com/en-us/mobilecapturesdk/1/reactnative_help/reactnative-abbyymobilecapture-recognizetext/).
    Again: blocks of lines, words not first-class.
  - **`assessQualityForOcr`** returns rectangles typed Text/Unknown, each with an OCR-suitability **quality 0–100**
    (http://help.abbyy.com/en-us/mobilecapturesdk/1/reactnative_help/reactnative-abbyymobilecapture-assessqualityforocr/).
  - The Image Capture scenario aims "to automatically capture the best quality image suitable for OCR"
    (http://help.abbyy.com/en-us/mobilecapturesdk/1/ios_help/introduction/).
  - ABBYY patent US 9412030 names the frame defects it analyses: glare, blur, defocus, noise.

---

## 6. Translation-first apps: DeepL, Papago, Yandex

- **DeepL** (https://www.deepl.com/features/translate-image , https://support.deepl.com/hc/articles/24657655145372):
  - Mobile camera/photo: translation **overlaid** on the original; the user can switch the overlay off, view original
    and translation separately, or **rotate the image**.
  - Document image translation re-embeds the text into the image; ≥300 DPI recommended.
  - Small, badly lit, handwritten or stylized text is listed as a limitation.
- **Naver Papago Image Translation**
  (https://guide.ncloud-docs.com/docs/en/papago-image-translation-overview):
  - Two products: text output (OCR + translation) and an **image output via "Text Editing (inpainting +
    rendering)"** that preserves position, colour and size.
  - Auto source detection; images ≤1960×1960, ≤20 MB.
- **Yandex** (https://www.yandex.com/company/press_center/press_releases/2024/11-10-2024):
  - A fine-tuned, distilled **YandexGPT** translates image text "context-aware"; it handles fragmented or partial
    words.
  - It **erases the original text and redraws the translation matching font, size and colour**, removing artifacts
    and improving contrast.
  - Yandex Vision OCR (https://yandex.cloud/en/docs/vision/concepts/ocr): `pages → blocks → lines → words`, block
    layout types (header, footer, list…), **`rotate` per page and `orientation` per line ∈ {0, 90, 180, 270}**, and a
    separate `page-column-sort` model for multi-column reading order.

---

## 7. Capture-quality references (SDKs and patents)

- **Amazon, "Sharpness-based frame selection for OCR"** (US 9,418,316 / 9,576,210,
  https://patents.google.com/patent/US9418316): edge map → **edge transition widths** (sharp = narrow) → statistics →
  trained classifier decides whether a frame is sharp enough for OCR. The classifier is trained on frames labelled by
  OCR-ability after binarization.
- **Scanbot SDK** (https://api-docs.scanbot.io/document-scanner-sdk/capacitor/v8.0.0/types/DocumentDetectionStatus.html):
  - Explicit live statuses: OK, OK_BUT_BAD_ANGLES ("too much perspective distortion"), OK_BUT_TOO_DARK,
    ERROR_TOO_DARK (mean brightness < 50 by default), and others.
  - Auto-snap fires when "a document is detected, conditions are good and the auto-snapping time-out elapses".
    Sensitivity 0..1 maps to a 0–3 s delay, default 0.66.
- **Pixel Super Res Zoom** (https://research.google/blog/see-better-and-further-with-super-res-zoom-on-the-pixel-3/ ,
  https://arxiv.org/abs/1905.03277): multi-frame handheld super-resolution at ~100 ms per 12 MP frame, inside the
  camera HAL.
  - **[inference]** Glass Vision gets it for free on supporting phones by using CameraX zoom (`setZoomRatio`) for
    distant signs instead of upscaling a crop afterwards. Software super-resolution of a single JPEG isn't something
    any vendor documents for OCR.

---

## 8. Comparison matrix

| Product | Capture UX | Preprocessing | Recognition unit → output | Grouping / order | Translation display | Where |
|---|---|---|---|---|---|---|
| Google Lens / Translate | live overlay; shutter = freeze; Scan = highlight on still; flash hint | homography-aware geometry; (2022) GAN inpainting | char boxes → lines → words; line recognizer + LM + Knowledge Graph | CNN block detection + alignment/language/paragraph geometry; research: line masks + paragraph affinity | **sentence-level NMT**, reflowed into the block, colour-matched, original erased | live: on-device; Lens OCR: server (scaled-down image) |
| Google Cloud Vision | n/a | — | words/symbols with breaks | pages/blocks/paragraphs; rotation in vertex order | — | cloud |
| ML Kit Text v2 | developer's job | rotationDegrees | Block/Line/Element/Symbol, line angle, language, confidence | blocks | — | on-device |
| ML Kit Doc Scanner | auto-capture, edge detection | crop, perspective, **auto-rotate**, filters, stain/finger removal | (image only) | — | — | on-device (Play services) |
| Apple Vision / Live Text | indicator → tap → frozen frame; DataScanner guidance, stable item IDs, capturePhoto | Document Camera: crop, perspective, even lighting; smudge detector | accurate = line finder + recognizer + LM; customWords; min text height; ROI | lines → **paragraphs**, tables, lists; reading order; data detectors | overlay on paused frame (iOS 16) | on-device (A12+) |
| Microsoft Lens / Azure | auto-capture when stable | edges, perspective, white balance, glare, shadow, auto-rotate | words with polygons | blocks → lines; paragraphs (DI); page `angle`; natural reading order | — | cleanup on device, OCR cloud |
| Adobe Scan / CamScanner | auto-capture | boundaries, perspective, shadow removal, enhance modes | — | — | — | device + cloud |
| Dropbox | live quad detection 8–10 fps | ML edge map + Hough quad; orientation net | MSER words → CNN-LSTM-CTC | lines | — | device (detect) + server (OCR) |
| ABBYY | live text with multi-frame merge + stability; auto best-shot | boundary, crop, rotate, **quality 0–100 per region** | blocks → lines → chars; **orientation 0/90/180/270** | blocks | (TextGrabber: list) | on-device |
| DeepL | photo/camera | user rotate | — | — | overlay on/off, or side by side | cloud |
| Papago / Yandex | photo | — | OCR | blocks; Yandex: rotate/orientation, column-sort model | **inpaint + re-render** matching style; Yandex: LLM context translation | cloud |

---

## 9. What Glass Vision should copy (ranked by impact ÷ effort)

Current state, checked in code:
- Server: EasyOCR with `paragraph=False` and merge thresholds `width_ths=0.5`, `ycenter_ths=0.5`, `height_ths=0.5`
  (`server/glass_vision/pipelines/text.py:45-54`). A horizontal gap over half a box height starts a new box, so
  letter-spaced signs and justified text split into words.
- `/text/read` applies only the EXIF orientation (PROTOCOL.md "Read text").
- Translations align with `lines`, i.e. they are made per line.
- Phone: ML Kit + `TextLayout.mergeBlocks`.

1. **Lines as the unit, blocks as the user-facing object** (complaint 1). *High impact, low effort.*
   - Every vendor returns lines inside blocks/paragraphs; Google measured line > word for layout (53.6 vs 39.1 PQ).
   - Server: merge EasyOCR boxes into lines more aggressively (width_ths ≈ 1.0–1.5 × height for same-baseline boxes,
     plus a vertical-overlap test) before recognition or as post-merge.
   - Keep `block` but make it the primary entity in the sheet *and* the overlay: one box per block, word boxes only
     for tap-to-select.
   - Phone: present ML Kit `TextBlock`/`Line`, never `Element`.
   - Adopt HierText's definitions (line = same direction, aligned, near; paragraph = geometrically coherent lines) as
     the written grouping spec and test it on real photos.
2. **Translate per block, not per line** (part of complaint 1). *High impact, low effort.*
   - Lens translates whole sentences and reflows them into the block; Yandex sells "context-aware" whole-text
     translation.
   - Join a block's lines into one string, de-hyphenating line-end hyphens (cf. Cloud Vision HYPHEN/EOL_SURE_SPACE
     breaks), translate once, and show the translation per block. M2M100 on a single-line fragment of a two-line
     sentence is exactly the "one-word sign" failure AGENTS.md describes.
   - Protocol: add `blocks: [{box, line_ids, text, lang, translations}]` and keep per-line `translations` for old apps.
3. **Explicit orientation detection + deskew on the still** (complaint 2). *High impact, medium effort.*
   - ABBYY, Azure, Yandex, Dropbox and the ML Kit Doc Scanner all do this and *report* it.
   - Phone: take the confidence-weighted median of ML Kit `Line.getAngle()`. Near ±90/180, rotate the bitmap and
     re-run; a small angle → deskew (or just rotate the result boxes).
   - Server: a cheap 4-way test (detector box aspect: tall-narrow boxes ⇒ 90°/270°; then recognizer confidence at
     0° vs 180° on the top-N crops), or PaddleOCR's open-source orientation / text-line 180° classifiers.
   - Gate on high confidence (Dropbox's lesson: false rotations of upright images are the main risk).
   - Return `angle` in the response; the app should also send the still upright using the device rotation at capture
     time, not only EXIF.
4. **Read at native resolution where the text is: ROI and region tiling** (complaint 3, the processing that actually
   matters). *High impact, medium effort.*
   - Apple: ROI and minimum text height; ML Kit: 16–24 px per character; Azure: ≥12 px text.
   - Today a 4000×3000 still is read at ≤2560 px. Instead, detect on the downscaled image, then **recognize each line
     from a crop of the full-res still**, upscaled to the recognizer's height when lines are tiny.
   - Allow **tap-a-region to read** (Translate Scan mode, Lens text selection, Apple `regionOfInterest`): crop the
     tapped block from the full still and read only that.
   - For distant signs, a zoom control (CameraX zoom → the phone's multi-frame SR pipeline on Pixels) beats any
     post-hoc upscaling.
5. **Capture quality gate + guidance** (complaint 3). *Medium impact, low–medium effort.*
   - Before Read, check sharpness on the still: Laplacian variance or Amazon-style edge-width over detected text
     regions, normalised by text height. Also check brightness (Scanbot: mean < 50 ⇒ too dark → offer the torch,
     like Google Translate) and motion (gyro during exposure).
   - On failure, show a one-line hint ("Hold still", "Too dark – flash?", "Move closer") and retake automatically,
     instead of returning garbage. ABBYY's per-region quality 0–100 is a good output shape.
   - Optionally keep the last ~1 s of analysis frames and pick the sharpest if the still is blurred (Lens Go's
     high-res stream / ABBYY best shot).
   - Run CameraX tap-to-focus/metering on the text region before `takePicture`.
6. **Document mode via ML Kit Document Scanner.** *High impact for paper and menus, very low effort.*
   - Gives edge detection, perspective, auto-rotate, shadow and stain cleanup and auto-capture for free, on-device,
     with no extra camera permission.
   - Launch it from the Read pill ("Document" option), then send its JPEG through the existing read path. This is
     the Microsoft two-edition split (fast scene vs high-res document) at zero model cost.
7. **Live stability, no flicker** (Translate 2019, ABBYY stability, Apple item IDs). *Medium impact, low effort.*
   - Already tracked with ByteTrack. Expose a per-track "stable" state (seen ≥3 frames with consistent geometry) and
     only draw marks or chips once stable; snap block boxes to the union of their lines so they don't jitter.
8. **Data detectors on read text** (Apple DataDetection, Live Text call/message). *Medium impact, low effort.*
   Android `Linkify` / ML Kit Entity Extraction for phone numbers, URLs, emails, addresses and dates, shown as
   actions in the sheet. Also skip translation and language correction for codes and serials (Apple's "C001 → cool"
   warning; Glass Vision already skips digit-heavy lines).
9. **In-place translated overlay on the still** (Word Lens, Lens 2019/2022, Papago, Yandex, DeepL). *High wow,
   medium effort for the simple version.*
   - Per block: sample background and foreground colours (Lens: classify pixels by luminance, majority = background),
     fill the block quad with the background colour or `cv2.inpaint`, draw the translation reflowed to the block width
     with auto-fit font size, and give the sheet a toggle "Show original" (Translate, DeepL).
   - GAN inpainting is optional polish.
10. **Language-model correction / custom vocabulary** (Lens LM + Knowledge Graph, Apple `customWords`, Dropbox
    lexicon). *Medium impact, medium effort.* Use top-k beams or EasyOCR `beamsearch` with a per-language word list
    for low-confidence words only; never for codes.
11. **A learned unified detector + paragraph affinity** (Unified Detector / HTS, HierText data). *Highest ceiling,
    high effort.* Only worth it if the heuristics in items 1–2 measurably fail on a set of real photos. HierText
    (CC-BY) is also the right **evaluation set** for grouping quality even without training anything.

**Don't copy**: binarization or "magic colour" filters ahead of the neural recognizers. Scanners use them for the
saved document's appearance, and no vendor documents them as a scene-text OCR step **[inference]**.
