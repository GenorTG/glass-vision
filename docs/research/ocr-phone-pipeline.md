# Phone-side Read text pipeline: capture → preprocessing → on-device OCR → grouping → display

Research note, 2026-10-24. Scope: the Android half of "Read text" (still capture, preprocessing, ML Kit /
Tesseract OCR, block grouping, results UX). The server lane is covered separately. Status: research only, nothing
here is implemented yet. Sources are linked inline. **[measure]** marks a number that is an engineering estimate
and still has to be measured on a real phone. HANDOFF.md says Read text hasn't been run on physical hardware yet.

Owner's complaints, and the root cause found for each:

| Complaint | Root cause (phone side) |
|---|---|
| Snapshot not rotated the right way | The activity is locked to portrait (`AndroidManifest.xml:27`), and no use case ever gets `setTargetRotation`. A phone held sideways therefore produces a still that is upright relative to *portrait*, so the text comes out 90° rotated. CameraX docs say a locked-orientation app must update target rotation from an `OrientationEventListener` ([CameraX rotation](https://developer.android.com/media/camera/camerax/orientation-rotation)). |
| One visual text block comes back as many small objects | (a) Live "Mark text areas" and the Read thumbnail draw one box **per line** (`OnDeviceText.kt:414`, `ReadTextSheet.kt:105`), not per block. (b) `TextLayout.mergeBlocks` only joins blocks that overlap horizontally (`TextLayout.kt:87`), so pieces of one row that ML Kit split at a wide gap stay separate, which goes against PROTOCOL.md's "or side by side on one row". (c) Translation runs per line (`OnDeviceText.kt:497-515`), so a wrapped paragraph gets translated as fragments. On the server, EasyOCR is word-level; that's the other lane's job. |
| No image enhancement | There is no focus-before-capture, the capture runs in `CAPTURE_MODE_MINIMIZE_LATENCY` (`CameraBinder.kt:123`), and there's no blur/motion check. Tesseract crops get only a grayscale conversion: no polarity fix (light-on-dark signs), no deskew (`OnDeviceText.kt:168-174`). Section 2 explains why the fix is mostly *capture quality + geometry*, not binarization filters. |

---

## 1. Capture quality

### 1.1 Rotation (the #1 fix)
- CameraX semantics: `rotationDegrees` (ImageProxy) / EXIF (file) = "the rotation degrees by which the captured image
  needs to be rotated clockwise to match ImageCapture's target rotation". By default the target rotation is the
  display rotation. With `screenOrientation="portrait"` that is always ROTATION_0, so stills are always
  portrait-upright ([CameraX rotation](https://developer.android.com/media/camera/camerax/orientation-rotation)).
- Official recipe for a locked orientation: `OrientationEventListener.onOrientationChanged` → map degrees
  `45..134 → ROTATION_270, 135..224 → ROTATION_180, 225..314 → ROTATION_90, else ROTATION_0` → set
  `imageCapture.targetRotation` (and analysis). Enable it in `onStart`, disable it in `onStop`. Preview is excluded in the
  four-orientation case (same page).
- **Glass Vision specifics:** set it **only on ImageCapture** (BoxMapper and the live overlay assume portrait analysis
  frames). Keep the last known value: the listener reports `ORIENTATION_UNKNOWN` when the phone is flat (a document on a
  table), and in that case the content-based fallback (1.1b) decides. Add ~10° hysteresis so a phone held near 45° doesn't flip.
- 1.1b **Content-based check** (handles a phone flat over a page, or a sign that is itself sideways): ML Kit
  `Text.Line.getAngle()` gives each line's rotation in degrees, clockwise positive, range [-180, 180], and
  `getCornerPoints()` gives a 4-point quad that "is not necessarily a rectangle"
  ([Text.Line ref](https://developers.google.com/android/reference/com/google/mlkit/vision/text/Text.Line); the
  rotation info was added Sept 2022 per [ML Kit release notes](https://developers.google.com/ml-kit/release-notes)).
  Take the confidence-weighted median angle. If it is near ±90° or 180°, rotate the bitmap and read again.
  **Don't rely on `InputImage.fromBitmap(bmp, rotationDegrees)` for this**: an open ML Kit bug (Apr 2025) reports that
  every `rotationDegrees` value gives the same result, and that the workaround is to rotate the bitmap yourself
  ([mlkit#937](https://github.com/googlesamples/mlkit/issues/937)). Our code already passes upright bitmaps with 0
  (`OnDeviceText.kt:113`). Keep it that way.
- Model option if the angle heuristic fails on real photos: PaddleOCR `PP-LCNet_x1_0_doc_ori` (4 classes 0/90/180/270,
  7 MB, 99.06 % top-1, ~3 ms server CPU) ([doc](https://www.paddleocr.ai/latest/en/version3.x/module_usage/doc_img_orientation_classification.html)),
  and per-line 0/180 `PP-LCNet_x0_25_textline_ori` (0.96 MB, 98.85 %)
  ([doc](https://www.paddleocr.ai/v3.3.2/en/version3.x/module_usage/textline_orientation_classification.html)). Both
  are Apache-2.0 and run under ONNX Runtime. Tesseract OSD (`--psm 0`) is a third option, but it needs page-like text
  and reports a top-vs-second confidence margin
  ([OpenMPF README](https://github.com/openmpf/openmpf-components/blob/master/cpp/TesseractOCRTextDetection/README.md)).
  It's poor for short sign text **[measure]**, so not recommended.

### 1.2 Focus, exposure, torch
- **CameraX does not refocus before a still in our mode.** In the CameraX 1.3.4 binary we ship
  (`Camera2CapturePipeline.submitStillCaptures`, checked with `javap` on the cached AAR), the `AfTask` is added only
  when `captureMode == CAPTURE_MODE_MAXIMIZE_QUALITY`. Even then it fires only if AF mode is AUTO/MACRO and AF state is INACTIVE.
  The default mode is CONTINUOUS_PICTURE, so the shot uses whatever the continuous AF has at that moment. AE precapture runs only with a
  flash unit and flash required. So `MINIMIZE_LATENCY` + continuous AF = no focus guarantee.
- Fix: before `takePicture`, call `CameraControl.startFocusAndMetering(FocusMeteringAction)` with metering points
  at the **live text marks** (we already have their boxes in analysis coordinates), using
  `SurfaceOrientedMeteringPointFactory(w, h, imageAnalysis)`. Use flags AF|AE (AE on the text helps backlit signs) and
  wait for `isFocusSuccessful` with a ~800 ms cap. The default point size is 0.15, auto-cancel defaults to 5 s, and
  execution is best-effort when the device doesn't support AF/AE regions
  ([CameraX configuration](https://developer.android.com/media/camera/camerax/configuration)). Also add tap-to-focus
  through `previewView.meteringPointFactory` (same page).
- `setCaptureMode(CAPTURE_MODE_MAXIMIZE_QUALITY)`: JPEG default quality goes from 95 to 100, and the HAL may use
  its quality path (multi-frame NR) ([ImageCapture.Builder](https://developer.android.com/reference/androidx/camera/core/ImageCapture.Builder)).
  The default mode is MINIMIZE_LATENCY ([options](https://developer.android.com/media/camera/camerax/take-photo/options)).
  Expect +100–500 ms per shot **[measure]**. That's worth it for a one-shot read.
- ZSL (`CAPTURE_MODE_ZERO_SHUTTER_LAG`) is experimental, needs API 23+ and PRIVATE reprocessing, doesn't work with
  flash ON/AUTO, and returns the ring-buffer frame closest to the button press
  ([ZSL](https://developer.android.com/media/camera/camerax/take-photo/zsl)). That frame is the one most likely shaken by the tap itself, so it's **wrong for OCR**.
- Torch: `CameraControl.enableTorch()` after checking `CameraInfo.hasFlashUnit()`. While the torch is on it overrides
  `flashMode` ([configuration](https://developer.android.com/media/camera/camerax/configuration)). Offer a torch
  toggle in Read text. Prefer the torch over flash for glossy menus and signs (flash creates a specular hotspot on the text).
- Exposure compensation (`setExposureCompensationIndex`) with a −1 EV option for back-lit LED signs and bright screens (same page).
- Upgrade CameraX 1.3.4 → **1.6.x** (1.6.0 stable 2026-03-25, 1.6.1 2026-05-06, built on CameraPipe)
  ([release notes](https://developer.android.com/jetpack/androidx/releases/camera)). The 1.5 line added the
  SessionConfig/Feature Group API and Low Light Boost (`enableLowLightBoostAsync`, preview/video only)
  ([CameraX 1.5 blog](https://android-developers.googleblog.com/2025/11/introducing-camerax-15-powerful-video.html)).
  Nothing there is required for this work, but staying on 1.3 means missing the device quirk fixes.
- CameraX Extensions (NIGHT/HDR/AUTO) apply to Preview + ImageCapture. ImageAnalysis support per mode has to be
  checked with `isImageAnalysisSupported`, and all three use cases may fail to bind together below FULL level
  ([ExtensionsManager](https://developer.android.com/reference/androidx/camera/extensions/ExtensionsManager)).
  We need analysis for live marks → **don't use Extensions**. A rebind to Night for a single still would cost
  more than it gains on signs.

### 1.3 Sharpness / motion gating
- Blur metric: variance of the Laplacian on the luma of the text regions. pyimagesearch's starting threshold is 100,
  but it is "quite domain dependent" and depends on resolution
  ([pyimagesearch](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)). Use a **relative**
  threshold instead: compare against the running max over the last ~1 s of analysis frames at the same scale.
  A 3×3 Laplacian over a 640×480 Y plane in plain Kotlin takes ~2–4 ms **[measure]**, so no OpenCV is needed.
- Motion gate: `Sensor.TYPE_GYROSCOPE` angular speed
  ([motion sensors](https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion)). After the
  button press, wait until |ω| < ~0.15 rad/s for ~100 ms (cap at 600 ms), then fire. The press itself shakes the phone.
- Post-check: compute the Laplacian on the decoded still's text crops. If it's below ~50 % of the live max, take one
  automatic retake (only one, then show the result with a "blurry — retake?" hint).
- Burst / multi-frame fusion: not worth building ourselves. `MAXIMIZE_QUALITY` already lets the HAL do multi-frame
  NR on most flagships. A cheap alternative is to take 2 stills and keep the sharper one **[measure]**.

### 1.4 Resolution, small text, zoom
- ML Kit: "each character should be at least 16x16 pixels … generally no accuracy benefit for characters to be
  larger than 24x24 pixels" ([ML Kit v2 Android](https://developers.google.com/ml-kit/vision/text-recognition/v2/android)).
  So: the read scale should come from the **measured text height**, not a fixed 2560 (`TextLayout.kt:136`). Large
  sign text can be read at 1280 px (4× fewer pixels, faster), while small print needs the full 4000 px or crops.
- Small or distant text: optical zoom beats super-resolution. `CameraControl.setZoomRatio` switches to the
  telephoto lens on logical multi-camera phones **[measure per device]**. Offer pinch-zoom in the viewfinder.
- Super-resolution (Real-ESRGAN etc.): a GAN SR model can invent glyph strokes, it's far too slow at 12 MP on a
  phone, and recognizers already resample to 48 px line height (PP-OCRv5 rec input `[3, 48, 320]`,
  [config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml)).
  **Not recommended**. Upscale small line crops bicubically (what `TextLayout.cropScale` already does for Tesseract).

---

## 2. Preprocessing

### 2.1 What *not* to do
- **Don't binarize or threshold for ML Kit / PP-OCR.** They are deep recognizers trained on natural scene images.
  Adaptive threshold or CLAHE destroys anti-aliasing and colour contrast that they use **[engineering consensus;
  measure on our test set]**. Feed them upright natural RGB at the right scale.
- Binarization only helps Tesseract. Tesseract binarizes internally with Otsu (v5 adds Adaptive Otsu and Sauvola), which is
  "suboptimal … if the page background is of uneven darkness". For 4.x and later, "use dark text on light background". It needs
  a ~10 px border, and its "line segmentation reduces significantly if a page is too skewed"
  ([tessdoc ImproveQuality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)).

### 2.2 What to do (in order of value)
1. **Orientation**: see 1.1.
2. **Per-line rectification for crop readers**: warp each line's ML Kit `cornerPoints` quad to a horizontal strip
   (`Matrix.setPolyToPoly` + `Canvas.drawBitmap`; no OpenCV needed). This replaces today's axis-aligned crop, which on
   a tilted sign includes the neighbouring lines.
3. **Polarity for Tesseract**: compare the median luma of the crop border ring with the inner glyph area, and invert if the text is
   lighter than the background (common on white-on-blue/green signs). Then stretch contrast to 1–99 percentile. Optionally use
   Sauvola only for Tesseract.
4. **Mild contrast for dark or backlit stills (all engines)**: apply a global gamma or a CLAHE on the L channel *only* when the luma
   histogram is crushed (p99 − p1 < ~80) **[measure]**.
5. **Documents (paper, menus, flat)**: ML Kit **Document Scanner API** (`play-services-mlkit-document-scanner`,
   GA 2025-09-29). It does auto-capture, edge detection, crop, automatic upright rotation, filters, and shadow, stain and finger
   removal (SCANNER_MODE_FULL). Models come through Play services (low APK impact). It needs no camera permission, API 21+
   and ≥ 1.7 GB RAM (otherwise `UNSUPPORTED`). **It is its own full-screen Google UI** launched with
   `getStartScanIntent()` → activity result → page JPEG/PDF URIs. There's no headless or inline API, and customization
   is limited to page limit, gallery import and mode
   ([overview](https://developers.google.com/ml-kit/vision/doc-scanner),
   [Android guide](https://developers.google.com/ml-kit/vision/doc-scanner/android),
   [release notes](https://developers.google.com/ml-kit/release-notes)). → Offer it as a **"Scan document"** secondary
   action that feeds the returned JPEG into the same `TextReader.read`. Don't put it in the main sign flow.
6. Perspective correction for a sign in the main flow: skip the whole-image warp. Line-level rectification (2) gets
   most of the gain. Whole-image quad detection (OpenCV contours + `approxPolyDP`) only works on rectangular
   documents, and the Document Scanner already covers those.
7. Dewarping (curved pages): UVDoc is ~30 MB, ~870 ms on a server CPU, with input fixed at 720×496 in the ONNX export
   ([PaddleOCR unwarping](https://www.paddleocr.ai/v3.4.1/en/version3.x/module_usage/text_image_unwarping.html),
   [ONNX export](https://huggingface.co/fredcallagan/uvdoc-grid-onnx/blob/main/example.py),
   [paper](https://arxiv.org/abs/2302.02887)). Too heavy for the phone path. Leave it to the Document Scanner or to the server.

### 2.3 OpenCV?
`org.opencv:opencv` is on Maven Central since 4.9 (minSdk 21)
([OpenCV Android docs](https://docs.opencv.org/4.x/d5/df8/tutorial_dev_with_OCV_on_Android.html)), but it adds
tens of MB per ABI **[measure]**, and the APK is already 75 MB (HANDOFF). Everything in 2.2 can be done with
`Matrix`/`Canvas`/`ColorMatrix` and small Kotlin loops. The one OpenCV-only item is `Photo.inpaint` for the
overlay mode (section 5). Only take OpenCV if we build that, and then consider a custom `imgproc+photo`-only build.

---

## 3. On-device OCR engines

| Engine | Scripts | Size | Latency | Notes | License |
|---|---|---|---|---|---|
| **ML Kit Text Recognition v2** (current) | Latin (bundled), Chinese, Devanagari, Japanese, Korean | bundled ≈ 4 MB per script per ABI; unbundled ≈ 260 KB + Play-services download ([doc](https://developers.google.com/ml-kit/vision/text-recognition/v2/android)) | "real-time" for Latin; no published ms figures. 12 MP still **[measure]** | Hierarchy Block→Line→Element→Symbol, each with box + cornerPoints. Line has `angle`, `confidence`, `recognizedLanguage` ([ref](https://developers.google.com/android/reference/com/google/mlkit/vision/text/Text.Line)). Confidence is **0** for unbundled models on Play services < 22.30. Latest artifacts: `text-recognition:16.0.1`, `play-services-mlkit-text-recognition:19.0.1` (2024-08-07) ([release notes](https://developers.google.com/ml-kit/release-notes)), so we're current. Finds no Arabic/Hebrew/Thai/Bengali lines at all (AGENTS.md). | Proprietary, free ML Kit terms |
| **PP-OCRv5 mobile** (det + per-script rec) via ONNX Runtime | ch/en/ja unified; `latin`, `eslav`, `cyrillic`, `arabic`, `devanagari`, `korean`, `th`, `el`, `ta`, `te`, `en` rec models ([multi-lang doc](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)). **No Hebrew, Georgian, Armenian, Bengali**. | det ≈ 4.6 MB, rec ≈ 16 MB (unified), per-script rec smaller **[measure]** ([HF blog](https://huggingface.co/blog/baidu/ppocrv5), [oar-ocr models](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md)) | Official Android demo on a GM1900 (OnePlus 7, SD855), 5 lines: **420 ms** total (det 349 ms, rec 66 ms), cold load 158 ms ([PaddleOCR Android](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/cross_platform/android_deployment.html)) | Detection Hmean 0.770 vs v4 mobile 0.624; recognition 0.80 vs 0.53 ([PP-OCRv5](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)). Per-script rec accuracy 80–94 %. Detector finds scene text in any script, which fills ML Kit's Arabic/Thai detection gap. | Apache-2.0 (models + code) |
| **PP-OCRv6 small / tiny** | 50 languages (ch, zh-tw, en, ja + 46 Latin-script); tiny 49 (no ja). Cyrillic/Arabic etc. **not listed** | 7.7 M / 1.5 M params | Apple M4 ONNX Runtime per image: small 1.29 s, tiny 0.35 s, v5_mobile 1.10 s ([PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)) | small: det 84.1 / rec 81.3 vs v5_mobile 75.2 / 73.7 on the same in-house benchmark. Official Android demo supports v6 small/tiny. | Apache-2.0 |
| **RapidOCR** | PaddleOCR models repackaged | – | – | Multi-runtime (ORT, OpenVINO, MNN…), has an `android` folder but no documented SDK ([repo](https://github.com/RapidAI/RapidOCR)). No advantage over the official PaddleOCR `ppocr-sdk` module. | Apache-2.0 |
| **Tesseract (Tesseract4Android 4.9)** (current fallback) | 100+ incl. Hebrew, Georgian, Armenian, Bengali, Tamil… | 1–15 MB per traineddata | 10–60 ms per line crop (`OnDeviceText.kt:319`) | Document engine. Good on clean, rectified, dark-on-light line crops, useless on whole camera frames (AGENTS.md). Keep it only for scripts no neural engine covers. | Apache-2.0 |
| ONNX Runtime Android | – | prebuilt AAR 24.4 MB (arm64 `.so` 16.3 MB); custom op-reduced build 7.5 MB AAR / 4.0 MB arm64 (v1.18 numbers) ([ORT custom build](https://onnxruntime.ai/docs/build/custom.html)) | – | PaddleOCR's SDK uses `onnxruntime-android` 1.21.1, minSdk 26 (= ours) | MIT |

Licensing: Apache-2.0 and MIT are compatible with the repo's AGPL-3.0. ML Kit is a proprietary binary (already the case today).

**Recommendation:**
- **Keep ML Kit v2 as the primary engine** for Latin and CJK/Devanagari. It's the fastest and best on Latin scene text, has no
  download for Latin, and gives quads, angles and languages.
- **Add PP-OCRv5 mobile (ONNX Runtime), downloaded on demand** like the tessdata packs. Use its *detector* for scripts
  ML Kit doesn't box (Arabic, Thai) and its *recognizers* on rectified ML Kit/PP line quads for `eslav/cyrillic`,
  `arabic`, `el`, `th`, `ta`, `te`. This replaces the Tesseract crop reads and the sparse/auto whole-image scans for those scripts.
- **Keep Tesseract only** for Hebrew, Georgian, Armenian, Bengali and the other long-tail packs, fed with rectified,
  polarity-normalized crops.
- Cost: ORT +4–16 MB per ABI. Fetch models on first use (det 4.6 MB + one rec per chosen script).

---

## 4. Grouping into paragraphs/blocks and reading order

How ML Kit does it: `TextBlock` is its paragraph-ish unit, but on signs it often makes each line its own block
(AGENTS.md). `Line.getText()` joins elements in reading order (left to right for Latin)
([ref](https://developers.google.com/android/reference/com/google/mlkit/vision/text/Text.Line)). So we need our own
grouping on top of it, with ML Kit's blocks as a prior.

Recommended algorithm (all on quads, after deskew):
1. **Deskew frame**: global angle θ = confidence-weighted median of `line.angle`. Rotate all quad points by −θ, then
   take height h = mean distance between the top and bottom edges of the quad (not the AABB height, which grows
   with tilt). Today's `sameRow`/`mergeBlocks` use axis-aligned boxes (`TextLayout.kt:43-46, 70-94`), so tilted lines
   overlap and interleave.
2. **Row stitching**: line pieces A and B are one line if their baselines differ by < 0.3·h, the height ratio is < 1.3 and
   the horizontal gap is < 1.2·h (a wider gap means a column gutter, so keep them separate).
3. **Block linking** (union-find, as today): B under A joins if
   - the vertical gap is < 0.8·min(h)
   - the height ratio is < 1.35 (1.6 today merges a title with its subtitle)
   - they're aligned: left edges within 1·h, OR centres within 1·h (centred signs), OR right edges within 1·h, OR horizontal
     overlap ≥ 30 % of the narrower
   - no other line lies between them.
   Side-by-side pieces on one row (step 2) are already merged, which covers the protocol's "side by side on one row" case
   that `TextLayout.kt:87` misses.
4. **Reading order**: recursive XY-cut over the block hulls. Cut at the widest whitespace valley (horizontal first,
   then vertical), recurse, stop when no valley ≥ 1.5·median h exists. In the fallback, cluster into rows and sort
   ([XY-cut, Nagy & Seth 1984, as summarised in "Reading Order Inference for Complex Document Layouts"](https://arxiv.org/pdf/2607.01018); the same idea works for screen
   order in [Screen Recognition](https://arxiv.org/pdf/2101.04893)). XY-cut needs clean gutters and is skew-sensitive,
   which is why step 1 deskews first (same papers). Inside a block, order the lines top to bottom. Within a row, sort
   **right to left for RTL scripts** (`Character.getDirectionality` of the majority of letters). Today it's always left to right (`TextLayout.kt:55`).
5. **Prose vs list per block** (this decides how to translate it): a block is *prose* if it has ≥ 2 lines, every line except the last
   fills ≥ 70 % of the block width, and lines don't end in `:`/digits. Prose gets joined (de-hyphenate a trailing `-`
   before a lowercase letter) and **translated as one string**. List or sign blocks are translated per line, but with the
   block text as language-ID context (already done in `TextLanguages.assign`).
6. Output the **block quad** (union in the deskewed frame, rotated back) for drawing. Lines stay in the protocol as they
   are, so this needs no wire change.

Cost: O(n²) on n ≤ ~200 lines, < 5 ms **[measure]**. All pure Kotlin in `TextLayout`, unit-testable like today.

---

## 5. Results UX

- **Immediate feedback**: on press, freeze the preview (`PreviewView.bitmap`) with a shutter flash, show "Hold
  still" while focus/gyro gating runs (≤ ~1 s), then slide up the sheet. Today the sheet opens expanded *before* the
  capture (`ReadTextFlow.kt:40-44`): the user reacts to it and moves, and there's no stable framing.
- **Block-level visuals**: draw one rounded quad per block on the frozen still and in live "Mark text areas", not one
  per line. Tap a block on the image to scroll and highlight its card in the sheet, and the reverse.
- **Selectable text**: block cards with `android:textIsSelectable` (system selection handles and the copy/share/
  translate toolbar). Add a "Copy block" chip per card. Android has no public "Live Text on an image" API for apps, so
  a per-glyph selection overlay on the photo would be custom work with little gain over selectable cards **[judgment]**.
- **In-place translated overlay (Google Lens style)**: Lens AR Translate *replaces* the original text using the
  GAN-based erasing behind Magic Eraser, tuned to ~100 ms
  ([TechCrunch](https://techcrunch.com/2022/09/28/google-turns-to-machine-learning-to-advance-translation-of-text-out-in-the-real-world/),
  [9to5Google](https://9to5google.com/2022/09/28/google-lens-ar-translate/)). A feasible version on our still (not live):
  - per **block**, sample the background colour as the median of a ring around the quad (dark/light glyph mask by Otsu inside the quad)
  - fill the quad with that colour, or `Photo.inpaint` TELEA on the glyph mask (OpenCV) for textured backgrounds
  - render the translation with `StaticLayout`, auto-sized to fit the quad, mapped with `Canvas.setPolyToPoly` for perspective
  - text colour = the glyph median.
  Make it a toggle "Overlay / List" on the thumbnail, defaulting to List (more robust).
- Copy and share: keep the current blank-line-separated blocks (`ReadTextFormat`).

---

## 6. Recommended phone pipeline

```
[live] analysis 1280×960 → ML Kit marks (block quads) + per-frame Laplacian on text regions + gyro ω
   │ press "Read text"
   ▼
freeze preview, haptic ──► targetRotation from OrientationEventListener (last known if flat)   0 ms
   ▼
startFocusAndMetering(points = largest live text blocks, AF|AE), wait ≤ 800 ms               150–800 ms [measure]
gyro gate |ω| < 0.15 rad/s for 100 ms (cap 600 ms)                                           0–600 ms
   ▼
takePicture  MAXIMIZE_QUALITY, 4:3 ~12 MP, torch if user enabled                             300–700 ms [measure]
   ▼  keep JPEG bytes + rotationDegrees (no full ARGB decode)
SERVER backend: send the JPEG bytes as is, adding the EXIF Orientation tag (the server applies EXIF, PROTOCOL.md)
                → no decode/rotate/re-encode on the phone
PHONE backend:
   decode at 1/2 (inSampleSize=2 → 2000×1500) + rotate                                       ~60 ms [measure]
   blur check (Laplacian on text areas vs live max) → one auto-retake if blurry
   ML Kit v2 pass 1 on 2000 px                                                                200–600 ms [measure]
   orientation check (median line angle; ±90/180 → rotate, rerun)                           (rare)
   small-text pass: lines with h < 16 px at pass-1 scale → BitmapRegionDecoder full-res
                    crops of those regions → ML Kit again                                     +100–300 ms when needed
   script fallback: low-conf / non-Latin lines → rectified quad crops →
                    PP-OCRv5 rec (ORT) for cyr/ar/el/th/ta/te; Tesseract (polarity-fixed) for he/ka/hy/bn…
                    ML-Kit-blind scripts (ar/th): PP-OCRv5 det on 960 px                     +50–400 ms [measure]
   grouping (section 4) → blocks, reading order, prose/list
   language ID per block → translate per block (prose) / per line (lists)
   ▼
sheet: frozen still with block quads ↔ selectable block cards; Overlay toggle
Target: press → first text ≤ 1.5 s on a 2021+ mid-range phone (Latin), ≤ 2.5 s with a fallback script [measure]
```

Memory: a 12 MP ARGB bitmap is 48 MB, and `CameraBinder.upright` holds two (decode + rotated copy) at once. Decoding
at inSampleSize 2 (12 MB) plus region decodes for small text removes the OOM risk on low-RAM phones.

---

## 7. Defects in the current Kotlin code relative to best practice

| # | Where | Defect | Fix |
|---|---|---|---|
| 1 | `android/app/src/main/AndroidManifest.xml:27` + `CameraBinder.kt:122-125` (no `setTargetRotation` anywhere) | Portrait-locked activity, and ImageCapture's target rotation is never updated → stills of a phone held sideways are 90° off. This is the owner's rotation complaint. | `OrientationEventListener` → `capture.targetRotation` (1.1), plus the content-angle fallback. |
| 2 | `CameraBinder.kt:123` | `CAPTURE_MODE_MINIMIZE_LATENCY`: JPEG q95, and (CameraX 1.3.4) no AF trigger before the still. | `MAXIMIZE_QUALITY` + explicit `startFocusAndMetering` on the text marks (1.2). |
| 3 | `CameraBinder.kt:56-69`, `StillCapture.kt:50-78` | No focus, motion or blur gating, no retake. The still is taken at the instant of the tap. | Gyro gate + post-capture Laplacian check (1.3). |
| 4 | `ReadTextFlow.kt:40-44` | The sheet opens expanded before capture → no frozen frame, and the user's reaction moves the phone. | Freeze → capture → sheet (5). |
| 5 | `CameraBinder.kt:163-170` | `toBitmap()` decodes the full 12 MP and then rotates into a second 48 MB bitmap, even though both readers downscale to 2560 next. | Keep JPEG bytes. Decode with inSampleSize, use region decode for crops. |
| 6 | `ServerTextReader.kt:113-116` | Decode → rotate → downscale → re-encode q90 of a JPEG the camera already encoded: generation loss plus CPU time. | Upload the camera JPEG bytes with an EXIF Orientation tag (the server applies EXIF per PROTOCOL.md). If the server must downscale anyway, keep that on the GPU side. |
| 7 | `OnDeviceText.kt:466-469` | The still is downscaled to 2560 **before** the Tesseract crops, so crop readers get the reduced pixels instead of the 4000 px original. | Crop from the full-res source (region decode). |
| 8 | `OnDeviceText.kt:129-135` | ML Kit `cornerPoints`, `angle`, `recognizedLanguage` and element-level data are dropped and only the AABB is kept. | Carry the quad + angle + lang in `OcrLine`. They drive deskewed grouping, rectified crops and the orientation check. |
| 9 | `OnDeviceText.kt:164-174` | Tesseract crops are axis-aligned (neighbour lines get in on tilted text), grayscale only, and have no polarity normalization. Tesseract 4+ wants dark text on light ([tessdoc](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)), and white-on-colour signs are common. | Rectify the quad to a strip, invert if light-on-dark, stretch contrast. |
| 10 | `OnDeviceText.kt:284` (with `:134`) | `line.conf` is 0 for unbundled ML Kit models on Play services < 22.30 ([ref](https://developers.google.com/android/reference/com/google/mlkit/vision/text/Text.Line)) → every CJK/Devanagari line counts as "unsure" and gets re-read by every pack. | Treat conf == 0 as unknown. Gate on the script mismatch (`TextLanguages.inScript`) instead. |
| 11 | `OnDeviceText.kt:413-414` | Live "Mark text areas" emits one detection per line → many small tracked boxes for one sign. | Emit one box per merged block (union quad). |
| 12 | `ReadTextSheet.kt:105` | The thumbnail outlines every line. | Outline blocks. Tap a block to highlight its card. |
| 13 | `TextLayout.kt:87` | `mergeBlocks` requires horizontal overlap, so pieces of one row split by ML Kit at a gap ("Gate   12") never merge. Contradicts PROTOCOL.md "side by side on one row". | Row stitching (section 4, step 2). |
| 14 | `TextLayout.kt:43-46, 51-55` | `sameRow`/`rowOrder` work on axis-aligned boxes and always order left to right. On tilted text, consecutive lines overlap and interleave; Arabic/Hebrew rows come out reversed. | Deskew by the median angle first. Order RTL by script. |
| 15 | `TextLayout.kt:59` | `BLOCK_HEIGHT_RATIO = 1.6` together with the only-gap test merges headings with body text and separate stacked signs. | Ratio ~1.35 + an alignment test (section 4, step 3). |
| 16 | `TextLayout.kt:136` | Fixed 2560 px read size. ML Kit needs only 16–24 px per character ([doc](https://developers.google.com/ml-kit/vision/text-recognition/v2/android)): too big for large signs (slow), too small for fine print. | Adaptive scale from the measured line height + a small-text crop pass. |
| 17 | `OnDeviceText.kt:497-515`, `ReadTextFlow.kt:141-143` | Translation is per line, so a wrapped paragraph gets translated as fragments. | Translate prose blocks joined; keep lists per line (section 4, step 5). Map the result back to the block. |
| 18 | `OnDeviceText.kt:290-307` + `TextLayout.kt` `OcrPlan` | Whole-image Tesseract `PSM_SPARSE_TEXT`/`PSM_AUTO` scans cover scripts ML Kit can't box. Tesseract is a document engine and weak on scene text (AGENTS.md). | Use the PP-OCRv5 mobile detector (4.6 MB) + per-script recognizer for ar/th/el/cyr. Tesseract only for the long tail. |
| 19 | `android/app/build.gradle.kts:110` | CameraX 1.3.4. Current stable is 1.6.1 (2026-05). | Upgrade, then re-verify box alignment (HANDOFF item 3). |
| 20 | (missing) | No torch, exposure-compensation, tap-to-focus or zoom controls in the Read flow. | Section 1.2 / 1.4. |

## 8. Suggested order of work
1. Defects 1, 2, 3, 4 (capture: rotation, focus, gating, freeze). Biggest quality gain, no new dependencies.
2. Defects 8, 11–15, 17 (quads in `OcrLine`, block grouping, block visuals, block translation). Pure Kotlin, JVM-testable.
3. Defects 5–7, 16 (JPEG pass-through, adaptive scale, full-res crops). Speed and memory.
4. Defects 9, 10 (Tesseract crop hygiene).
5. PP-OCRv5 mobile via ONNX Runtime for cyr/ar/el/th/ta/te (defect 18), as downloadable packs.
6. "Scan document" through the ML Kit Document Scanner, then the optional overlay mode.

Each step needs a real-photo test set (signs at angles, sideways phone, backlit, white-on-colour, a paragraph,
Arabic/Cyrillic), not only rendered text (AGENTS.md gotcha), plus device timing for every **[measure]** above.
