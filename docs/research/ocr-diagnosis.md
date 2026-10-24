# Read text — diagnosis of the current implementation (2026-10-24)

Owner's report after device testing 0.6.0.13: one visual block of text comes back as many small word objects; the
snapshot is not rotated to the right orientation; no image processing improves OCR quality. Causes in our code:

## 1. Word fragments instead of lines/paragraphs (server path)
- EasyOCR's detector output is merged into horizontal boxes with `width_ths` (`server/glass_vision/pipelines/text.py`
  ~line 53); any word gap wider than that starts a new box, and `paragraph=False` (`RECOGNIZE_KWARGS`, line 45) keeps
  every box separate. Each box becomes one protocol `line`.
- `text_layout.reading_order` (`server/glass_vision/pipelines/text_layout.py`) only assigns a `block` id: it links
  same-row pieces into the same block but never **merges** them into one line (text and box stay separate).
- The app renders every `line` of a block on its own row (`android/.../ReadText.kt:40,65`: `joinToString("\n")`),
  and the thumbnail draws one box per fragment. So "Notausgang 2" becomes "Notausgang" / "2" and a paragraph turns
  into a column of words.
- Phone path is better but still line-level: ML Kit `TextBlock` → `Line` (`OnDeviceText.kt:129`), and
  `TextLayout.mergeBlocks` merges ML Kit blocks; paragraphs are shown as lines, not reflowed text.

## 2. No orientation handling
- `MainActivity` is locked to portrait (`AndroidManifest.xml:27`) and the `ImageCapture` target rotation is never
  updated from the physical device orientation, so a photo taken with the phone held sideways is "upright" for a
  portrait screen: the text is rotated 90°. CameraX docs recommend an `OrientationEventListener` →
  `imageCapture.targetRotation` for exactly this case.
- Neither backend detects text orientation (0/90/180/270) or skew: the server only applies EXIF
  (`text_read.py:59`, the app's in-memory JPEG has none after `upright()`), EasyOCR's recognizer reads horizontal
  text only, ML Kit tolerates small angles but not 90/180°.

## 3. No preprocessing / capture quality control
- No focus-before-capture, no blur/motion check, no best-frame choice; `CAPTURE_MODE_MINIMIZE_LATENCY`.
- No document/sign quad detection, perspective correction, deskew, contrast/illumination normalization or upscaling of
  small text before recognition (server reads the still downscaled to ≤2560 px as-is).

## 4. Smaller issues seen while testing
- Translation quality depends on line structure: fragments ("Notausgang" / "2") translate worse than whole lines
  or paragraphs; M2M100 sees one fragment at a time.
- First server read after idle ≈10 s (all 12 script recognizers + translator load).
