# Models and their licenses

The server downloads every model from its **original upstream source** (`scripts/download_models.py`; the
installer and the Docker entrypoint run it). Nothing is re-hosted by this project. Each file is checked against a
pinned sha256, recorded on 2026-10-22/24 from files downloaded from these same URLs. Hugging Face files are fetched at a
pinned commit (`resolve/<commit>/<file>`).

**This project's license does not cover the models.** Each has its own terms, and two of them are
**non-commercial research only**. If you use glass-vision for anything commercial, disable those pipelines
(faces, open_vocab) or replace their models.

| pipeline | file(s) | source | license |
|---|---|---|---|
| objects | `yolo11s.pt` (18 MB) | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt) | AGPL-3.0 ([Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); [commercial licenses](https://www.ultralytics.com/license)) |
| pose | `yolo11s-pose.pt` (19 MB) | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt) | AGPL-3.0 (as above) |
| pose (hands) | `mediapipe/hand_landmarker.task` (7.5 MB) | [Google MediaPipe models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) | Apache-2.0 ([model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf), "Licensed under Apache License, Version 2.0") |
| open_vocab | `yoloe-11s-seg.pt` (27 MB) | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt) | AGPL-3.0 (as above) |
| open_vocab | `mobileclip_blt.ts` (572 MB) — Apple MobileCLIP-B(LT) text encoder, TorchScript export by Ultralytics | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts) | **Apple ML Research Model Terms of Use — non-commercial research only** ([LICENSE_MODELS](https://github.com/apple/ml-mobileclip/blob/main/LICENSE_MODELS), [README § License](https://github.com/apple/ml-mobileclip#license); the code is MIT) |
| faces | `insightface/models/buffalo_l/*.onnx` from `buffalo_l.zip` (275 MB): SCRFD-10G detector + ArcFace R50 recognizer (+3 unused models in the pack) | [deepinsight/insightface model-zoo release](https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip) | **Non-commercial research only** ([InsightFace README § License](https://github.com/deepinsight/insightface#license): "The training data … and the models trained with these data are available for non-commercial research purposes only"; the code is MIT) |
| gestures | `mediapipe/gesture_recognizer.task` (8 MB) | [Google MediaPipe models](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task) | Apache-2.0 — MediaPipe ([LICENSE](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)); the bundle's hand models per the hand-tracking model card above. The [gesture classifier model card](https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf) states no separate license. |
| text | `ppocr/PP-OCRv6_medium_det/*` + `PP-OCRv6_medium_rec/*` (line detector + recognizer, 133 MB) | [PaddlePaddle PP-OCRv6 ONNX](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) + [recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) | Apache-2.0 ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)) |
| text | `ppocr/PP-LCNet_x1_0_{textline_ori,doc_ori}/*` (line/page orientation, 13 MB total) | [PaddlePaddle ONNX models](https://huggingface.co/PaddlePaddle) | Apache-2.0 (as above) |
| text | `ppocr/{cyrillic,eslav,arabic,devanagari,korean,th,el,te,ta}_PP-OCRv5_mobile_rec/*` (script recognizers, 74 MB total) | [PaddlePaddle PP-OCRv5 mobile ONNX](https://huggingface.co/PaddlePaddle) | Apache-2.0 (as above) |
| text (translation) | `translation/m2m100_418M/*` (1.9 GB: `pytorch_model.bin` fp32 + tokenizer files) — M2M100 418M | [facebook/m2m100_418M @ 55c2e61](https://huggingface.co/facebook/m2m100_418M/tree/55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636) | MIT ([model card](https://huggingface.co/facebook/m2m100_418M/blob/55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636/README.md) `license: mit`; [fairseq M2M-100](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100), MIT) |

The Android app reads text on-device with [Google ML Kit Text Recognition v2](https://developers.google.com/ml-kit/vision/text-recognition/v2)
(Latin model bundled in the APK; Chinese, Japanese, Korean and Devanagari models delivered by Google Play services) and
translates with ML Kit Translation (models downloaded on first use), under the
[ML Kit terms](https://developers.google.com/ml-kit/terms). For scripts ML Kit can't read it uses
[Tesseract4Android](https://github.com/adaptech-cz/Tesseract4Android) (Apache-2.0) on ML Kit's line crops.
Tesseract's 41 optional language packs are downloaded individually from [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
(Apache-2.0), pinned to commit `87416418657359cb625c412a48b6e1d6d41c29bd`; byte lengths and SHA-256 values are in
`android/app/src/main/java/pl/genor/glassvision/OcrLanguage.kt`. The selected pack is downloaded on first use and
remains available offline until deleted in Settings → Models on this phone.

Total ≈ 3.0 GB download, ≈ 3.1 GB on disk (including the default M2M100 418M weights).

### Optional translation model

`--translation-models 1.2B` downloads M2M100 1.2B (4.6 GB, MIT). OCR models for all supported scripts are included
in the default text install; there is no per-script download-on-demand path. The old `--ocr-scripts` argument is
accepted for compatibility but ignored.

| `--translation-models` | files | size | source | license |
|---|---|---|---|---|
| `1.2B` | `translation/m2m100_1.2B/*` | 4.6 GB | [facebook/m2m100_1.2B @ 7b36184](https://huggingface.co/facebook/m2m100_1.2B/tree/7b36184180524c1a1bbfa37f120a608046250b98) | MIT ([model card](https://huggingface.co/facebook/m2m100_1.2B/blob/7b36184180524c1a1bbfa37f120a608046250b98/README.md) `license: mit`) |

**Translation models** (RTX 3090, fp16): 418M = 0.9 GB VRAM, a 5-line sign ≈ 0.1 s uncached (one batch, 4 beams),
cached lines free; 1.2B = 2.4 GB VRAM, ≈ 2× slower, clearly better on one-word signs ("Ausgang" → "Exit" instead of
"output", "Kein Ausgang" → "No exit" instead of "No outcome"). Pick it in the app (Text → Translation model) after
downloading it. Source-language detection: [lingua-language-detector](https://github.com/pemistahl/lingua-py)
(Apache-2.0, a pip package; its language models ship inside the wheel). High-accuracy mode, loaded with the first
translation: ~1 GB RAM for the Latin script group (34 candidate languages), ~0.1 GB for Cyrillic.

## Notes

- **Ultralytics AGPL-3.0**: the YOLO weights *and* the `ultralytics` Python package are AGPL-3.0. Running a
  network service with them means offering the corresponding source to its users; this repository being open source
  covers the code side. Commercial use without AGPL obligations needs an Ultralytics Enterprise license.
- **Face recognition** stores face embeddings of the people you enroll (`<data>/faces/gallery.npz`) on your server.
  Biometric data is regulated in many places (e.g. GDPR art. 9): only enroll people who agreed to it.
- Where the files go: `GV_MODELS_DIR` (default `server/models`, Docker volume `/models`). The layout matches each
  pipeline's load code (`server/glass_vision/pipelines/*.py`).
- Download a subset: `scripts/download_models.py --only objects,gestures`; verify: `--check`; list: `--list`.
  A pipeline whose weights are missing reports an error in the app when you enable it; the others work.
- If an upstream file changes, the sha256 check refuses it (nothing half-verified is installed). Update the pinned
  hash in `scripts/download_models.py` only after checking the new file's provenance.
