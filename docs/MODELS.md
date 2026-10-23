# Models and their licenses

The server downloads every model from its **original upstream source** (`scripts/download_models.py`; the
installer and the Docker entrypoint run it). Nothing is re-hosted by this project. Each file is checked against a
pinned sha256, recorded on 2026-10-22 from files downloaded from these same URLs; EasyOCR also publishes md5 sums of
its weights, which are checked too.

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
| text | `easyocr/craft_mlt_25k.pth` (CRAFT detector, 79 MB) | [JaidedAI/EasyOCR pre-v1.1.6](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip) | Apache-2.0 ([EasyOCR LICENSE](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)) |
| text | `easyocr/english_g2.pth`, `easyocr/latin_g2.pth` (14 MB each) | [JaidedAI/EasyOCR v1.3](https://github.com/JaidedAI/EasyOCR/releases/tag/v1.3) | Apache-2.0 (as above) |

Total ≈ 1.03 GB download, ≈ 1.1 GB on disk.

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
