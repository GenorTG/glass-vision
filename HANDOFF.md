# HANDOFF — glass-vision

Project status for the next contributor (human or agent). Host-specific notes: `HANDOFF.local.md` / `AGENTS.local.md` (gitignored).

## What exists (v0.6)
- **Server** (`server/`, FastAPI): pluggable recognition pipelines over one WebSocket (`docs/PROTOCOL.md` v3.1).
  | id | model | per frame (RTX 3090) | notes |
  |---|---|---|---|
  | objects | YOLO11s (COCO) | 8 ms | default on |
  | open_vocab | YOLOE-11s + MobileCLIP | 11 ms | text prompt; 3 recent prompts cached (switch 11 ms, new prompt ~0.5 s) |
  | faces | InsightFace buffalo_l (ONNX Runtime) | 11 ms | gallery in `data/faces/`; actions enroll / list / delete |
  | text | EasyOCR; 12 script groups, auto-select among installed recognizers + M2M100 418M / optional 1.2B | 55 ms | background lane every 400 ms; translations cached |
  | gestures | MediaPipe GestureRecognizer (GPU delegate, CPU fallback) | 5 ms | 7 gestures + hand skeleton |
  | pose | YOLO11s-pose + rule-based names | 9 ms | "Standing · Arms up"…; action `capture` → OpenPose body/full PNG + JSON + photo + overlay for ComfyUI ControlNet |
  - Per-connection tracking (ByteTrack + hold + label vote), lazy load / idle unload, optional token auth (`GV_TOKEN`), config via `server/config.env`.
  - All six at once at 10 fps: 0 unanswered frames, RTT median ~48 ms, ~4.2 GB VRAM. CPU-only works (slow).
- **App** (`android/`): CameraX front/back, Material3 camera-style UI (mode chips, action pills, sheets), recognition UI built from the server's `hello`, box glide, photo button (view + boxes → gallery), server files (pose captures) → gallery, first-run connect, in-app self-update. Optional **On this phone** backend runs objects, gestures, pose, face detection and OCR; MediaPipe models download on first use. Phone OCR uses Tesseract with 41 downloadable, SHA-256-verified language packs; selected packs work offline. OCR has an Auto-detect choice: it bootstraps with English, uses ML Kit language identification on recognized text, then downloads and switches to the matching pack for later frames. ML Kit translation downloads target-language models on demand. Find-by-text and face naming remain server-only. `scripts/emulator/` provides the Docker Android emulator harness.
- **Install**: `scripts/install.sh` (self-contained venv, CUDA wheel by driver, models from original sources, listen address/token prompts, optional systemd unit) — tested from a fresh copy; Docker (`docker-compose.yml`, GPU + `cpu` profile) — tested on GPU and CPU.
- **CI/CD**: `ci.yml` (server tests without GPU, Android build/unit/lint/Paparazzi); `release.yml` (tag → signed APK, ClamAV, VirusTotal, SHA256SUMS, GitHub release).

## Verification state
- Server: 214 tests with a GPU (168 + 46 skipped without), ruff clean. Live e2e: all pipelines, multilingual OCR/translation, face enroll/recognize/forget, pose capture + file download.
- App: 101 JVM tests incl. Paparazzi snapshots (`android/app/src/test/snapshots/images/`), lint 0 errors.
- On-device mode and local translation: debug APK builds; Android lint clean. Emulator harness itself was verified, but this feature build has not yet been exercised in the emulator or on a phone.
- OCR: server offers auto-detection across installed script models; phone pack-based OCR and language auto-detection have not yet been exercised on a physical phone. Phone auto-detection depends on readable text from the initial English OCR pass, so other scripts may need an explicit language selection.
- Confirmed on a real phone: streaming, tracking, in-app update path. **Not yet on a device**: photo button, capture files → gallery, first-run connect sheet, token flow.

## Next / open
1. Device-test the v0.6 app features above.
2. Box alignment assumes preview and analysis are both full-frame 4:3 (FIT_CENTER); if offset on some phone, bind with a CameraX `UseCaseGroup` + `previewView.viewPort`.
3. Gestures `num_hands`/`min_conf` are creation-time options: clients with different values would rebuild the recognizer each frame.
4. All pipelines together raise objects/pose latency from ~9 to ~30 ms (GPU + GIL contention).
5. Ideas: smart-glasses client on the same protocol; chess board → move pipeline; face keypoints in the OpenPose export.
