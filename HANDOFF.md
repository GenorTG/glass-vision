# HANDOFF — glass-vision

Project status for the next contributor (human or agent). Host-specific notes: `HANDOFF.local.md` / `AGENTS.local.md` (gitignored).

## What exists (v0.6)
- **Server** (`server/`, FastAPI): pluggable recognition pipelines over one WebSocket + Read-text HTTP endpoints (`docs/PROTOCOL.md` v3.2).
  | id | model | per frame (RTX 3090) | notes |
  |---|---|---|---|
  | objects | YOLO11s (COCO) | 8 ms | default on |
  | open_vocab | YOLOE-11s + MobileCLIP | 11 ms | text prompt; 3 recent prompts cached (switch 11 ms, new prompt ~0.5 s) |
  | faces | InsightFace buffalo_l (ONNX Runtime) | 11 ms | gallery in `data/faces/`; actions enroll / list / delete |
  | text | EasyOCR (fp16 CRAFT detector), 12 script groups, auto-select + M2M100 418M / 1.2B | Mark 24–34 ms, Read live 33–60 ms | background lane every 400 ms; option `live` = Mark text areas / Read live |
  | gestures | MediaPipe GestureRecognizer (GPU delegate, CPU fallback) | 5 ms | 7 gestures + hand skeleton |
  | pose | YOLO11s-pose + rule-based names | 9 ms | "Standing · Arms up"…; action `capture` → OpenPose body/full PNG + JSON + photo + overlay for ComfyUI ControlNet |
  - **Read text** (`text_read.py`): `POST /text/read` (JPEG still, read at ≤2560 px, reading order + `block` grouping, per-line language, N translation targets with the largest installed model), `POST /text/translate`, `GET /text/languages`; `hello.features = ["text_read"]`. Warm read of a sign ~0.3–0.6 s; first call after a load ~10 s.
  - Per-connection tracking (ByteTrack + hold + label vote), lazy load / idle unload, optional token auth (`GV_TOKEN`), config via `server/config.env`.
- **App** (`android/`): CameraX front/back (preview + analysis + 4000×3000 still capture), Material3 camera-style UI, recognition UI built from `hello`, box glide, photo button, server files → gallery, first-run connect, in-app self-update.
  - **Read text**: with Text (OCR) on, live view marks text areas (thin boxes, no chips); the "Read text" pill above the shutter takes a high-res still → results sheet: thumbnail with line boxes, original grouped by sign (copy/share), one section per target language, "Add language" translates the already-read lines. Targets remembered (first run: phone language).
  - **On this phone** backend: objects, gestures, pose, face detection, OCR. Phone OCR = ML Kit Text Recognition v2 (Latin bundled, CJK/Devanagari via Play services); scripts ML Kit lacks use the 41 SHA-256-verified Tesseract packs on ML Kit's line crops (`TextLayout.kt` `OcrPlan`). `LocalTextReader` does Read text on-device with ML Kit translation (models download on first use).
  - `scripts/emulator/` = Docker emulator harness (see AGENTS.md gotcha for reaching a server from it).
- **Install**: `scripts/install.sh` (venv, CUDA wheel by driver, models, systemd unit); Docker (`docker-compose.yml`, GPU + `cpu` profile).
- **CI/CD**: `ci.yml` (server tests without GPU, Android build/unit/lint/Paparazzi); `release.yml` (tag → signed APK, ClamAV, VirusTotal, GitHub release).

## Verification state (2026-10-23, v0.6.0.12+)
- Server: 260 tests with a GPU, ruff clean. Live service on :8770 runs this code; `/text/read` on bus.jpg verified.
- App: 221 JVM tests incl. Paparazzi, lint 0 errors. Instrumented on-device OCR tests: 17/17 in the emulator (rendered signs in 12 scripts + bus.jpg).
- Back-compat: the previously installed app (c98fb97) parses the v3.2 server's fixtures (15/15); old apps keep "Read live".
- **Emulator end-to-end, both backends**: live marks → Read text → sheet with German detected; server: 1.2B "Exit / Emergency Exit 2 / Departure at 12.45"; phone: ML Kit "Exit / Emergency exit 2 / Track 3 / Departure 12:45"; Add language (Czech) works. First phone read waits ~25 s for the translation-model download.
- Fixed today: server auto-script mode went permanently blank after one textless frame (0.6.0.9); short misread lines got a wrong language ("cerỌ" → vi).
- **Not yet on a physical phone**: Read text, still-capture resolution, front-camera stills, ML Kit CJK/Devanagari recognizers (emulator Play services never downloads them), copy/share sheets, photo button, token flow.

## Next / open
1. Install the new APK (in-app update from the server) and device-test Read text on real signs, both backends.
2. APK grew 59 → 75 MB (bundled ML Kit Latin, 3 ABIs); drop x86_64 from release builds (−4.9 MB) if size matters.
3. Box alignment assumes preview and analysis are both full-frame 4:3 (FIT_CENTER); if offset on some phone, bind with a CameraX `UseCaseGroup` + `previewView.viewPort`.
4. Gestures `num_hands`/`min_conf` are creation-time options: clients with different values would rebuild the recognizer each frame.
5. All pipelines together raise objects/pose latency from ~9 to ~30 ms (GPU + GIL contention).
6. Ideas: smart-glasses client on the same protocol; chess board → move pipeline; face keypoints in the OpenPose export.
