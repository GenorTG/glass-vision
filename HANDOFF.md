# HANDOFF — glass-vision

Project status for the next contributor (human or agent). Host-specific notes: `HANDOFF.local.md` / `AGENTS.local.md` (gitignored).

## What exists (v0.6)
- **Server** (`server/`, FastAPI): pluggable recognition pipelines over one WebSocket (`docs/PROTOCOL.md` v3.1).
  | id | model | per frame (RTX 3090) | notes |
  |---|---|---|---|
  | objects | YOLO11s (COCO) | 8 ms | default on |
  | open_vocab | YOLOE-11s + MobileCLIP | 11 ms | text prompt; 3 recent prompts cached (switch 11 ms, new prompt ~0.5 s) |
  | faces | InsightFace buffalo_l (ONNX Runtime) | 11 ms | gallery in `data/faces/`; actions enroll / list / delete |
  | text | EasyOCR en / en+pl | 55 ms | background lane every 400 ms |
  | gestures | MediaPipe GestureRecognizer (GPU delegate, CPU fallback) | 5 ms | 7 gestures + hand skeleton |
  | pose | YOLO11s-pose + rule-based names | 9 ms | "Standing · Arms up"…; action `capture` → OpenPose body/full PNG + JSON + photo + overlay for ComfyUI ControlNet |
  - Per-connection tracking (ByteTrack + hold + label vote), lazy load / idle unload, optional token auth (`GV_TOKEN`), config via `server/config.env`.
  - All six at once at 10 fps: 0 unanswered frames, RTT median ~48 ms, ~4.2 GB VRAM. CPU-only works (slow).
- **App** (`android/`): CameraX front/back, Material3 camera-style UI (mode chips, action pills, sheets), recognition UI built from the server's `hello`, box glide, photo button (view + boxes → gallery), server files (pose captures) → gallery, first-run "connect to your server" (address + optional token), in-app self-update from the user's own server.
- **Install**: `scripts/install.sh` (self-contained venv, CUDA wheel by driver, models from original sources, listen address/token prompts, optional systemd unit) — tested from a fresh copy; Docker (`docker-compose.yml`, GPU + `cpu` profile) — tested on GPU and CPU.
- **CI/CD**: `ci.yml` (server tests without GPU, Android build/unit/lint/Paparazzi); `release.yml` (tag → signed APK, ClamAV, VirusTotal, SHA256SUMS, GitHub release).

## Verification state
- Server: 160 tests with a GPU (132 + 28 skipped without), ruff clean. Live e2e: all pipelines, face enroll/recognize/forget, pose capture + file download.
- App: 101 JVM tests incl. Paparazzi snapshots (`android/app/src/test/snapshots/images/`), lint 0 errors.
- Confirmed on a real phone: streaming, tracking, in-app update path. **Not yet on a device**: photo button, capture files → gallery, first-run connect sheet, token flow.

## Next / open
1. Device-test the v0.6 app features above.
2. Box alignment assumes preview and analysis are both full-frame 4:3 (FIT_CENTER); if offset on some phone, bind with a CameraX `UseCaseGroup` + `previewView.viewPort`.
3. Gestures `num_hands`/`min_conf` are creation-time options: clients with different values would rebuild the recognizer each frame.
4. All pipelines together raise objects/pose latency from ~9 to ~30 ms (GPU + GIL contention).
5. Ideas: smart-glasses client on the same protocol; chess board → move pipeline; face keypoints in the OpenPose export.
