# HANDOFF — glass-vision

Status for the next contributor. Host-specific notes: `HANDOFF.local.md` / `AGENTS.local.md` (gitignored).

## Current state

- **Android app** (`android/`): CameraX preview/analysis/stills, Material3 UI, server and on-device recognition, text reading.
- **Server** (`server/`): FastAPI pipelines over WebSocket plus still-read HTTP endpoints; YOLO objects/open-vocab/pose, InsightFace, MediaPipe gestures, PP-OCR text, and M2M100 translation. Lazy loading, optional token auth (`GV_TOKEN`), config at `server/config.env`.
- **Read text server**: PP-OCR ONNX detector/recognizers with script-specific fallback, EXIF-aware image input, orientation correction, full-resolution line crops, suspect-text detection and v3.3 block responses. `format=lines` remains available to old clients.
- **Read text app**: capture rotation, focus/metering, hold-still and blur gate with one retake, frozen preview, camera JPEG + EXIF upload. Results support blocks, per-block translation, suspect hints, data detectors, selectable cards and a translated-photo overlay with “Show original”.
- **Idle memory** (2026-10-27): every load/unload runs `memory.release()` (gc, CUDA cache, `malloc_trim`); once nothing is loaded and no client is connected the server re-`exec`s itself in place (`refresh.py`, same PID, listening socket inherited, no preload). Measured on the 3090 with all 6 pipelines + 1.2B translator: peak 5.1 GB VRAM / 4.4 GB RSS (was 6.6 GB RSS), idle **0 MiB VRAM / 0.8 GB RSS** (was stuck at 1.56 GB / 6.6 GB). Off: `GV_IDLE_RESTART=false`.
- **Protocol**: `docs/PROTOCOL.md`; protocol v3.3 is backward-compatible with existing line-response clients.
- **Research/evaluation**: OCR engine evaluation at `docs/research/ocr-eval/README.md`; translation benchmark and methodology at `docs/research/translation-bench.md` and `docs/research/translation-bench/`.

## Verification (2026-10-24)

- Combined server tree: **292 passed**, 1 upstream PyTorch deprecation warning; Ruff check clean; 47 Python files formatted.
- Android: `assembleDebug testDebugUnitTest lintDebug verifyPaparazziDebug` passed (257 JVM tests, 0 lint errors); `connectedDebugAndroidTest` passed (18 tests).
- OCR eval: PP-OCR stack achieved **55/67 key phrases (82%)** vs EasyOCR **28/67 (42%)**; rendered exact lines 42/42; rotated/skewed signs 5/5; 1.00 boxes/GT line vs 1.52. Separate earlier 10-photo research bench has different metrics; see the eval report.
- Translation benchmark: best automatic score was Index-Translate-9B Q4_K_M (COMET .933, chrF++ 76.5, key-term 94%); human blind review and production integration have not happened.
- Android snapshots for cards and translated overlay were visually reviewed; emulator tests passed.
- **Not verified on a physical phone**: sideways capture behavior, real autofocus/blur retake, torch, front-camera rotation, or textured-photo overlay quality.

## Remaining work

1. Implement and test the v3.3 `/text/careful` endpoint with an optional user-triggered heavy vision model; the client displays suspect blocks, but “Read carefully” is not wired/enabled until the server advertises `text_careful`.
2. Human blind review and service-level A/B/integration for the translation candidates before changing the production translator (M2M100 remains current).
3. On-device OCR improvements: PP-OCR mobile packs, full-resolution region reads and Tesseract long-tail cleanup.
4. Add ML Kit document-scanning mode and live-result stability.
5. Test capture/overlay behavior on a physical phone; obtain phone-shot examples for the evaluation set.
6. First public release `v0.6.0.16` published 2026-10-25 (<https://github.com/GenorTG/glass-vision/releases/tag/v0.6.0.16>): release-key signed, ClamAV clean, **VirusTotal skipped** — `VT_API_KEY` secret not set yet; the next tag after Genor adds it gets both scans. Server not redeployed; ask Genor before deploys.

## Repo conventions

- Follow `AGENTS.md`; keep project state here and update this file as work moves.
- Release workflow: CI checks server tests and Android build/unit/lint/Paparazzi; `release.yml` creates signed APKs from tags.
