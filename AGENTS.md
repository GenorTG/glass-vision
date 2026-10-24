# AGENTS.md — glass-vision

Phone camera → WebSocket → recognition pipelines on your own GPU server → boxes, labels and keypoints drawn on the phone.
Wire protocol: `docs/PROTOCOL.md` (v3.3; change it there first, then both sides). User docs: `README.md`, models: `docs/MODELS.md`.
Host-specific notes (your own server, release key location, …) go in the gitignored `AGENTS.local.md` — read it if present.

## Layout
- `server/` — Python 3.11–3.12, FastAPI. venv `server/.venv` (`scripts/install.sh`), weights in `server/models/`, user data (face gallery, captures) in `server/data/` (both gitignored; overridable via `GV_MODELS_DIR` / `GV_DATA_DIR`). Settings: `server/config.env` + env vars (`glass_vision/config.py`).
  - `glass_vision/pipelines/` — one plug-in per recognizer (`base.py` = the contract; registry in `__init__.py`): objects (YOLO11s), open_vocab (YOLOE text prompt), faces (InsightFace buffalo_l + gallery), text (PP-OCRv6/v5 ONNX), gestures (MediaPipe), pose (YOLO11s-pose + `pose_names.py` + OpenPose capture `openpose.py`).
  - `glass_vision/session.py` — PipelineManager (lazy load, idle unload, per-model lock, status broadcast) + StreamSession (per socket: fast pipelines inline in parallel, slow ones in background lanes, one TrackSmoother each).
  - `glass_vision/tracking.py` — ByteTrack + hold + label vote, pipeline-agnostic.
- `android/` — Kotlin, plain Views + Material3, CameraX + OkHttp. AGP 8.2.2 / Kotlin 1.9.22 / Gradle 8.5 / SDK 34 / JDK 17 (`local.properties` → `sdk.dir`). Builds its recognition UI from `hello` (no app change for new pipelines).
- `scripts/` — `install.sh` (venv), `update.sh`, `download_models.py` (original sources, sha256), `fetch_release_apk.py`, `virustotal_scan.py`, `git-hooks/`.
- `Dockerfile`, `docker-compose.yml`, `deploy/` — container install and the systemd --user unit template.
- `.github/workflows/` — `ci.yml` (tests on push/PR), `release.yml` (tag `v*` → signed APK, ClamAV + VirusTotal, GitHub release).

## Commands
- **Always run server Python through `server/run`** (loads the venv, CUDA/cuDNN library paths, `YOLO_AUTOINSTALL=false`).
- Server tests (real models; GPU tests auto-skip without CUDA): `cd server && ./run -m pytest -q tests && ruff check . --exclude .venv && ruff format --check --exclude .venv .`
- End-to-end over the network: `cd server && ./run tools/stream_client.py --url ws://<host>:<port>/ws [--token …] --pipelines objects,pose,faces,gestures,text,open_vocab --prompt "bus, person" .venv/lib/python3.*/site-packages/ultralytics/assets/{bus,zidane}.jpg`
- APK: `cd android && ./gradlew assembleDebug testDebugUnitTest lintDebug verifyPaparazziDebug` (UI snapshots in `app/src/test/snapshots/images/`; after an intended UI change `./gradlew recordPaparazziDebug` and look at the changed PNGs). Release APK: `GV_KEYSTORE_FILE=… GV_KEYSTORE_PASSWORD=… GV_KEY_ALIAS=… GV_KEY_PASSWORD=… ./gradlew assembleRelease` (refuses to build unsigned).

## Adding a pipeline
Subclass `Pipeline` (`server/glass_vision/pipelines/base.py` docstring), add it to `ALL_PIPELINES`, add its model URLs to `scripts/download_models.py` + `docs/MODELS.md`, add `tests/test_pipeline_<id>.py` with the real model on a real image (`@pytest.mark.gpu` if it needs CUDA). Fast (`interval_ms=0`) must stay ≲20 ms/frame on a GPU; anything slower gets `interval_ms` and runs in the background. Options/actions in `PipelineInfo` appear in the app automatically; actions may return `files` (see PROTOCOL.md).

## Versioning + releases
- Repo-root `VERSION` = `MAJOR.MINOR.PATCH.BUILD`; `scripts/git-hooks/pre-commit` bumps BUILD on every commit (`git config core.hooksPath scripts/git-hooks` once per clone; skip with `GV_NO_BUMP=1`). Gradle derives versionName/versionCode from it. Build the APK AFTER committing, or it carries the previous BUILD number.
- Release = tag `v<VERSION>` on the commit whose VERSION says exactly that → `release.yml`.
- The server distributes an APK for in-app updates (`GV_APK_PATH`, default `<data>/app/glass-vision.apk` + sidecar `.json`, falling back to the local debug build output). Android only installs an update signed with the SAME key as the installed app: release builds (GitHub) and local debug builds can't update each other.

## Gotchas
- **Never break installed apps with a server change.** Phones can only self-update through a working app. The server answers v2-shaped results (int track ids) until a client sends a config with `pipelines` or `protocol` >= 3. Before shipping a protocol change, run the previous release's `Protocol.kt` (git worktree at its tag) against output captured from the new server, and re-capture `android/app/src/test/resources/*_v3.json` with `server/tools/capture_fixtures.py` (`ServerFixtureTest`).
- Every frame must get exactly one reply (`detections`/`dropped`/`error`): the client's in-flight window depends on it.
- **Never let anything pip-install into the venv implicitly.** ultralytics' AutoUpdate (`YOLO_AUTOINSTALL`) once installed `clip` from git and with it a different torch + torchvision, breaking torchvision (`operator torchvision::nms does not exist`). `run`, `__main__` and the test conftest set `YOLO_AUTOINSTALL=false`; `install.sh` installs the `--no-deps` packages (ultralytics, mediapipe, clip) and their real deps explicitly.
- insightface depends on CPU `onnxruntime`, which overwrites `onnxruntime-gpu`'s files (same package dir) → CUDA EP silently gone. `install.sh` reinstalls onnxruntime-gpu after it.
- onnxruntime-gpu dlopens unversioned `libcudnn*.so`; the pip cuDNN wheel only ships `.so.9`. `install.sh` puts `libcudnn*.so → …so.9` symlinks in `server/.venv/ort-libs/`, which `server/run` adds to LD_LIBRARY_PATH.
- mediapipe is installed `--no-deps`: its dependency opencv-contrib-python would shadow opencv-python's `cv2`.
- Multi-GPU hosts: expose ONE modern GPU to the server (`CUDA_VISIBLE_DEVICES`, PCI_BUS_ID order). With an old GPU (< SM 7.5) visible, torch may disable cuDNN process-wide; expose one modern GPU only.
- MediaPipe GPU delegate (gestures) prints `tensor.cc] Tensors are designed for single writes ... undefined behavior` once at graph build: internal and harmless (CPU vs GPU results match). Its EGL context picks a GPU itself (not `CUDA_VISIBLE_DEVICES`); check the `renderer:` log line. Tests run both delegates.
- YOLOE: give it the MobileCLIP text encoder explicitly (`open_vocab.py`); otherwise ultralytics loads/downloads `mobileclip_blt.ts` (570 MB) from the **cwd** on every prompt. `set_classes` drops the predictor (next predict re-fuses, ~250 ms) → one cached model per recent prompt.
- Ultralytics `Model.predict(device=…)` doesn't move `model.model`; call `.to(device)` before anything that uses the module directly (`get_text_pe`).
- Slow lanes (OCR) are held for 2.5 × max(interval, measured run time); on a CPU an OCR run takes seconds and a hold based on the interval alone would never show a box.
- Front camera: PreviewView mirrors the preview, ImageAnalysis frames are not mirrored → the overlay mirrors x (`BoxMapper`); the server and gesture handedness need nothing.
- OpenPose export (`openpose.py`) copies comfyui_controlnet_aux's renderer and JSON format (normalized coords, confidence 1.0, missing hand = `[]`); don't "fix" it towards pixel coordinates — ComfyUI wouldn't load it back.
- OCR translation (`translate.py`): M2M100 418M mistranslates one-word signs ("Ausgang" → "output"); the 1.2B option fixes most. ALL-CAPS lines are sentence-cased before translating ("ВХОД ВОСПРЕЩЁН" otherwise became "Entrance allowed"); digit-heavy lines are never translated. Language detection on single short words is unreliable → short lines inherit the sign's language; users can pin `translate_from`.
- Phone OCR = ML Kit Text Recognition v2 (Latin bundled; Chinese/Japanese/Korean/Devanagari via Play services) finding and reading lines; Tesseract4Android only reads ML Kit's line crops for scripts ML Kit lacks, with a SHA-256-verified pack in app-private `tessdata/` (URLs/checksums pinned in `android/.../OcrLanguage.kt`; plan per choice: `OcrPlan` in `TextLayout.kt`). Phone translation runs on OCR's slow lane, asks ML Kit to download source/target models on demand, caches results, and must fall back to the raw OCR line on any language-ID/model error. Server OCR uses PP-OCRv6 line detection/recognition and lazily tries PP-OCRv5 script recognizers only on uncertain/out-of-alphabet lines; all OCR weights ship with the default text model install (no runtime surprise downloads).
- Tesseract on whole camera frames (`PSM_AUTO`) reads almost nothing (document engine, not scene text); test any phone OCR change on a real photo (`ultralytics/assets/bus.jpg`), never only on rendered white-background text.
- ML Kit Latin boxes Cyrillic/Greek/Tamil lines (garbage text, conf 0.3–0.55) but finds no Arabic/Hebrew/Thai/Bengali line at all (those packs also scan the still: `PSM_SPARSE_TEXT`, then `PSM_AUTO`, which reads a single short Arabic word sparse misses). A Cyrillic pack reads Latin "EXIT" as "ЕХІТ" at 0.94, above ML Kit's 0.83: a pack reading only wins in its own Unicode script (`TextLanguages.choose`). ML Kit also makes each line of a sign its own block (`TextLayout.mergeBlocks`). Tess4Android's `getResultIterator()` is null until recognition ran: call `getUTF8Text()` first. The emulator's Play services (23.18, no Play Store) never get the unbundled CJK modules, so instrumented tests exercise the Tesseract fallback.
- Kotlin: `/*` inside a KDoc comment opens a nested comment (e.g. a glob like `resources/*.json`) → "Unclosed comment".
- org.json on the JVM test classpath doesn't preserve key order (Android's does): assert JSON maps as sets in unit tests.
- Agent runs that open many snapshot PNGs can die on output size (`CLI JSONL output exceeded …`). Look at a handful of changed PNGs per run, not all of them.
- `pkill -f "python -m glass_vision"` from a shell also kills that shell (its own command line matches); stop the service with `systemctl --user stop glass-vision`.
- Text `live` option: the server defaults to "Read live" because installed apps echo every `hello` default back; the app (`Protocol.markTextByDefault`) turns that default into "Mark text areas" when `hello.features` has `text_read`. Never flip the server default.
- Read text uses the largest installed translation model (`text_read.best_installed_model`), not the live `translation_model`. `format=blocks` keeps v3.2 clients unchanged; the app opts in only when `hello.features` includes `text_blocks`.
- Server text models are official PaddleOCR ONNX exports, evaluated with ONNX Runtime CPU parity tests and executed by `pipelines/onnx_torch.py` on torch/cuDNN because ONNX Runtime GPU shape-planning was seconds per new OCR shape on this host. Keep parity coverage when changing model exports or interpreter ops.
- Emulator harness: the container cannot reach the host (tailnet IP and docker bridge both "No route to host"). For an app↔server test run a server on `127.0.0.1:<port>` and `adb reverse tcp:<port> tcp:<port>`; the app connects to `127.0.0.1:<port>`. `camera.sh poster IMG wall` + `look-at-poster` puts a sign in front of the camera.
