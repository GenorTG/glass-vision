# glass-vision wire protocol (v3)

One WebSocket per client: `ws://<host>:8770/ws` (`<host>` = the address the server listens on, see README).
Auth is optional (see Auth below); bind the server to a private interface (LAN / VPN such as Tailscale), never the open internet.

v3 = **pipelines**: several recognizers (objects, open-vocabulary text prompt, faces, OCR, gestures, pose…)
that the client switches on and off and configures. The server *describes* every pipeline (options,
actions, skeleton) in `hello`, and the app builds its settings UI from that description, so a new
pipeline needs no app update.

## Client → server

**Binary frame** = 16-byte big-endian header + JPEG bytes (unchanged since v1):

| offset | type | field |
|---|---|---|
| 0 | uint32 | `seq` (monotonic per connection) |
| 4 | int64 | `client_ts` (ms, client clock) |
| 12 | uint32 | reserved, send 0 |
| 16 | bytes | JPEG, rotated upright by the client |

**Config** (text, any time; omitted fields keep their value; unknown pipeline ids/option keys → `error`):

```json
{"type": "config", "conf": 0.35, "imgsz": 640, "hold_ms": 300,
 "pipelines": {"objects": {"enabled": true},
               "open_vocab": {"enabled": true, "options": {"prompt": "red mug, keys, glasses"}},
               "text": {"enabled": false}}}
```

- `conf` 0.1..0.95 (default 0.35): threshold for pipelines with `uses_global_conf` (YOLO family); others filter internally.
- `imgsz` 320..1280, multiple of 32 (default 640): model input size for pipelines that take one.
- `hold_ms` 0..2000 (default 300): how long a missed track stays visible. Slow (`interval_ms` > 0) pipelines are
  held at least 2.5 × their interval, so their results don't blink between runs.
- `pipelines.<id>.enabled` (bool) and `pipelines.<id>.options.<key>` (typed as declared in `hello`).

**Command** (pipeline action, e.g. learn a face from the frame the user is looking at):

```json
{"type": "command", "req": 7, "pipeline": "faces", "action": "enroll", "args": {"name": "Alex"}}
```

`req` is a client-chosen integer echoed in the result. Actions run against the most recent frame the
server received on this connection (`ctx.frame`), so stream first, then send the command.

## Server → client

**hello** (on connect):

```json
{"type": "hello", "protocol": 3, "device": "cuda:0 (NVIDIA GeForce RTX 3090)", "version": "0.4.0.3",
 "pipelines": [
   {"id": "objects", "name": "Objects (80 COCO classes)", "description": "...",
    "default_enabled": true, "uses_global_conf": true, "interval_ms": 0,
    "options": [], "actions": [], "skeleton": null, "status": "ready"},
   {"id": "open_vocab", "name": "Find by text", "default_enabled": false, "uses_global_conf": true,
    "options": [{"key": "prompt", "label": "What to look for (comma-separated)", "type": "text", "default": "person, cup"}],
    "actions": [], "skeleton": null, "status": "not_loaded"},
   {"id": "faces", "name": "Faces", "options": [{"key": "threshold", "label": "Match threshold", "type": "number",
     "default": 0.4, "min": 0.2, "max": 0.8, "step": 0.05}],
    "actions": [{"id": "enroll", "label": "Learn face", "args": [{"key": "name", "label": "Name", "type": "text", "default": ""}]},
                {"id": "list", "label": "Known faces", "args": []}],
    "skeleton": null, "status": "not_loaded"},
   {"id": "gestures", "skeleton": [[0, 1], [1, 2], "..."], "...": "..."}
 ]}
```

Option `type`: `bool` | `text` | `number` (with `min`/`max`/`step`) | `choice` (with `choices: [str]`).
`status`: `not_loaded` | `loading` | `ready` | `error` (pipelines load lazily on first enable, unload when unused for 10 min).

**pipeline_status** (whenever a pipeline's status changes):

```json
{"type": "pipeline_status", "pipeline": "text", "status": "error", "message": "CUDA out of memory"}
```

**detections** (one per frame, as before):

```json
{"type": "detections", "seq": 42, "client_ts": 1792681655531, "w": 960, "h": 720,
 "decode_ms": 2.1, "server_ms": 14.0, "pipeline_ms": {"objects": 8.1, "faces": 4.2},
 "dets":   [{"pipeline": "objects", "label": "person", "conf": 0.91, "box": [0.12, 0.30, 0.48, 0.97]}],
 "tracks": [{"id": "objects:7", "pipeline": "objects", "label": "person", "conf": 0.88,
             "box": [0.12, 0.30, 0.48, 0.97], "seen_ms_ago": 0},
            {"id": "faces:2", "pipeline": "faces", "label": "Alex", "detail": "0.71", "conf": 0.93,
             "box": [0.20, 0.31, 0.33, 0.50], "seen_ms_ago": 0},
            {"id": "gestures:1", "pipeline": "gestures", "label": "Thumb_Up", "conf": 0.87,
             "box": [0.6, 0.4, 0.8, 0.7], "points": [[0.71, 0.66, 1.0], "..."], "seen_ms_ago": 0}]}
```

- `box` = `[x1, y1, x2, y2]` normalized 0..1 against the JPEG the client sent (already upright).
- `points` (optional) = `[[x, y, visibility], …]` normalized like `box`; join them with the pipeline's `skeleton` edges.
- `detail` (optional) = secondary text (match score, raw OCR text, …).
- `dets` = raw annotations of the pipelines that ran on *this* frame (flicker); `tracks` = what to draw:
  per pipeline ByteTrack ids (`"<pipeline>:<n>"`, stable while tracked), low-confidence rescue,
  2-frame confirmation, `hold_ms` hold at the Kalman-predicted position (`seen_ms_ago` > 0),
  label voted over recent frames, `conf` EMA.
- `client_ts` is echoed unchanged → round trip = `now - client_ts` on the client clock.
- Slow pipelines (`interval_ms` > 0, e.g. OCR) run in the background on the newest frame; their
  latest results are merged into every reply via their tracker, so they never delay the fast ones.

**dropped** / **error** (unchanged): every binary frame gets exactly one of `detections` / `dropped` / `error`.

```json
{"type": "dropped", "seq": 41}
{"type": "error", "seq": 43, "message": "cannot decode JPEG"}
```

**command_result**:

```json
{"type": "command_result", "req": 7, "pipeline": "faces", "action": "list", "ok": true,
 "message": "2 known faces",
 "items": [{"key": "Alex", "label": "Alex (3 samples)"}, {"key": "Sam", "label": "Sam (1 sample)"}],
 "item_action": {"id": "delete", "label": "Forget", "confirm": true}}
```

`items` / `item_action` are optional. If present, the client lists the items; picking one sends
`{"type":"command", "pipeline": …, "action": item_action.id, "args": {"key": item.key}}`.

`files` (optional, v3.1) = images/data the action produced on the server, e.g. the pose pipeline's `capture`
(OpenPose renders for ComfyUI ControlNet). The client downloads them over HTTP (same auth as the WebSocket) and
saves them to the phone gallery; they also stay on the server under `data/captures/<capture id>/`.

```json
{"type": "command_result", "req": 9, "pipeline": "pose", "action": "capture", "ok": true,
 "message": "Captured 2 people → 5 files",
 "files": [{"name": "photo.jpg", "kind": "photo", "url": "/captures/20261023-101500-123/photo.jpg"},
           {"name": "openpose_body.png", "kind": "openpose", "url": "/captures/20261023-101500-123/openpose_body.png"},
           {"name": "openpose_full.png", "kind": "openpose", "url": "/captures/20261023-101500-123/openpose_full.png"},
           {"name": "pose_keypoints.json", "kind": "json", "url": "/captures/20261023-101500-123/pose_keypoints.json"},
           {"name": "pose_on_photo.jpg", "kind": "overlay", "url": "/captures/20261023-101500-123/pose_on_photo.jpg"}]}
```

## Read text (v3.2)

Live OCR on every frame is slow and reads small or distant text badly. So the `text` pipeline's live result
*marks* text areas by default; the user then takes a high-resolution still and the server reads that one image
once, translating it into any number of languages. All of this is HTTP (same auth as the WebSocket) so a large
still never sits in the frame window.

- `hello` gains `"features": ["text_read"]` (absent on older servers → the client hides Read; `[]` on a server
  without the OCR pipeline).
- Text pipeline option `live` (choice): `"Mark text areas"` (detector only: every track `label` = `"text"`, `conf` 1,
  no `detail`; 25–40 % cheaper per run on a 3090) or `"Read live"` (the old behaviour: recognized text as the label).
  **Default `"Read live"`**: installed v3.1 apps send every option at the default `hello` advertises, so a Mark
  default would turn their live OCR into unlabeled boxes they have no Read button for. A client that offers Read
  sends `"live": "Mark text areas"` itself (e.g. while the user hasn't picked a mode). Changing `live` restarts the
  text pipeline's tracks (no "text" label voted onto read lines).

`POST /text/read` — body = the upright still as `image/jpeg` (any size up to 40 MB, else 413; an EXIF orientation
is applied, `w`/`h` are the upright size; the server reads it at up to 2560 px long side). Query (all optional):
`languages` = a value of the text pipeline's `languages` option (default: auto-detect among installed scripts),
`translate_to` = comma-separated language codes (`pl,de`), `translate_from` = code (default: detect per line),
`model` = a `translation_model` choice (default: the largest installed model — one read, unlike the live lane, can
afford it; the same default applies to `/text/translate`). Response:

```json
{"w": 4000, "h": 3000, "script": "latin", "ms": {"ocr": 310.2, "translate": 120.5},
 "lines": [{"text": "Ausgang", "conf": 0.93, "box": [0.12, 0.30, 0.41, 0.38], "block": 0, "lang": "de"},
           {"text": "Notausgang 2", "conf": 0.88, "box": [0.12, 0.40, 0.55, 0.47], "block": 0, "lang": "de"}],
 "translations": {"pl": ["Wyjście", "Wyjście awaryjne 2"]},
 "errors": {}}
```

- `lines` in reading order, block by block: blocks top to bottom, then left to right (blocks starting at about the
  same height are one row), and inside a block its lines top to bottom, then left to right — so a block's lines are
  contiguous and `block` ids count up from 0 in that order. `block` groups lines that belong together (same
  sign/paragraph: similar height, stacked ≤ ~1 line height apart or side by side on one row). `box` normalized like
  `detections`. `script` = the recognizer group that read it (`latin`, `cyrillic`, `zh_sim`…).
- `lang` = source language code (detected per line among the script's languages; a weak guess borrows its block's
  language, then the whole still's), the `translate_from` code if one was given, or null (digits/codes/URLs, or no
  confident guess). Only lines with a `lang` are translated.
- `translations.<code>` aligns with `lines`; an entry is null where the line is not translated (no `lang`, already in
  that language, model returned it unchanged). A target that failed entirely is absent from `translations` and
  listed in `errors.<code>` (message). No text found → `lines: []`, `translations.<code>: []`, still HTTP 200.
- `ms.ocr` = detection + recognition (auto mode tries every installed script on one detection), `ms.translate` =
  language detection + every target.
- 400 = not a JPEG / bad query value (checked before anything loads); 503 = the OCR model could not be loaded
  (message in `detail`).

`POST /text/translate` — add a language without reading again. Body
`{"lines": ["Ausgang", "Notausgang 2"], "from": ["de", "de"], "to": "fr", "model": null}`
(`from` optional, per line, null = detect among the languages of the lines' script) →
`{"to": "fr", "translations": ["Sortie", "Sortie de secours 2"]}` (same null rules; at most 500 lines). 400 = body
not JSON / bad field; 503 = OCR pipeline or translation model unavailable (`detail`).

`GET /text/languages` → `{"languages": [{"code": "pl", "name": "Polish"}, …]}`: targets `translate_to` accepts.

## Auth (optional, v3.1)

If the server is started with `GV_TOKEN=<secret>`, every request must carry it: WebSocket `ws://host:port/ws?token=<secret>`,
HTTP `?token=<secret>` or header `Authorization: Bearer <secret>`. Wrong/missing token → WebSocket closed with code 4401
(HTTP 401). `GET /health` stays open (no data beyond status). Without `GV_TOKEN` everything is open (trusted LAN/tailnet only).

## HTTP

- `GET /health` → `{"ok": true, "device": "...", "version": "..."}`
- `GET /app.apk` → the APK this server distributes; `GET /app/version` → its version + sha256 (in-app updater)
- `GET /captures/<id>/<file>` → files listed in a `command_result.files`
- `POST /text/read`, `POST /text/translate`, `GET /text/languages` → see Read text
- `GET /` → landing page with the APK link
