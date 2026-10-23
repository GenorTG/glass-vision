#!/bin/sh
# Docker entrypoint: fetch missing models into /models on first start (resumes; ~3 GB with the translation model), then run the server.
# GV_SKIP_MODEL_DOWNLOAD=1 skips the download (pipelines without their weights then report an error in the app).
set -eu
cd /app/server
if [ "${GV_SKIP_MODEL_DOWNLOAD:-0}" != 1 ]; then
  python3 /app/scripts/download_models.py --models-dir "${GV_MODELS_DIR:-/models}" \
    || echo "warning: some models failed to download; restart the container to retry" >&2
fi
exec python3 -m glass_vision "$@"
