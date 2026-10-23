# glass-vision server image. GPU (default) or CPU:
#   docker compose up -d                                  # NVIDIA GPU (see docker-compose.yml, README "Docker")
#   docker compose --profile cpu up -d glass-vision-cpu   # no GPU
#
# Base: the NVIDIA CUDA *base* image (driver plumbing only), not cudnn-runtime: the PyTorch cu130 wheels already
# bundle the CUDA 13 runtime, cuBLAS and cuDNN as pip packages, and onnxruntime-gpu uses those same libraries via
# /opt/venv/ort-libs. A cudnn-runtime base would add ~3 GB of duplicate libraries. Swap it with --build-arg BASE_IMAGE.
ARG TORCH_VARIANT=cu130
ARG BASE_IMAGE=nvidia/cuda:13.0.2-base-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_VARIANT
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1
# python3 = 3.12 on Ubuntu 24.04. libgl1/libglib2.0-0: opencv-python; libegl1/libgles2: MediaPipe's GPU delegate
# (gestures; falls back to CPU without them).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv libgl1 libglib2.0-0 libegl1 libgles2 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Same install order as scripts/install.sh (AGENTS.md "Gotchas"): torch from the PyTorch index first, pinned deps
# constrained to that torch, --no-deps packages, onnxruntime last.
# One layer per step, so a network hiccup on a later step doesn't re-download PyTorch.
WORKDIR /app/server
ENV PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=60
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" torch==2.12.0 torchvision==0.27.0 \
 && printf 'torch==2.12.0\ntorchvision==0.27.0\n' > /opt/venv/torch-constraints.txt
COPY server/requirements.txt ./
RUN /opt/venv/bin/pip install -r requirements.txt -c /opt/venv/torch-constraints.txt
COPY server/requirements-*.txt ./
RUN /opt/venv/bin/pip install --no-deps -r requirements-nodeps.txt \
 && if [ "$TORCH_VARIANT" = cpu ]; then ort=requirements-cpu.txt; \
    elif [ "$TORCH_VARIANT" = cu126 ]; then ort=requirements-cu12.txt; else ort=requirements-cu13.txt; fi \
 && /opt/venv/bin/pip install --no-deps -r "$ort" \
 && mkdir -p /opt/venv/ort-libs \
 && for lib in /opt/venv/lib/python3*/site-packages/nvidia/*/lib/lib*.so*; do \
      [ ! -e "$lib" ] || ln -sf "$lib" "/opt/venv/ort-libs/$(basename "$lib")"; done \
 && for lib in /opt/venv/lib/python3*/site-packages/nvidia/cudnn/lib/libcudnn*.so.[0-9]*; do \
      [ ! -e "$lib" ] || { base="$(basename "$lib")"; ln -sf "$lib" "/opt/venv/ort-libs/${base%%.so.*}.so"; }; done \
 && /opt/venv/bin/python -c "import clip, easyocr, insightface, mediapipe, onnxruntime, torch, ultralytics; from torchvision.ops import nms; nms(torch.zeros((1, 4)), torch.zeros(1), 0.5)"

COPY VERSION /app/VERSION
COPY scripts/download_models.py scripts/fetch_release_apk.py /app/scripts/
COPY deploy/docker-entrypoint.sh /app/docker-entrypoint.sh
COPY server/glass_vision /app/server/glass_vision
COPY server/tools /app/server/tools

RUN useradd --create-home --uid 10001 glassvision \
 && mkdir -p /models /data \
 && chown glassvision:glassvision /models /data

ENV PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/venv/ort-libs \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    GV_ENV_FILE= \
    GV_HOST=0.0.0.0 \
    GV_MODELS_DIR=/models \
    GV_DATA_DIR=/data
USER glassvision
VOLUME ["/models", "/data"]
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD python3 -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"GV_PORT\", \"8770\")}/health', timeout=4)"
ENTRYPOINT ["/app/docker-entrypoint.sh"]
