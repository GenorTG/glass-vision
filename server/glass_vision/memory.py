"""Give memory back after a model is dropped, and measure what this process holds.

Dropping a pipeline's references is not enough on its own: reference cycles keep tensors alive until the next gc
pass, torch's caching allocator keeps freed blocks reserved, and glibc keeps freed heap pages (a model loaded on the
CPU before moving to the GPU leaves gigabytes of them). `release()` handles all three. What it cannot free is the
per-process GPU state: the CUDA context, cuDNN/cuBLAS handles, ONNX Runtime's CUDA provider and MediaPipe's GL
context stay until the process exits (~1.5 GB VRAM measured on the 3090), which is why the server restarts itself
when everything is idle (`glass_vision/refresh.py`).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import os
import sys
from functools import cache

log = logging.getLogger(__name__)


@cache
def _libc() -> ctypes.CDLL | None:
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    libc = ctypes.CDLL(name)
    return libc if hasattr(libc, "malloc_trim") else None  # glibc only (not musl / macOS)


def _torch_cuda_initialized() -> bool:
    torch = sys.modules.get("torch")  # never import torch (and never create a CUDA context) just to free memory
    return torch is not None and torch.cuda.is_initialized()


def release() -> None:
    """Collect garbage, return torch's cached CUDA blocks, and hand free heap pages back to the OS."""
    gc.collect()
    if _torch_cuda_initialized():
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    libc = _libc()
    if libc is not None:
        libc.malloc_trim(0)


def rss_mib() -> float:
    """Resident memory of this process, MiB (Linux; 0 elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
    except OSError:
        return 0.0
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def cuda_reserved_mib() -> float:
    """VRAM held by torch's allocator (the CUDA context and other libraries come on top), MiB."""
    if not _torch_cuda_initialized():
        return 0.0
    import torch

    return torch.cuda.memory_reserved() / 2**20


def summary() -> str:
    return f"RSS {rss_mib():.0f} MiB, torch CUDA reserved {cuda_reserved_mib():.0f} MiB"
