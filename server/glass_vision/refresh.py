"""Restart the server process in place once it is completely idle, to return the memory an unload cannot.

After every pipeline is unloaded the process still holds the CUDA context, cuDNN/cuBLAS handles, ONNX Runtime's
CUDA provider, MediaPipe's GL context and a grown heap: ~1.5 GB VRAM and several GB RSS measured on the 3090 after
a session that used text reading. Only an exit frees those. So when nothing is loaded, no client is connected and
no request is running, the server `exec`s itself:

- same PID (systemd / docker see no restart), same arguments and environment;
- the listening socket is inherited (`GV_LISTEN_FD`): connections that arrive meanwhile wait in the kernel backlog
  and are answered by the new process, none is refused;
- the new process neither probes the GPU nor preloads default pipelines (`GV_REFRESHED_DEVICE`), so it starts at
  zero VRAM and loads models again on the next client's first use, as after any idle unload.

Only `python -m glass_vision` (which owns the listening socket) enables this; an app served another way (tests,
`uvicorn glass_vision.app:app`) never restarts itself.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

LISTEN_FD_ENV = "GV_LISTEN_FD"
DEVICE_ENV = "GV_REFRESHED_DEVICE"  # set: this process is a refresh; value = the device the first one resolved
DEVICE_LABEL_ENV = "GV_REFRESHED_DEVICE_LABEL"
QUIET_S = 30.0  # no request / connection for this long before a refresh (shorter if GV_IDLE_UNLOAD_S is)


@dataclass(frozen=True)
class Refreshed:
    device: str
    device_label: str


_listen_socket: socket.socket | None = None


def adopt_or_bind(host: str, port: int) -> socket.socket:
    """The listening socket: inherited from the process this one replaced, else a new one. Called by __main__."""
    global _listen_socket
    fd = os.environ.pop(LISTEN_FD_ENV, None)
    if fd is not None:
        sock = socket.socket(fileno=int(fd))  # family/type are read from the fd
        sock.set_inheritable(False)
    else:
        sock = socket.create_server((host, port), family=socket.AF_INET6 if ":" in host else socket.AF_INET)
    _listen_socket = sock
    return sock


def refreshed() -> Refreshed | None:
    """Set when this process replaced an idle one (read once; the variables are removed from the environment)."""
    device = os.environ.pop(DEVICE_ENV, None)
    label = os.environ.pop(DEVICE_LABEL_ENV, None)
    return Refreshed(device, label or device) if device else None


def available() -> bool:
    return _listen_socket is not None


class Activity:
    """Open WebSockets and running HTTP requests (ASGI middleware), and when the last one ended."""

    def __init__(self) -> None:
        self.active = 0
        self.last_active = time.monotonic()

    def enter(self) -> None:
        self.active += 1

    def exit(self) -> None:
        self.active -= 1
        self.last_active = time.monotonic()

    def quiet(self, window_s: float, now: float | None = None) -> bool:
        """Nothing open, and nothing ended during the last `window_s` seconds."""
        return self.active == 0 and (time.monotonic() if now is None else now) - self.last_active >= window_s


def restart(device: str, device_label: str) -> None:
    """Replace this process with a fresh copy of itself. Does not return."""
    assert _listen_socket is not None, "refresh.restart needs the socket from adopt_or_bind"
    _listen_socket.set_inheritable(True)
    env = dict(os.environ)
    env[LISTEN_FD_ENV] = str(_listen_socket.fileno())
    env[DEVICE_ENV] = device
    env[DEVICE_LABEL_ENV] = device_label
    logging.shutdown()  # flush handlers; exec skips atexit
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
