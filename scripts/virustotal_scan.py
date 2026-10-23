#!/usr/bin/env python3
"""Upload a file to VirusTotal (API v3), wait for the analysis, print a Markdown summary line.

    VT_API_KEY=... python3 scripts/virustotal_scan.py FILE [--max-malicious N] [--timeout 900]

Used by .github/workflows/release.yml. Exit 0 = analysis finished with at most N engines (default 2: Android
installers that self-update often draw a heuristic flag or two) reporting "malicious"; 1 = more, or timeout/error.
The summary (stdout) links the public report: https://www.virustotal.com/gui/file/<sha256>.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

API = "https://www.virustotal.com/api/v3"
DIRECT_UPLOAD_LIMIT = 32 * 2**20  # bigger files need a one-time upload URL


def call(method: str, url: str, key: str, body: bytes | None = None, content_type: str | None = None) -> dict:
    headers = {"x-apikey": key, "accept": "application/json"}
    if content_type:
        headers["content-type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=300) as r:
        return json.loads(r.read())


def upload(path: Path, key: str) -> str:
    """→ analysis id."""
    url = f"{API}/files"
    if path.stat().st_size > DIRECT_UPLOAD_LIMIT:
        url = call("GET", f"{API}/files/upload_url", key)["data"]
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return call("POST", url, key, body, f"multipart/form-data; boundary={boundary}")["data"]["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--max-malicious", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for the analysis")
    args = parser.parse_args()
    key = os.environ.get("VT_API_KEY", "")
    if not key:
        print("VT_API_KEY not set", file=sys.stderr)
        return 1

    sha256 = hashlib.sha256(args.file.read_bytes()).hexdigest()
    report = f"https://www.virustotal.com/gui/file/{sha256}"
    analysis_id = upload(args.file, key)
    print(f"uploaded {args.file.name}, analysis {analysis_id}", file=sys.stderr)
    deadline = time.monotonic() + args.timeout
    while True:
        attributes = call("GET", f"{API}/analyses/{analysis_id}", key)["data"]["attributes"]
        if attributes.get("status") == "completed":
            break
        if time.monotonic() > deadline:
            print(f"VirusTotal: analysis not finished after {args.timeout:.0f} s — [report]({report})")
            return 1
        time.sleep(20)  # public API: 4 requests/minute
    stats = attributes["stats"]
    malicious, suspicious = stats.get("malicious", 0), stats.get("suspicious", 0)
    engines = sum(v for k, v in stats.items() if k in ("malicious", "suspicious", "undetected", "harmless"))
    print(f"VirusTotal: {malicious} malicious, {suspicious} suspicious of {engines} engines — [report]({report})")
    return 0 if malicious <= args.max_malicious else 1


if __name__ == "__main__":
    sys.exit(main())
