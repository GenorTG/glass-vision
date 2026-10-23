#!/usr/bin/env python3
"""Download the latest app release (APK + version sidecar) from GitHub so this server can hand it out to phones.

    python3 scripts/fetch_release_apk.py --repo GenorTG/glass-vision [--dest server/data/app] [--tag v0.6.0]

Writes `<dest>/glass-vision.apk` and `<dest>/glass-vision.apk.json` (served at /app.apk and /app/version, see
server/glass_vision/apk_info.py). The APK's sha256 is checked against the release's SHA256SUMS. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/{repo}/releases/{which}"
UA = {
    "User-Agent": "glass-vision-fetch-release/1",
    "Accept": "application/vnd.github+json",
}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="GitHub owner/name")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "server" / "data" / "app",
    )
    parser.add_argument("--tag", help="release tag (default: latest)")
    args = parser.parse_args()

    release = json.loads(get(API.format(repo=args.repo, which=f"tags/{args.tag}" if args.tag else "latest")))
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    apk_name = next((n for n in assets if n.endswith(".apk")), None)
    if apk_name is None or apk_name + ".json" not in assets:
        print(
            f"release {release.get('tag_name')} has no APK + .apk.json sidecar: {sorted(assets)}",
            file=sys.stderr,
        )
        return 1

    meta = json.loads(get(assets[apk_name + ".json"]))
    if not {"application_id", "version_code", "version_name"} <= meta.keys():
        print(f"bad sidecar {apk_name}.json: {meta}", file=sys.stderr)
        return 1
    expected = None
    if "SHA256SUMS" in assets:
        for line in get(assets["SHA256SUMS"]).decode().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == apk_name:
                expected = parts[0].lower()

    args.dest.mkdir(parents=True, exist_ok=True)
    apk, sidecar = args.dest / "glass-vision.apk", args.dest / "glass-vision.apk.json"
    tmp = apk.with_name(apk.name + ".part")
    print(f"downloading {apk_name} ({meta['version_name']}) from {release.get('html_url')}")
    data = get(assets[apk_name])
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        print(
            f"sha256 mismatch for {apk_name}: {actual} != SHA256SUMS {expected}",
            file=sys.stderr,
        )
        return 1
    if expected is None:
        print(
            "warning: release has no SHA256SUMS entry for the APK; not verified",
            file=sys.stderr,
        )
    tmp.write_bytes(data)
    os.replace(tmp, apk)
    sidecar.write_text(json.dumps({**meta, "sha256": actual}, indent=2) + "\n")
    print(f"installed {apk} ({len(data) / 2**20:.1f} MiB, sha256 {actual[:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
