#!/usr/bin/env python3
"""Writes release.json and runtime-manifest.json for a staged runtime, after signing.

release.json keeps the fields upstream's own packaging writes (version, binary_sha256, metallib_sha256), now describing
the files as they ship, and adds where they came from. runtime-manifest.json lists the SHA-256 of every file that is
not in the bundled Python tree, plus every Mach-O file inside it: what the app checks before it starts the runtime.
"""
import hashlib
import json
import os
import subprocess
import sys

root, upstream_path = sys.argv[1], sys.argv[2]
upstream = json.load(open(upstream_path))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_macho(path):
    with open(path, "rb") as f:
        return f.read(4) in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xce\xfa\xed\xfe")


files = {}
for base, _dirs, names in os.walk(root):
    for name in sorted(names):
        path = os.path.join(base, name)
        rel = os.path.relpath(path, root)
        if rel in ("release.json", "runtime-manifest.json") or os.path.islink(path):
            continue
        in_python = rel.startswith("python" + os.sep)
        if in_python and not is_macho(path):
            continue
        files[rel] = digest(path)

release = {
    "version": upstream["runtime_version"],
    "binary_sha256": files["engine/splash"],
    "metallib_sha256": files["engine/splash.metallib"],
    "base": {
        "upstream_version": upstream["upstream_version"],
        "archive_sha256": upstream["sha256"],
        "upstream_engine_sha256": upstream["upstream_engine_sha256"],
    },
    "features": ["drain"],
    "patches": ["server/connections.py", "server/server.py"],
}
json.dump(release, open(os.path.join(root, "release.json"), "w"), indent=2)
open(os.path.join(root, "release.json"), "a").write("\n")
files["release.json"] = digest(os.path.join(root, "release.json"))
json.dump({"runtime_version": upstream["runtime_version"], "files": dict(sorted(files.items()))},
          open(os.path.join(root, "runtime-manifest.json"), "w"), indent=1)
print(f"manifest: {len(files)} files")
