"""End-to-end smoke test of Omni mode through the running web UI API.

Usage: python scripts/omni_smoke.py IMAGE OUT_DIR [--url http://127.0.0.1:8765] [--isolation contour --depth body]
Posts the image with mode omni, polls the job until it ends, prints each step's status and detail, and
saves model.glb, metadata.json and job.json into OUT_DIR. Exit 0 only if the job succeeded.
"""
import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path


def call(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    return body if url.endswith((".glb", ".obj", ".png")) else json.loads(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--isolation", default="contour")
    ap.add_argument("--depth", default="body")
    a = ap.parse_args()
    img = Path(a.image)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"views": {"front": {"name": img.name, "data": base64.b64encode(img.read_bytes()).decode()}},
               "options": {"mode": "omni", "omni_isolation": a.isolation, "omni_depth": a.depth}}
    t0 = time.time()
    job = call(f"{a.url}/api/generate", payload)["job"]
    while True:
        snap = call(f"{a.url}/api/jobs/{job}")
        if snap.get("status") in ("done", "failed", "error", "success"):
            break
        time.sleep(5)
    snap["wall_s"] = time.time() - t0
    (out / "job.json").write_text(json.dumps(snap, indent=1))
    for s in snap.get("steps", []):
        print(f"[{s.get('status'):>8}] {s.get('key')}: {s.get('detail')}")
    print("status", snap.get("status"), "wall", round(snap["wall_s"]), "s", snap.get("error") or "")
    if snap.get("files"):
        (out / "model.glb").write_bytes(call(a.url + snap["files"]["glb"]))
        (out / "metadata.json").write_text(json.dumps(call(a.url + snap["files"]["metadata"]), indent=1))
    return 0 if snap.get("files") else 1


if __name__ == "__main__":
    sys.exit(main())
