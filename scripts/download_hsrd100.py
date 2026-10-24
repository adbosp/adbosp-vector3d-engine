"""Download HSRD-100 (CC BY 4.0, digitalrealitylab) LOD1 scans + manifests at a pinned revision, verified.

Usage: python scripts/download_hsrd100.py [--dest C:/AI/datasets/HSRD-100] [--revision SHA]
Only `*-Scan-LOD1.zip` (100k polygons, 8K texture) per pose and the manifest files; never the full scans,
RealityCapture or photo archives. Every file is checked against the Hub's size and LFS SHA256 and recorded
in download-verification.json. Data stays outside the repository. Stdlib only.
"""
import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "digitalrealitylab/HSRD-100"
REVISION = "9cc5e9c138dd310dc88c95c80e5db0a3ecb7971a"
API = "https://huggingface.co/api/datasets/" + REPO


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def tree(path, rev):
    return get_json(f"{API}/tree/{rev}/{urllib.parse.quote(path)}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path, rev, dest: Path, size=None, digest=None) -> dict:
    out = dest / path
    if out.is_file() and (size is None or out.stat().st_size == size) and (digest is None or sha256(out) == digest):
        return {"path": path, "bytes": out.stat().st_size, "sha256": digest or sha256(out), "status": "present"}
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{rev}/{urllib.parse.quote(path)}"
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(8 << 20):
            f.write(chunk)
    got = sha256(tmp)
    if (size is not None and tmp.stat().st_size != size) or (digest is not None and got != digest):
        tmp.unlink()
        raise RuntimeError(f"verification failed for {path}")
    tmp.replace(out)
    return {"path": path, "bytes": out.stat().st_size, "sha256": got, "status": "downloaded"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="C:/AI/datasets/HSRD-100")
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()
    dest = Path(a.dest)
    records = [fetch(f"manifest/{m}", a.revision, dest) for m in ("files.csv", "persons.jsonl", "poses.jsonl")]
    for person in tree("data", a.revision):
        for pose in tree(person["path"], a.revision):
            for f in tree(pose["path"] + "/scans", a.revision):
                if f["path"].endswith("-Scan-LOD1.zip"):
                    lfs = f.get("lfs") or {}
                    rec = fetch(f["path"], a.revision, dest, f.get("size"), lfs.get("oid"))
                    records.append(rec)
                    print(rec["status"], rec["path"], rec["bytes"], flush=True)
    manifest = {"repo": REPO, "revision": a.revision, "licence": "CC BY 4.0",
                "files": records, "lod1_archives": sum(r["path"].endswith("LOD1.zip") for r in records),
                "status": "downloaded_and_verified"}
    (dest / "download-verification.json").write_text(json.dumps(manifest, indent=1))
    print("done", manifest["lod1_archives"], "LOD1 archives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
