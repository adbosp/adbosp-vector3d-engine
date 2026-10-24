"""hsrd_v1: clothed real-person scans (HSRD-100, CC BY 4.0, Digital Reality Lab) in the human_v1 layout.

Source: digitalrealitylab/HSRD-100, LOD1 meshes (100k polygons, 8K texture), downloaded and verified by
scripts/download_hsrd100.py outside the repository. Attribution: NOTICE. Not for surveillance, medical or
biometric identification use (dataset terms); the per-person demographic metadata is never copied here.

Split by PERSON (8 train / 2 test, fixed seed), so no test person appears in training. Scans are Z-up with
the front toward +Y; they are rotated to the engine frame (+Y up, front toward +Z, right side at -X),
centred in X/Z with the feet on y = 0, metres kept. Renders use the scan's own texture (vertex colours
sampled from it) with the engine's Lambert shading.
  test   one sample per pose, no augmentation: gt.glb, cameras.json, views/*.png (RGBA)
  train  per pose: `augment` samples with yaw jitter (+-20 deg) and a varied key light, the same layout,
         plus a finetune pair (cond.png + surface.npz in the front camera frame) whose target is the GT scan
         surface itself (label GT_SCAN). Background colour is not varied: TripoSG's conditioning replaces the
         background through the alpha channel, so it would not change the training signal.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..reconstruct.surface import Mesh

SOURCE = "digitalrealitylab/HSRD-100 (CC BY 4.0)"
ATTRIBUTION = ("HSRD-100: 100 High-Quality 3D Human Scans Dataset, Digital Reality Lab, 2025, "
               "https://huggingface.co/datasets/digitalrealitylab/HSRD-100, CC BY 4.0")
TEST_COUNT = 2
SPLIT_SEED = 0


def to_engine_frame(v: np.ndarray) -> np.ndarray:
    """HSRD (Z up, front +Y) -> engine (+Y up, front +Z): (x, y, z) -> (-x, z, y), a proper rotation."""
    v = np.asarray(v, float)
    return np.stack([-v[:, 0], v[:, 2], v[:, 1]], 1)


def ground(v: np.ndarray) -> np.ndarray:
    """Centre X/Z on the bounding box, feet on y = 0 (metres unchanged)."""
    lo, hi = v.min(0), v.max(0)
    return v - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]


def split(persons: list[str], test_count: int = TEST_COUNT, seed: int = SPLIT_SEED) -> dict:
    p = sorted(persons)
    test = sorted(np.random.default_rng(seed).choice(p, test_count, replace=False).tolist())
    return {"train": [x for x in p if x not in test], "test": test, "seed": seed, "by": "person"}


def poses(root) -> dict[str, list[Path]]:
    """person -> sorted LOD1 archives."""
    out: dict[str, list[Path]] = {}
    for z in sorted(Path(root, "data").glob("*/*/scans/*-Scan-LOD1.zip")):
        out.setdefault(z.parts[-4], []).append(z)
    return out


def load_scan(archive: Path, cache: Path) -> Mesh:
    """OBJ + texture from a LOD1 archive (extracted once into cache), in the engine frame, coloured."""
    import trimesh

    d = cache / archive.stem
    if not d.is_dir():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(d, [n for n in zf.namelist() if n.endswith((".obj", ".mtl", ".png"))])
    obj = next(d.glob("*.obj"))
    m = trimesh.load(obj, force="mesh", process=False)
    tex = next(d.glob("*_diffuse.png"))          # the archives carry no .mtl, so trimesh cannot find it
    cols = texture_colours(np.asarray(m.visual.uv, float), Image.open(tex))
    return Mesh(ground(to_engine_frame(m.vertices)), np.asarray(m.faces, np.int64), cols)


def texture_colours(uv: np.ndarray, texture: Image.Image) -> np.ndarray:
    """Per-vertex RGB in [0, 1] sampled (nearest texel) from the scan texture at each vertex's UV."""
    Image.MAX_IMAGE_PIXELS = None                # 8K textures exceed PIL's decompression-bomb default
    img = np.asarray(texture.convert("RGB"))
    h, w = img.shape[:2]
    u, v = np.clip(uv[:, 0], 0, 1), np.clip(uv[:, 1], 0, 1)     # scan UVs lie in [0, 1]; clamp, never wrap
    x = np.round(u * (w - 1)).astype(int)
    y = np.round((1.0 - v) * (h - 1)).astype(int)
    return img[y, x].astype(float) / 255.0


def yawed(mesh: Mesh, deg: float) -> Mesh:
    a = np.radians(deg)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return Mesh(ground(mesh.vertices @ r.T), mesh.faces, mesh.colors)


def write_sample(mesh: Mesh, d: Path, size: int, key=None) -> dict:
    """gt.glb, cameras.json and RGBA views (reference + held-out) like human_v1."""
    from ..export.writers import to_glb
    from . import dataset
    from .render import render, shade

    (d / "views").mkdir(parents=True, exist_ok=True)
    glb = to_glb(Mesh(mesh.vertices, mesh.faces), d.name, {"label": "GROUND_TRUTH", "source": SOURCE})
    (d / "gt.glb").write_bytes(glb)
    cams = dataset.cameras_for(mesh, size)
    for name, cam in cams.items():
        Image.fromarray(shade(mesh, render(mesh, cam), key=key), "RGBA").save(d / "views" / f"{name}.png")
    (d / "cameras.json").write_text(json.dumps({k: c.to_dict() for k, c in cams.items()}, indent=1))
    return {"name": d.name, "triangles": int(len(mesh.faces)), "height_m": float(np.ptp(mesh.vertices[:, 1])),
            "gt_sha256": hashlib.sha256(glb).hexdigest()}


def write_pair(mesh: Mesh, d: Path, pair_dir: Path, points: int, rng) -> None:
    """Finetune pair: the front view as cond.png, the GT scan surface in the front camera frame."""
    import shutil

    from . import dataset
    from .finetune import surface_samples
    from .omni_distill import to_camera_frame

    pair_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d / "views" / "front.png", pair_dir / "cond.png")
    cam = dataset.load_cameras(d)["front"]
    pts, nrm = surface_samples(to_camera_frame(mesh, cam), points, rng)
    np.savez_compressed(pair_dir / "surface.npz", points=pts, normals=nrm)
    (pair_dir / "provenance.json").write_text(json.dumps(
        {"target": "GT_SCAN", "source": SOURCE, "attribution": ATTRIBUTION, "uses_ground_truth": True,
         "sample": d.name}, indent=1))


def build(root, out_dir, size: int = 512, augment: int = 3, points: int = 40960, seed: int = 0,
          cache=None, persons=None) -> dict:
    root, out = Path(root), Path(out_dir)
    cache = Path(cache or root / "unpacked")
    by_person = poses(root)
    sp = split(list(by_person))
    rng = np.random.default_rng(seed)
    manifest = {"version": "hsrd_v1", "source": SOURCE, "attribution": ATTRIBUTION, "split": sp,
                "image_size": size, "augment_per_train_pose": augment, "train": [], "test": [],
                "license": "CC BY 4.0; not for surveillance, medical or biometric identification"}
    for person, archives in by_person.items():
        if persons and person not in persons:
            continue
        part = "test" if person in sp["test"] else "train"
        for z in archives:
            mesh = load_scan(z, cache)
            pose = z.parts[-3]
            if part == "test":
                manifest["test"].append(write_sample(mesh, out / "test" / pose, size))
                continue
            for k in range(augment):
                yaw = float(rng.uniform(-20, 20))
                key = rng.normal([0.4, 0.5, 0.6], 0.25)
                d = out / "train" / f"{pose}_a{k}"
                row = write_sample(yawed(mesh, yaw), d, size, key=key / np.linalg.norm(key))
                write_pair(yawed(mesh, yaw), d, out / "pairs" / d.name, points, np.random.default_rng([seed, len(manifest["train"])]))
                manifest["train"].append({**row, "yaw_deg": yaw, "key_light_camera_axes": (key / np.linalg.norm(key)).tolist()})
    for part in ("train", "test"):
        (out / part).mkdir(parents=True, exist_ok=True)
        sub = {k: v for k, v in manifest.items() if k not in ("train", "test")}
        sub["objects"] = [{"name": r["name"], "category": "human_clothed", "triangles": r["triangles"]}
                          for r in manifest[part]]
        from . import dataset
        sub.update(version=f"hsrd_v1_{part}", reference_views=list(dataset.REFERENCE_VIEWS),
                   heldout_views=[h[0] for h in dataset.HELDOUT_VIEWS])
        (out / part / "manifest.json").write_text(json.dumps(sub, indent=1))
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    (out / "pairs" / "manifest.json").write_text(json.dumps(
        {"version": "v3d_finetune_v1", "kind": "gt_scan", "source": SOURCE, "attribution": ATTRIBUTION,
         "count": len(manifest["train"]), "points": points}, indent=1))
    (out / "hsrd_v1.json").write_text(json.dumps(manifest, indent=1))
    return manifest
