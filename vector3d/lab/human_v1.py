"""human_v1: synthetic people with held-out ground truth, in the synthetic_v1 layout.

People are sampled from MHR v1.0.1 (Apache-2.0) by backends/human_v1_gen.py in the anatomy WSL env
(random identity, yaw, arm/spine motion, per-finger curl and spread within MHR's own joint limits).
Each person gets gt.glb, joints.json (MHR joints, metres) and the usual reference + held-out views, so
lab benchmark runs on it unchanged. Methods only ever see views/front.png; everything else is GT.

Limits: smooth MHR skin (no hair, clothes, nails or wrinkles), untextured Lambert renders, and the
bodies come from the same model the anatomy step fits, so that step is in-distribution here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..reconstruct.surface import Mesh
from .metrics import sample_surface

HAND_JOINTS = ("wrist", "thumb_null", "index_null", "middle_null", "ring_null", "pinky_null", "index1", "pinky1")


def hand_region(joints: dict, side: str, margin: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around one hand's joints (wrist to fingertips), metres."""
    pts = np.array([joints[f"{side}_{j}"] for j in HAND_JOINTS])
    return pts.min(0) - margin, pts.max(0) + margin


def region_metrics(pred: Mesh, gt: Mesh, box: tuple[np.ndarray, np.ndarray], n: int = 200000, seed: int = 0) -> dict:
    """Chamfer (mm) and F-score@5 mm restricted to a GT region box; pred must be in the GT frame."""
    lo, hi = box
    gp, pp = sample_surface(gt, n, seed), sample_surface(pred, n, seed + 1)
    gin = np.all((gp >= lo) & (gp <= hi), 1)
    pin = np.all((pp >= lo) & (pp <= hi), 1)
    if gin.sum() < 50 or pin.sum() < 50:
        return {"status": "empty", "gt_points": int(gin.sum()), "pred_points": int(pin.sum())}
    d_g = cKDTree(pp).query(gp[gin])[0]        # GT region -> all of pred (missing parts)
    d_p = cKDTree(gp).query(pp[pin])[0]        # pred region -> all of GT (extra parts)
    t = 0.005
    prec, rec = float((d_p < t).mean()), float((d_g < t).mean())
    return {"chamfer_mm": float((d_g.mean() + d_p.mean()) / 2 * 1000),
            "fscore@5mm": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "gt_points": int(gin.sum()), "pred_points": int(pin.sum())}


def crop_box(mesh: Mesh, box: tuple[np.ndarray, np.ndarray], pad: float = 0.0) -> Mesh:
    lo, hi = box[0] - pad, box[1] + pad
    keep = np.all((mesh.vertices >= lo) & (mesh.vertices <= hi), axis=1)
    fk = keep[mesh.faces].all(1)
    remap = -np.ones(len(keep), int)
    remap[keep] = np.arange(keep.sum())
    return Mesh(mesh.vertices[keep], remap[mesh.faces[fk]])


def region_metrics_local(pred: Mesh, gt: Mesh, box, pad: float = 0.05) -> dict:
    """Shape-only variant: crop both meshes around the box, rigidly ICP the predicted crop onto the GT crop
    (uses GT, evaluation only), then region_metrics. Separates hand shape from whole-body placement."""
    from .handzoom import similarity_icp

    pc, gc = crop_box(pred, box, pad), crop_box(gt, box, pad)
    if len(pc.faces) < 20 or len(gc.faces) < 20:
        return {"status": "empty", "pred_faces": int(len(pc.faces))}
    aligned, info = similarity_icp(pc, gc, scale=False)
    r = region_metrics(aligned, gc, box)
    r["alignment"] = "rigid ICP of the cropped region onto GT (evaluation only)"
    r["icp_shift_mm"] = float(np.linalg.norm(info["translation"]) * 1000)
    return r


def build(raw_dir, out_dir, size: int = 512) -> dict:
    """Turn human_v1_gen.py output (person_XX.npz) into the benchmark layout."""
    from ..export.writers import to_glb
    from . import dataset
    from .render import render, shade

    raw, out = Path(raw_dir), Path(out_dir)
    manifest = {"version": "human_v1", "image_size": size, "reference_views": list(dataset.REFERENCE_VIEWS),
                "heldout_views": [h[0] for h in dataset.HELDOUT_VIEWS],
                "license": "bodies sampled from MHR v1.0.1 (Apache-2.0) by vector3d/lab/backends/human_v1_gen.py",
                "objects": []}
    for f in sorted(raw.glob("person_*.npz")):
        z = np.load(f)
        mesh = Mesh(z["vertices"].astype(float), z["faces"].astype(np.int64))
        d = out / f.stem
        (d / "views").mkdir(parents=True, exist_ok=True)
        glb = to_glb(mesh, f.stem, {"label": "GROUND_TRUTH", "source": "MHR v1.0.1 sample"})
        (d / "gt.glb").write_bytes(glb)
        joints = {str(n): z["joints"][i].tolist() for i, n in enumerate(z["joint_names"])}
        (d / "joints.json").write_text(json.dumps(joints))
        cams = dataset.cameras_for(mesh, size)
        for name, cam in cams.items():
            Image.fromarray(shade(mesh, render(mesh, cam)), "RGBA").save(d / "views" / f"{name}.png")
        (d / "cameras.json").write_text(json.dumps({k: c.to_dict() for k, c in cams.items()}, indent=1))
        manifest["objects"].append({"name": f.stem, "category": "human", "triangles": int(len(mesh.faces)),
                                    "height_m": float(np.ptp(mesh.vertices[:, 1])),
                                    "gt_sha256": hashlib.sha256(glb).hexdigest()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def hand_scores(bench_dir, data_dir, variant: str = "A_placed.glb") -> dict:
    """Region metrics of each person's placed output (benchmark A) for both hands."""
    from .mesh_io import load_mesh

    bench_dir, data_dir = Path(bench_dir), Path(data_dir)
    out = {}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        f = bench_dir / d.name / variant
        if not f.is_file():
            out[d.name] = {"status": "missing"}
            continue
        gt, pred = load_mesh(d / "gt.glb"), load_mesh(f)
        joints = {k: np.array(v) for k, v in json.loads((d / "joints.json").read_text()).items()}
        out[d.name] = {side: {"placed": region_metrics(pred, gt, hand_region(joints, side)),
                              "local": region_metrics_local(pred, gt, hand_region(joints, side))}
                       for side in ("l", "r")}
    return out
