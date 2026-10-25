"""human_v1: synthetic people with held-out ground truth, in the synthetic_v1 layout.

People are sampled from MHR v1.0.1 (Apache-2.0) by backends/human_v1_gen.py in the anatomy WSL env
(random identity, yaw, arm/spine motion, per-finger curl and spread within MHR's own joint limits).
Each person gets gt.glb, joints.json (MHR joints, metres) and the usual reference + held-out views, so
lab benchmark runs on it unchanged. Generation only ever sees views/front.png. Placement follows a named
protocol (PROTOCOLS): front_only is the single-image end-to-end protocol; multiview_oracle is what lab
benchmark's A_placed.glb uses (three extra reference silhouettes). Held-out views and GT are for scoring.

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


DENSITY = 2.0e6          # surface samples per m^2 on both sides (about 0.7 mm spacing)
QUERY_PAD = 0.05         # nearest neighbours are searched within the box plus this margin


def area(mesh: Mesh) -> float:
    v = mesh.vertices[mesh.faces]
    return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2)


def region_metrics(pred: Mesh, gt: Mesh, box: tuple[np.ndarray, np.ndarray], seed: int = 0,
                   density: float = DENSITY) -> dict:
    """Chamfer (mm) and F-score@5 mm restricted to a GT region box; pred must be in the GT frame.
    Both meshes are cropped to the box plus QUERY_PAD and sampled at the same areal density with fixed
    seeds, so the sampling noise is the same whichever way the prediction was placed. Distances to
    surface farther than QUERY_PAD outside the box are not seen (errors above 5 cm are underestimated)."""
    lo, hi = box
    gc, pc = crop_box(gt, box, QUERY_PAD), crop_box(pred, box, QUERY_PAD)
    if len(gc.faces) < 20 or len(pc.faces) < 20:
        return {"status": "empty", "pred_faces": int(len(pc.faces))}
    ng = int(min(4e5, max(2000, density * area(gc))))
    npd = int(min(4e5, max(2000, density * area(pc))))
    gp, pp = sample_surface(gc, ng, seed), sample_surface(pc, npd, seed + 1)
    gin = np.all((gp >= lo) & (gp <= hi), 1)
    pin = np.all((pp >= lo) & (pp <= hi), 1)
    if gin.sum() < 50 or pin.sum() < 50:
        return {"status": "empty", "gt_points": int(gin.sum()), "pred_points": int(pin.sum())}
    d_g = cKDTree(pp).query(gp[gin])[0]        # GT region -> pred near the region (missing parts)
    d_p = cKDTree(gp).query(pp[pin])[0]        # pred region -> GT near the region (extra parts)
    t = 0.005
    prec, rec = float((d_p < t).mean()), float((d_g < t).mean())
    return {"chamfer_mm": float((d_g.mean() + d_p.mean()) / 2 * 1000),
            "fscore@5mm": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "gt_points": int(gin.sum()), "pred_points": int(pin.sum()), "density_per_m2": density}


def summarize(scores: dict, variant: str, expected=None) -> dict:
    """Coverage and statistics over all hands; an empty hand counts as F = 0 in fscore_all.
    A person without outputs ({"status": "missing"}) or listed in `expected` (e.g. the manifest's people)
    but absent from `scores` counts as two missing hands, F = 0: a partial batch cannot look complete."""
    people = {k: v for k, v in scores.items() if isinstance(v, dict) and ("l" in v or "status" in v)}
    for name in expected or ():
        people.setdefault(name, {"status": "missing"})
    rows, missing = [], 0
    for v in people.values():
        if "l" in v:
            rows += [v[s][variant] for s in ("l", "r")]
        else:
            rows += [{"status": "missing"}] * 2
            missing += 2
    got = [r for r in rows if "chamfer_mm" in r]
    return {"hands": len(rows), "measured": len(got), "empty": len(rows) - len(got) - missing,
            "missing": missing, "people": len(people),
            "coverage": len(got) / len(rows) if rows else 0.0,
            "chamfer_mm_median_measured": float(np.median([r["chamfer_mm"] for r in got])) if got else None,
            "chamfer_mm_mean_measured": float(np.mean([r["chamfer_mm"] for r in got])) if got else None,
            "fscore@5mm_mean_all": float(np.mean([r.get("fscore@5mm", 0.0) for r in rows])) if rows else None,
            "fscore@5mm_mean_measured": float(np.mean([r["fscore@5mm"] for r in got])) if got else None}


def manifest_people(data_dir) -> list[str]:
    """Expected people of a human_v1 folder, from manifest.json (not from whatever folders exist)."""
    return [o["name"] for o in json.loads((Path(data_dir) / "manifest.json").read_text())["objects"]]


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
    r = region_metrics(aligned, gt, box)
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


PROTOCOLS = {
    "multiview_oracle": "single-image generation + placement from the front/back/left/right reference "
                        "silhouettes (benchmark A_placed.glb); the three extra views are oracle information",
    "front_only": "single-image end to end: generation and placement use only views/front.png and the front "
                  "camera (align.place_front_only on the raw model output)",
    "body_front_only_hires": "single-image end to end on the high-resolution front render (render_front): "
                             "whole-body generation placed front-only; the same input as the teacher",
    "teacher_front_only": "Omni teacher v0 on the high-resolution front render: body + hand close-ups from 2D "
                          "keypoints of that image, placed and merged without GT (teacher.run)",
}
PROTOCOL_FILES = {"multiview_oracle": "A_placed.glb", "front_only": "raw_triposg.glb",
                  "body_front_only_hires": "body_placed.glb", "teacher_front_only": "teacher_placed.glb"}
RAW_OUTPUTS = ("raw_triposg.glb", "raw_trellis.glb")      # raw model output of each lab backend


def protocol_file(run_dir: Path, protocol: str) -> Path:
    """The file a protocol scores in one benchmark run folder (front_only: whichever backend's raw output)."""
    if protocol == "front_only":
        found = [run_dir / f for f in RAW_OUTPUTS if (run_dir / f).is_file()]
        return found[0] if found else run_dir / RAW_OUTPUTS[0]
    return run_dir / PROTOCOL_FILES[protocol]


def render_front(obj_dir, size: int = 2048):
    """views/front_<size>.png: the front camera of cameras.json at a higher resolution (same view, same
    framing). Stands in for a high-resolution photo; nothing but the image and its camera is derived."""
    from dataclasses import replace

    from . import dataset
    from .mesh_io import load_mesh
    from .render import render, shade

    obj_dir = Path(obj_dir)
    cam = replace(dataset.load_cameras(obj_dir)["front"], size=int(size), name=f"front_{size}")
    png = obj_dir / "views" / f"front_{size}.png"
    if not png.is_file():
        mesh = load_mesh(obj_dir / "gt.glb")
        Image.fromarray(shade(mesh, render(mesh, cam)), "RGBA").save(png)
    return png, cam


def placed_mesh(run_dir: Path, obj_dir: Path, protocol: str) -> Mesh:
    from . import dataset
    from .align import place_front_only
    from .mesh_io import load_mesh

    if protocol == "multiview_oracle":
        return load_mesh(run_dir / "A_placed.glb")
    if protocol in ("body_front_only_hires", "teacher_front_only"):
        return load_mesh(run_dir / PROTOCOL_FILES[protocol])      # already placed by teacher.run
    if protocol == "front_only":
        cams = dataset.load_cameras(obj_dir)
        front = dataset.load_masks(obj_dir, ["front"])["front"]
        return place_front_only(load_mesh(protocol_file(run_dir, protocol)), front, cams["front"])[0]
    raise ValueError(f"unknown protocol {protocol!r}; choose from {sorted(PROTOCOLS)}")


def hand_scores(bench_dir, data_dir, protocol: str = "front_only") -> dict:
    """Per person and hand: placed (protocol placement, no GT) and local (rigid crop ICP, uses GT) region
    metrics, plus the GT-vs-GT noise floor of the same sampling."""
    from .mesh_io import load_mesh

    bench_dir, data_dir = Path(bench_dir), Path(data_dir)
    out = {"protocol": protocol, "protocol_description": PROTOCOLS[protocol], "density_per_m2": DENSITY}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        run = bench_dir / d.name
        if not protocol_file(run, protocol).is_file():
            out[d.name] = {"status": "missing"}
            continue
        gt, pred = load_mesh(d / "gt.glb"), placed_mesh(run, d, protocol)
        joints = {k: np.array(v) for k, v in json.loads((d / "joints.json").read_text()).items()}
        out[d.name] = {side: {"placed": region_metrics(pred, gt, hand_region(joints, side)),
                              "local": region_metrics_local(pred, gt, hand_region(joints, side)),
                              "noise_floor": region_metrics(gt, gt, hand_region(joints, side), seed=7)}
                       for side in ("l", "r")}
    return out


def body_scores(bench_dir, data_dir, protocol: str = "front_only") -> dict:
    """Whole-body A metrics (experiments.evaluate) under a placement protocol, from the raw outputs."""
    from . import dataset
    from .experiments import evaluate
    from .mesh_io import load_mesh

    bench_dir, data_dir = Path(bench_dir), Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = {"protocol": protocol, "protocol_description": PROTOCOLS[protocol]}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        run = bench_dir / d.name
        if not protocol_file(run, protocol).is_file():
            out[d.name] = {"status": "missing"}
            continue
        cams = dataset.load_cameras(d)
        out[d.name] = evaluate(placed_mesh(run, d, protocol), load_mesh(d / "gt.glb"), cams,
                               dataset.load_masks(d, manifest["reference_views"]),
                               dataset.load_masks(d, manifest["heldout_views"]))
    return out
