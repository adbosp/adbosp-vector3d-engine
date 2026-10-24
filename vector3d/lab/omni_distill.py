"""Omni distillation data: TRELLIS (and other teachers) -> training pairs for TripoSG, the Omni core.

Ropy's pipeline (2026-10-24): Omni stays one TripoSG pass; knowledge from TRELLIS is distilled into it
(LoRA trained with lab/finetune.py). Only teacher outputs that measurably beat TripoSG on the same image
become training targets:

  per sample (a human_v1-layout folder: views/*.png, cameras.json; the front image is the input)
    -> every generator runs on views/front.png            (TripoSG = the base, teachers = e.g. TRELLIS)
    -> each mesh is placed from the front silhouette only (align.place_front_only)
    -> arbiter: mean silhouette IoU over the other views the generators never saw, plus surface smoothness
    -> a teacher is accepted when it beats the base by `margin` IoU without being rougher; its placed mesh,
       in the front camera's frame (+X right, +Y up, +Z toward the viewer), is written as cond.png +
       surface.npz in the lab/finetune.py format (v3d_finetune_v1)

Labels: teacher targets are AI_GENERATED (selected by a measured check, not ground truth). Nothing here
reads gt.glb. Rejected samples are listed with their scores; nothing is written for them.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..reconstruct.surface import Mesh
from .align import place_front_only
from .metrics import iou, normal_consistency
from .render import silhouette

Generate = Callable[[Path, Path], Path]       # (input png, output dir) -> raw mesh file


def to_camera_frame(mesh: Mesh, cam) -> Mesh:
    """World -> the camera's own axes (right, up, toward), so +Z points at the viewer of the input."""
    b = np.array([cam.right, cam.up, cam.toward], float)
    return Mesh((mesh.vertices - np.asarray(cam.center, float)) @ b.T, mesh.faces, mesh.colors)


def score(mesh: Mesh, masks: dict, cams: dict, views: list[str]) -> dict:
    """Held-out silhouette IoU per view and smoothness of an already placed mesh."""
    per = {v: iou(silhouette(mesh, cams[v]), masks[v]) for v in views}
    return {"iou": per, "mean_iou": float(np.mean(list(per.values()))) if per else 0.0,
            "normal_consistency": normal_consistency(mesh)}


def arbitrate(scores: dict, base: str, margin: float = 0.02, max_rougher: float = 0.01) -> dict:
    """Best teacher that beats the base by `margin` mean IoU and is at most `max_rougher` less smooth."""
    b = scores[base]
    best, reason = None, "no teacher beat the base"
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["mean_iou"]):
        if name == base or "error" in s:
            continue
        gain = s["mean_iou"] - b["mean_iou"]
        if gain < margin:
            reason = f"best teacher {name} gains {gain:+.3f} IoU (< {margin})"
            break
        if b["normal_consistency"] - s["normal_consistency"] > max_rougher:
            reason = f"{name} gains {gain:+.3f} IoU but is rougher by {b['normal_consistency'] - s['normal_consistency']:.3f}"
            continue
        best, reason = name, f"{name} gains {gain:+.3f} IoU over {base}"
        break
    return {"accepted": best, "reason": reason}


def build(data_dir, out_dir, generators: dict[str, Generate], base: str = "triposg", names=None,
          points: int = 40960, margin: float = 0.02, seed: int = 0, load=None) -> dict:
    """Run every generator on each sample's front view, arbitrate, write accepted pairs. A generator that
    fails on a sample is recorded and scored as missing; if the base fails the sample is skipped."""
    from . import dataset
    from .finetune import surface_samples
    from .mesh_io import load_mesh

    load = load or load_mesh
    if base not in generators:
        raise ValueError(f"base generator {base!r} missing")
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for d in sorted(p for p in data_dir.iterdir() if (p / "cameras.json").is_file()):
        if names and d.name not in names:
            continue
        cams = dataset.load_cameras(d)
        views = [v for v in cams if v != "front" and (d / "views" / f"{v}.png").is_file()]
        masks = dataset.load_masks(d, ["front", *views])
        front = d / "views" / "front.png"
        work = out / "_runs" / d.name
        scores, placed = {}, {}
        for name, gen in generators.items():
            t1 = time.perf_counter()
            try:
                raw = load(gen(front, work / name))
                placed[name], _ = place_front_only(raw, masks["front"], cams["front"])
                scores[name] = {**score(placed[name], masks, cams, views), "seconds": time.perf_counter() - t1}
            except Exception as e:  # recorded; never replaced by another mesh
                scores[name] = {"error": f"{type(e).__name__}: {e}", "mean_iou": -1.0}
        row = {"name": d.name, "heldout_views": views, "scores": scores}
        if "error" in scores[base]:
            row.update(status="skipped", reason=f"base {base} failed: {scores[base]['error']}")
        else:
            row.update(arbitrate(scores, base, margin))
            if row["accepted"]:
                s = out / d.name
                s.mkdir(exist_ok=True)
                shutil.copy2(front, s / "cond.png")
                pts, nrm = surface_samples(to_camera_frame(placed[row["accepted"]], cams["front"]), points,
                                           np.random.default_rng([seed, len(rows)]))
                np.savez_compressed(s / "surface.npz", points=pts, normals=nrm)
                (s / "provenance.json").write_text(json.dumps(
                    {"teacher": row["accepted"], "label": "AI_GENERATED", "selected_by": row["reason"],
                     "scores": scores, "uses_ground_truth": False}, indent=1, default=str))
                row["status"] = "accepted"
            else:
                row["status"] = "rejected"
        rows.append(row)
    manifest = {"version": "v3d_finetune_v1", "kind": "omni_distill", "base": base,
                "teachers": [g for g in generators if g != base], "margin": margin, "points": points,
                "accepted": sum(r["status"] == "accepted" for r in rows), "count": len(rows),
                "seconds": time.perf_counter() - t0, "samples": rows,
                "label": "teacher targets are AI_GENERATED, selected by held-out silhouette IoU against the base"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def backend_generator(name: str, profile: str = "quality", seed: int = 42, **kw) -> Generate:
    """Generate callable for any registered lab backend (triposg, trellis, ...); raises on failure."""
    from .backends import get_backend

    be = get_backend(name, **kw)
    ok, why = be.check()
    if not ok:
        raise RuntimeError(f"{name} blocked: {why}")

    def gen(image: Path, out: Path) -> Path:
        rec = be.generate({"front": image}, quality_profile=profile, out_dir=out, seed=seed)
        if rec.status != "success" or not rec.output_path:
            raise RuntimeError(f"{name} {rec.status}: {rec.reason}")
        return Path(rec.output_path)

    return gen
