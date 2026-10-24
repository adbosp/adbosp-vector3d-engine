"""Omni teacher pipeline, step 1: hand close-ups assembled into the whole body, single image, no GT.

  front image + front camera
    -> whole-body generation, placed with align.place_front_only           (body, ESTIMATED)
    -> hand boxes from the 2D whole-body keypoints (RTMW, COCO-WholeBody)   (AI_GENERATED)
    -> per hand: crop (hand isolated by a disc around its keypoints), generate, place the part with the
       crop camera, depth matched to the body's surface in the same pixels  (ESTIMATED)
    -> merge: body faces under the hand region and within the part's depth band are removed, the part is
       added. The result is a triangle soup, not watertight (reported, not hidden).

Outputs per run: body_placed.glb (the same-input baseline: whole body only) and teacher_placed.glb.
Every input comes from the one front image and its camera: generation is a callable, so the same code runs
TripoSG on the RTX 3060 and a stand-in in tests. Nothing here reads GT; scoring is human_v1.hand_scores.
Limits: orthographic cameras only; the depth match uses the body mesh, so a body arm placed at the wrong
depth carries the hand with it; MHR finger priors are not used yet (next step).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from ..reconstruct.surface import Mesh
from .align import place_front_only
from .render import OrthoCamera

# COCO-WholeBody 133: 0-16 body, 17-22 feet, 23-90 face, 91-111 left hand, 112-132 right hand.
HAND_KEYPOINTS = {"l": range(91, 112), "r": range(112, 133)}


@dataclass(frozen=True)
class HandBox:
    side: str
    cx: float            # crop centre, full-image pixels
    cy: float
    side_px: float       # square crop side, full-image pixels
    points: np.ndarray   # (k, 2) confident hand keypoints, full-image pixels
    radius_px: float     # isolation disc radius around each keypoint

    def to_dict(self) -> dict:
        return {"side": self.side, "cx": self.cx, "cy": self.cy, "side_px": self.side_px,
                "keypoints_used": int(len(self.points)), "radius_px": self.radius_px}


def hand_box(xy, score, side: str, min_score: float = 0.3, min_points: int = 6, pad: float = 0.35,
             min_side_px: float = 24.0) -> HandBox | None:
    """Square crop around one hand's confident keypoints; None when too few are confident."""
    xy, score = np.asarray(xy, float), np.asarray(score, float)
    idx = np.array(list(HAND_KEYPOINTS[side]))
    pts = xy[idx][score[idx] >= min_score]
    if len(pts) < min_points:
        return None
    lo, hi = pts.min(0), pts.max(0)
    extent = float(max((hi - lo).max(), 1.0))
    side_px = max(extent * (1 + 2 * pad), min_side_px)
    c = (lo + hi) / 2
    return HandBox(side, float(c[0]), float(c[1]), side_px, pts, extent * 0.18)


def crop_camera(cam: OrthoCamera, box: HandBox, size: int) -> OrthoCamera:
    """Orthographic camera that sees exactly the crop: same axes, centre moved in the image plane."""
    pix = 2 * cam.half_size / cam.size
    u = box.cx * pix - cam.half_size
    v = cam.half_size - box.cy * pix
    c = np.asarray(cam.center, float) + u * np.asarray(cam.right) + v * np.asarray(cam.up)
    return replace(cam, name=f"{cam.name}_hand_{box.side}", center=tuple(c.tolist()),
                   half_size=box.side_px * pix / 2, size=int(size))


def region_mask(box: HandBox, shape: tuple[int, int]) -> np.ndarray:
    """Full-image mask of the hand region: union of discs around the keypoints and their convex hull."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    r = box.radius_px
    for x, y in box.points:
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    if len(box.points) >= 3:
        from scipy.spatial import ConvexHull

        try:
            hull = box.points[ConvexHull(box.points).vertices]
            d.polygon([tuple(p) for p in hull], fill=255)
        except Exception:  # degenerate (collinear) keypoints: discs only
            pass
    return np.asarray(img) > 0


def crop_image(rgba: Image.Image, box: HandBox, size: int, isolate: bool = True) -> Image.Image:
    """The crop, resized to size x size; isolate=True keeps only the hand region's pixels."""
    im = rgba.convert("RGBA")
    if isolate:
        a = np.asarray(im).copy()
        a[..., 3] = np.where(region_mask(box, a.shape[:2]), a[..., 3], 0)
        im = Image.fromarray(a, "RGBA")
    h = box.side_px / 2
    crop = im.crop((round(box.cx - h), round(box.cy - h), round(box.cx + h), round(box.cy + h)))
    return crop.resize((size, size), Image.LANCZOS)


def depth_of_region(mesh: Mesh, cam: OrthoCamera, mask: np.ndarray, mode: str = "median") -> float | None:
    """Depth (along cam.toward, from cam.center) of the mesh vertices that project inside the mask."""
    p = cam.project(mesh.vertices)
    x, y = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    ok[ok] = mask[y[ok], x[ok]]
    if ok.sum() < 3:
        return None
    return float(np.median(p[ok, 2]) if mode == "median" else p[ok, 2].max())


def place_part(raw: Mesh, crop_rgba: Image.Image, crop_cam: OrthoCamera, body: Mesh, full_cam: OrthoCamera,
               full_region: np.ndarray) -> tuple[Mesh, dict]:
    """Place a close-up generation: scale and X/Y from the crop silhouette (place_front_only with the crop
    camera), then depth shifted so the part's median depth over the hand region matches the body's."""
    mask = np.asarray(crop_rgba)[..., 3] > 127
    part, info = place_front_only(raw, mask, crop_cam)
    target = depth_of_region(body, full_cam, full_region)
    own = depth_of_region(part, full_cam, full_region)
    shift = 0.0 if target is None or own is None else target - own
    part = Mesh(part.vertices + shift * np.asarray(full_cam.toward, float), part.faces, part.colors)
    info.update(depth_shift=shift, depth_source="body mesh median over the hand region" if target is not None
                else "none: body has no surface under the hand region; depth left at the camera centre",
                label="ESTIMATED")
    return part, info


def merge(body: Mesh, parts: list[tuple[Mesh, np.ndarray]], cam: OrthoCamera, depth_margin: float = 0.03
          ) -> tuple[Mesh, dict]:
    """Remove body faces whose vertices all project inside a part's region and lie in its depth band
    (+- depth_margin), then add the parts. Triangle soup: not watertight."""
    p = cam.project(body.vertices)
    x, y = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    drop = np.zeros(len(body.vertices), bool)
    removed = []
    for part, region in parts:
        d = cam.project(part.vertices)[:, 2]
        lo, hi = d.min() - depth_margin, d.max() + depth_margin
        ok = (x >= 0) & (y >= 0) & (x < region.shape[1]) & (y < region.shape[0])
        inside = np.zeros(len(drop), bool)
        inside[ok] = region[y[ok], x[ok]]
        hit = inside & (p[:, 2] >= lo) & (p[:, 2] <= hi)
        removed.append(int(hit.sum()))
        drop |= hit
    keep_faces = ~drop[body.faces].all(1)
    verts, faces, off = [body.vertices], [body.faces[keep_faces]], len(body.vertices)
    for part, _ in parts:
        verts.append(part.vertices)
        faces.append(part.faces + off)
        off += len(part.vertices)
    out = Mesh(np.vstack(verts), np.vstack(faces))
    return out, {"body_vertices_removed": removed, "body_faces_removed": int((~keep_faces).sum()),
                 "watertight": False, "label": "ESTIMATED",
                 "method": "region + depth-band cut of the body, parts added as separate shells"}


Generate = Callable[[Path, Path], Path]   # (input png, output dir) -> raw mesh file


def run(front_png, front_cam: OrthoCamera, keypoints: dict, generate: Generate, out_dir,
        crop_size: int = 1024, load=None) -> dict:
    """One person: body + both hands. keypoints = {"xy": (133,2), "score": (133,)} in front_png pixels.
    Writes a new run folder with teacher_placed.glb and teacher.json; returns the report."""
    from ..export.writers import to_glb
    from .mesh_io import load_mesh

    load = load or load_mesh
    out = Path(out_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True)
    img = Image.open(front_png).convert("RGBA")
    if img.size != (front_cam.size, front_cam.size):
        raise ValueError(f"image {img.size} does not match the front camera size {front_cam.size}")
    rep = {"run_dir": str(out), "inputs": ["front image", "front camera", "2D keypoints of the front image"],
           "uses_ground_truth": False, "steps": {}}
    t0 = time.perf_counter()
    body_raw = load(generate(Path(front_png), out / "body"))
    body, binfo = place_front_only(body_raw, np.asarray(img)[..., 3] > 127, front_cam)
    rep["steps"]["body"] = {**binfo, "seconds": time.perf_counter() - t0}
    parts = []
    for side in ("l", "r"):
        box = hand_box(keypoints["xy"], keypoints["score"], side)
        if box is None:
            rep["steps"][f"hand_{side}"] = {"status": "skipped", "reason": "too few confident hand keypoints"}
            continue
        t1 = time.perf_counter()
        crop = crop_image(img, box, crop_size)
        crop.save(out / f"hand_{side}.png")
        ccam = crop_camera(front_cam, box, crop_size)
        region = region_mask(box, (img.size[1], img.size[0]))
        part, pinfo = place_part(load(generate(out / f"hand_{side}.png", out / f"hand_{side}")), crop, ccam,
                                 body, front_cam, region)
        parts.append((part, region))
        rep["steps"][f"hand_{side}"] = {"status": "success", "box": box.to_dict(), "crop_camera": ccam.to_dict(),
                                        **pinfo, "seconds": time.perf_counter() - t1}
    (out / "body_placed.glb").write_bytes(to_glb(body, "omni_body", {"label": "ESTIMATED",
                                                                      "pipeline": "front_only body"}))
    merged, minfo = merge(body, parts, front_cam)
    rep["steps"]["merge"] = minfo
    rep["seconds"] = time.perf_counter() - t0
    (out / "teacher_placed.glb").write_bytes(to_glb(merged, "omni_teacher", {"label": "ESTIMATED",
                                                                             "pipeline": "omni teacher v0"}))
    (out / "teacher.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def triposg_generate(profile: str = "quality", seed: int = 42, **backend_kw) -> Generate:
    """Generate callable for run(): TripoSG on its raw (undecimated) output. A failed or blocked generation
    raises, so a run never continues with a substitute mesh."""
    from .backends.triposg import TripoSGBackend

    be = TripoSGBackend(**backend_kw)

    def gen(image: Path, out: Path) -> Path:
        rec = be.generate({"front": image}, quality_profile=profile, out_dir=out, seed=seed)
        raw = Path(out) / "raw_triposg.glb"
        if rec.status != "success" or not raw.is_file():
            raise RuntimeError(f"TripoSG {rec.status}: {rec.reason or 'no raw output'} ({image})")
        return raw

    return gen


def rtmw_keypoints(out_dir, **step_kw) -> Callable[[Path], dict]:
    """keypoints_for callable from the anatomy step (RTMW whole-body on the PC's WSL env); raises when the
    step is blocked or fails, so no person is assembled from missing keypoints."""
    from .anatomy import AnatomyStep

    step = AnatomyStep(**step_kw)

    def kp(image: Path) -> dict:
        r = step.run(image, Path(out_dir) / Path(image).parent.parent.name)
        if r.get("status") != "success":
            raise RuntimeError(f"anatomy step {r.get('status')}: {r.get('reason')}")
        k = r["keypoints_2d"]
        return {"xy": np.asarray(k["xy"], float), "score": np.asarray(k["score"], float)}

    return kp


def score(data_dir, bench_dir) -> dict:
    """hand and body scores for the teacher and its same-input whole-body baseline (human_v1 metrics)."""
    from .human_v1 import body_scores, hand_scores, summarize

    out = {}
    for proto in ("body_front_only_hires", "teacher_front_only"):
        h = hand_scores(bench_dir, data_dir, proto)
        out[proto] = {"hands": h, "summary": {v: summarize(h, v) for v in ("placed", "local", "noise_floor")},
                      "body": body_scores(bench_dir, data_dir, proto)}
    return out


def run_human_v1(data_dir, bench_dir, generate: Generate, keypoints_for: Callable[[Path], dict],
                 size: int = 2048, names=None, load=None) -> dict:
    """Teacher on every human_v1 person from a size x size front render (human_v1.render_front).
    keypoints_for(image) -> {"xy", "score"} from that image only (the RTMW anatomy step on the PC).
    Copies body_placed.glb / teacher_placed.glb into bench_dir/<person>/ for human_v1.hand_scores."""
    import shutil

    from .human_v1 import render_front

    data_dir, bench_dir = Path(data_dir), Path(bench_dir)
    out = {"size": size, "people": {}}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "gt.glb").is_file()):
        if names and d.name not in names:
            continue
        png, cam = render_front(d, size)
        dest = bench_dir / d.name
        try:
            rep = run(png, cam, keypoints_for(png), generate, dest / "teacher_runs", load=load)
        except Exception as e:  # recorded per person; the batch goes on
            out["people"][d.name] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
            continue
        for f in ("body_placed.glb", "teacher_placed.glb"):
            shutil.copy2(Path(rep["run_dir"]) / f, dest / f)
        out["people"][d.name] = {"status": "success", "run_dir": rep["run_dir"], "seconds": rep["seconds"]}
    (bench_dir / "teacher_batch.json").write_text(json.dumps(out, indent=1))
    return out
