"""Omni teacher pipeline, step 1: hand close-ups assembled into the whole body, single image, no GT.

  front image + front camera
    -> whole-body generation, placed with align.place_front_only           (body, ESTIMATED)
    -> hand boxes from the 2D whole-body keypoints (RTMW, COCO-WholeBody)   (AI_GENERATED)
    -> per hand: crop, isolate the hand, generate, place the part with the crop camera, depth matched to
       the body's surface in the same pixels                                 (ESTIMATED)
       isolation "contour" (default): the image's own silhouette inside the dilated keypoint hull, cut at
       the wrist line; "band": contour within reach of the 2D hand skeleton (hands over torso or face);
       "discs" (v0, kept for reproducibility): discs around the keypoints plus their hull
    -> merge: body faces under the hand region and within the part's depth band are removed, the part is
       added. The result is a triangle soup, not watertight (reported, not hidden).

Outputs per run: body_placed.glb (the same-input baseline: whole body only) and teacher_placed.glb.
Every input comes from the one front image and its camera: generation is a callable, so the same code runs
TripoSG on the RTX 3060 and a stand-in in tests. Nothing here reads GT; scoring is human_v1.hand_scores.
Limits: orthographic cameras only; the depth match uses the body mesh, so a body arm placed at the wrong
depth carries the hand with it unless depth="skeleton" (MHR hand-minus-torso depth from a 2D-only fit,
ESTIMATED). human_v1 people are MHR bodies, so skeleton placement is measured on data that favours it.
MHR finger priors are not used yet.
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
    wrist: tuple[float, float] | None = None      # hand keypoint 0, when confident
    knuckle: tuple[float, float] | None = None    # hand keypoint 9 (middle finger MCP), when confident
    kp21: np.ndarray | None = None                # (21, 2) all hand keypoints in COCO-WholeBody hand order
    ok21: np.ndarray | None = None                # (21,) which of them are confident

    def to_dict(self) -> dict:
        return {"side": self.side, "cx": self.cx, "cy": self.cy, "side_px": self.side_px,
                "keypoints_used": int(len(self.points)), "radius_px": self.radius_px,
                "wrist": self.wrist, "knuckle": self.knuckle}


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
    at = lambda i: tuple(map(float, xy[idx[i]])) if score[idx[i]] >= min_score else None  # noqa: E731
    return HandBox(side, float(c[0]), float(c[1]), side_px, pts, extent * 0.18, at(0), at(9),
                   xy[idx].copy(), score[idx] >= min_score)


def crop_camera(cam: OrthoCamera, box: HandBox, size: int) -> OrthoCamera:
    """Orthographic camera that sees exactly the crop: same axes, centre moved in the image plane."""
    pix = 2 * cam.half_size / cam.size
    u = box.cx * pix - cam.half_size
    v = cam.half_size - box.cy * pix
    c = np.asarray(cam.center, float) + u * np.asarray(cam.right) + v * np.asarray(cam.up)
    return replace(cam, name=f"{cam.name}_hand_{box.side}", center=tuple(c.tolist()),
                   half_size=box.side_px * pix / 2, size=int(size))


def disc_mask(box: HandBox, shape: tuple[int, int]) -> np.ndarray:
    """v0 isolation: union of discs around the keypoints and their convex hull (full-image pixels)."""
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


def wrist_halfplane(box: HandBox, shape: tuple[int, int], behind: float = 0.15) -> np.ndarray | None:
    """Pixels on the fingers' side of the line through the wrist, perpendicular to wrist -> middle knuckle,
    moved back by `behind` of that length so the wrist itself stays. None without both keypoints."""
    if box.wrist is None or box.knuckle is None:
        return None
    w, k = np.asarray(box.wrist), np.asarray(box.knuckle)
    axis = k - w
    n = float(np.linalg.norm(axis))
    if n < 1.0:
        return None
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    s = ((xx + 0.5 - w[0]) * axis[0] + (yy + 0.5 - w[1]) * axis[1]) / n
    return s >= -behind * n


# hand order: 0 wrist, 1-4 thumb CMC..tip, 5-8 index MCP..tip, 9-12 middle, 13-16 ring, 17-20 pinky
FINGER_CHAINS = [[0, 1, 2, 3, 4]] + [[b, b + 1, b + 2, b + 3] for b in (5, 9, 13, 17)]
PALM = [0, 1, 5, 9, 13, 17]


def hand_band(box: HandBox, shape: tuple[int, int], width: float = 0.15) -> np.ndarray | None:
    """Pixels near the 2D hand skeleton: finger bones and the palm polygon (with the wrist widened across
    the hand axis), dilated by width x the wrist -> middle-knuckle length. Separates a hand from the torso
    or face it overlaps in the image, where the alpha cannot. None without wrist and middle knuckle."""
    if box.kp21 is None or box.wrist is None or box.knuckle is None:
        return None
    from scipy import ndimage

    w, k = np.asarray(box.wrist), np.asarray(box.knuckle)
    length = float(np.linalg.norm(k - w))
    if length < 2.0:
        return None
    perp = np.array([-(k - w)[1], (k - w)[0]]) / length
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    palm = [tuple(box.kp21[i]) for i in PALM if box.ok21[i]]
    palm += [tuple(w + 0.35 * length * perp), tuple(w - 0.35 * length * perp)]
    from scipy.spatial import ConvexHull

    try:
        pts = np.array(palm)
        d.polygon([tuple(p) for p in pts[ConvexHull(pts).vertices]], fill=255)
    except Exception:  # degenerate palm: bones only
        pass
    for ch in FINGER_CHAINS:
        for a, b in zip(ch[:-1], ch[1:]):
            if box.ok21[a] and box.ok21[b]:
                d.line([tuple(box.kp21[a]), tuple(box.kp21[b])], fill=255, width=1)
    bones = np.asarray(img) > 0
    return ndimage.distance_transform_edt(~bones) <= width * length


def region_mask(box: HandBox, shape: tuple[int, int], alpha: np.ndarray | None = None,
                mode: str = "contour") -> np.ndarray:
    """Full-image mask of the hand region.
    contour: alpha > 127 inside the keypoint hull dilated by radius_px, on the fingers' side of the wrist
    line, keeping only the connected pieces that hold a keypoint. Falls back to the discs (and says so via
    isolation_info) when no alpha is given. discs: disc_mask (v0). band: contour restricted to hand_band,
    for hands over the torso or face (contour alone when the band cannot be drawn)."""
    if mode == "discs" or alpha is None:
        return disc_mask(box, shape)
    if mode not in ("contour", "band"):
        raise ValueError(f"unknown isolation mode {mode!r}")
    from scipy import ndimage

    hull = disc_mask(replace(box, radius_px=0.0), shape)
    if not hull.any():
        hull = disc_mask(box, shape)
    near = ndimage.distance_transform_edt(~hull) <= box.radius_px
    m = near & (np.asarray(alpha) > 127)
    cut = wrist_halfplane(box, shape)
    if cut is not None:
        m &= cut
    if mode == "band":
        band = hand_band(box, shape)
        if band is not None:
            m &= band
    lab, n = ndimage.label(m)
    if n > 1:
        x = np.clip(box.points[:, 0].astype(int), 0, shape[1] - 1)
        y = np.clip(box.points[:, 1].astype(int), 0, shape[0] - 1)
        hit = set(lab[y, x].tolist()) - {0}
        if hit:
            m = np.isin(lab, list(hit))
    return m


def isolation_info(box: HandBox, mode: str, alpha_given: bool) -> dict:
    if mode == "discs" or not alpha_given:
        return {"isolation": "discs", "reason": "requested" if mode == "discs" else "no alpha channel"}
    out = {"isolation": mode, "wrist_cut": box.wrist is not None and box.knuckle is not None}
    if mode == "band":
        out["band_used"] = box.kp21 is not None and out["wrist_cut"]
    return out


def crop_image(rgba: Image.Image, box: HandBox, size: int, isolate: bool = True, region: np.ndarray | None = None
               ) -> Image.Image:
    """The crop, resized to size x size; isolate=True keeps only the region's pixels (default: discs)."""
    im = rgba.convert("RGBA")
    if isolate:
        a = np.asarray(im).copy()
        keep = region if region is not None else disc_mask(box, a.shape[:2])
        a[..., 3] = np.where(keep, a[..., 3], 0)
        im = Image.fromarray(a, "RGBA")
    h = box.side_px / 2
    crop = im.crop((round(box.cx - h), round(box.cy - h), round(box.cx + h), round(box.cy + h)))
    return crop.resize((size, size), Image.LANCZOS)


def depth_of_region(mesh: Mesh, cam: OrthoCamera, mask: np.ndarray, mode: str = "median") -> float | None:
    """Depth (along cam.toward, from cam.center) of the mesh vertices that project inside the mask:
    median, max, or mid (halfway between the 5th and 95th percentiles: the mid-plane of a closed shell,
    independent of how densely its front and back are meshed)."""
    p = cam.project(mesh.vertices)
    x, y = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    ok[ok] = mask[y[ok], x[ok]]
    if ok.sum() < 3:
        return None
    d = p[ok, 2]
    if mode == "mid":
        return float(np.percentile(d, 5) + np.percentile(d, 95)) / 2
    return float(np.median(d) if mode == "median" else d.max())


TORSO_KEYPOINTS = {5: "l_uparm", 6: "r_uparm", 11: "l_upleg", 12: "r_upleg"}   # shoulders and hips


@dataclass(frozen=True)
class Skeleton:
    """The MHR fit of the anatomy step, in its own orthographic frame: px = (x, -y) * scale + shift.
    joints_cm: (J, 3) with the person's facing applied, +Z toward the camera. ESTIMATED: a 2D-only fit,
    its depth comes from bone lengths and priors, not from a depth measurement."""
    names: tuple[str, ...]
    joints_cm: np.ndarray
    scale: float            # image pixels per cm

    def joint(self, name: str) -> np.ndarray:
        return self.joints_cm[self.names.index(name)]

    def hand_centre(self, side: str) -> np.ndarray:
        return self.joints_cm[[i for i, n in enumerate(self.names) if n.startswith(f"{side}_") and
                               any(f in n for f in ("wrist", "thumb", "index", "middle", "ring", "pinky"))]].mean(0)


def skeleton_depth(sk: Skeleton, side: str, keypoints: dict, body: Mesh, cam: OrthoCamera,
                   min_score: float = 0.3) -> tuple[float | None, dict]:
    """Target depth (along cam.toward) of the hand's centre: the body's depth over the torso (hull of the
    shoulder and hip keypoints) plus the skeleton's hand-minus-torso depth, cm -> m through the fit's scale
    and the camera's pixel size. Torso and part depths are mid-planes (depth_of_region "mid").
    None (with the reason) when the torso is not seen."""
    xy, sc = np.asarray(keypoints["xy"], float), np.asarray(keypoints["score"], float)
    ok = [k for k in TORSO_KEYPOINTS if sc[k] >= min_score]
    if len(ok) < 3:
        return None, {"reason": f"only {len(ok)} confident torso keypoints"}
    torso = disc_mask(HandBox("torso", 0, 0, 1, xy[ok], 0.0), (cam.size, cam.size))
    body_d = depth_of_region(body, cam, torso, "mid")
    if body_d is None:
        return None, {"reason": "body has no surface under the torso keypoints"}
    m_per_cm = (2 * cam.half_size / cam.size) * sk.scale
    dz_cm = float(sk.hand_centre(side)[2] - np.mean([sk.joint(TORSO_KEYPOINTS[k])[2] for k in ok]))
    # both the fit's +Z and cam.toward point at the viewer, so a nearer hand has a larger depth
    return body_d + dz_cm * m_per_cm, {"torso_depth": body_d, "hand_minus_torso_cm": dz_cm,
                                       "m_per_cm": m_per_cm}


MAX_DEPTH_RATIO = 1.0   # depth extent / largest image-plane extent of a close-up (hand ~0.15-0.5, plus a forearm stub)


def tidy_part(part: Mesh, cam: OrthoCamera, max_ratio: float = MAX_DEPTH_RATIO, min_fraction: float = 0.05
              ) -> tuple[Mesh, dict]:
    """Close-up generations of a hand come back with thin plates scattered in depth (ROPY-PC smoke test,
    TEACHER_V2_SKELETON.md: placed hand depth median 0.36 m, up to 1.7 m). Keep the pieces with at least
    min_fraction of the faces, then, if the depth extent is still more than max_ratio x the largest
    image-plane extent, squash depth about its median to that ratio. Both are reported."""
    from ..reconstruct.fit import drop_small_parts

    before = np.ptp(cam.project(part.vertices)[:, 2])
    part, removed = drop_small_parts(part, min_fraction)
    p = cam.project(part.vertices)
    pix = 2 * cam.half_size / cam.size
    planar = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * pix)
    depth = float(np.ptp(p[:, 2]))
    info = {"pieces_removed": removed, "depth_extent_raw": float(before), "depth_extent_after_cleanup": depth,
            "planar_extent": planar, "max_depth_ratio": max_ratio, "depth_squash": 1.0}
    if planar > 0 and depth > max_ratio * planar:
        k = max_ratio * planar / depth
        toward = np.asarray(cam.toward, float)
        d = p[:, 2] - np.median(p[:, 2])
        part = Mesh(part.vertices - np.outer(d * (1 - k), toward), part.faces, part.colors)
        info["depth_squash"] = k
    info["depth_extent"] = float(np.ptp(cam.project(part.vertices)[:, 2]))
    info["label"] = "ESTIMATED (thickness limit is an assumption about hands, not a measurement)"
    return part, info


def place_part(raw: Mesh, crop_rgba: Image.Image, crop_cam: OrthoCamera, body: Mesh, full_cam: OrthoCamera,
               full_region: np.ndarray, target_depth: float | None = None, target_info: dict | None = None,
               core: np.ndarray | None = None, tidy: bool = True) -> tuple[Mesh, dict]:
    """Place a close-up generation: scale and X/Y from the crop silhouette (place_front_only with the crop
    camera), then depth shifted so the part's median depth over the hand region matches target_depth
    (skeleton) or, without it, the body's median depth over the same region (v0). With a target, the part's
    own depth is its mid-plane over `core` (the keypoint neighbourhood, default the region), so an arm stub
    that the region kept does not pull the hand's centre. tidy: tidy_part before the depth match."""
    mask = np.asarray(crop_rgba)[..., 3] > 127
    part, info = place_front_only(raw, mask, crop_cam)
    if tidy:
        part, info["tidy"] = tidy_part(part, full_cam)
    if target_depth is not None:
        target, source = target_depth, "skeleton: body torso depth + MHR hand-minus-torso depth"
    else:
        target = depth_of_region(body, full_cam, full_region)
        source = ("body mesh median over the hand region" if target is not None
                  else "none: body has no surface under the hand region; depth left at the camera centre")
    if target_depth is not None:
        own = depth_of_region(part, full_cam, full_region if core is None else core, "mid")
    else:
        own = depth_of_region(part, full_cam, full_region)
    shift = 0.0 if target is None or own is None else target - own
    part = Mesh(part.vertices + shift * np.asarray(full_cam.toward, float), part.faces, part.colors)
    info.update(depth_shift=shift, depth_source=source, label="ESTIMATED")
    if target_info:
        info["skeleton_depth"] = target_info
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
        crop_size: int = 1024, load=None, isolation: str = "contour", body: Mesh | None = None,
        depth: str = "body", progress: Callable[[str], None] | None = None, tidy: bool = True) -> dict:
    """One person: body + both hands. keypoints = {"xy": (133,2), "score": (133,)} in front_png pixels,
    plus "skeleton" (a Skeleton) for depth="skeleton"; without it the hand falls back to the body depth.
    body: an already placed whole-body mesh from the same image (reused so that one change is measured
    at a time); None generates it. progress(message) is told before each generation.
    Writes a new run folder with teacher_placed.glb and teacher.json."""
    from ..export.writers import to_glb
    from .mesh_io import load_mesh

    load = load or load_mesh
    out = Path(out_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True)
    img = Image.open(front_png).convert("RGBA")
    if img.size != (front_cam.size, front_cam.size):
        raise ValueError(f"image {img.size} does not match the front camera size {front_cam.size}")
    rep = {"run_dir": str(out), "inputs": ["front image", "front camera", "2D keypoints of the front image"],
           "uses_ground_truth": False, "isolation": isolation, "depth": depth, "tidy": tidy, "steps": {}}
    if depth not in ("body", "skeleton"):
        raise ValueError(f"unknown depth mode {depth!r}")
    t0 = time.perf_counter()
    alpha = np.asarray(img)[..., 3]
    if body is None:
        body_raw = load(generate(Path(front_png), out / "body"))
        body, binfo = place_front_only(body_raw, alpha > 127, front_cam)
        rep["steps"]["body"] = {**binfo, "seconds": time.perf_counter() - t0}
    else:
        rep["steps"]["body"] = {"status": "reused", "reason": "placed body from an earlier run of the same image"}
    parts = []
    for side in ("l", "r"):
        box = hand_box(keypoints["xy"], keypoints["score"], side)
        if box is None:
            rep["steps"][f"hand_{side}"] = {"status": "skipped", "reason": "too few confident hand keypoints"}
            continue
        t1 = time.perf_counter()
        if progress:
            progress(f"close-up of the {'left' if side == 'l' else 'right'} hand")
        region = region_mask(box, alpha.shape, alpha, isolation)
        crop = crop_image(img, box, crop_size, region=region)
        crop.save(out / f"hand_{side}.png")
        ccam = crop_camera(front_cam, box, crop_size)
        target, tinfo = None, None
        if depth == "skeleton":
            sk = keypoints.get("skeleton")
            target, tinfo = (skeleton_depth(sk, side, keypoints, body, front_cam) if sk is not None
                             else (None, {"reason": "no skeleton from the anatomy step"}))
            if target is None:
                tinfo["fallback"] = "body depth over the hand region"
        part, pinfo = place_part(load(generate(out / f"hand_{side}.png", out / f"hand_{side}")), crop, ccam,
                                 body, front_cam, region, target, tinfo,
                                 core=region & disc_mask(box, region.shape), tidy=tidy)
        parts.append((part, region))
        rep["steps"][f"hand_{side}"] = {"status": "success", "box": box.to_dict(), "crop_camera": ccam.to_dict(),
                                        **isolation_info(box, isolation, True), **pinfo, "seconds": time.perf_counter() - t1}
    (out / "body_placed.glb").write_bytes(to_glb(body, "omni_body", {"label": "ESTIMATED",
                                                                      "pipeline": "front_only body"}))
    merged, minfo = merge(body, parts, front_cam)
    rep["steps"]["merge"] = minfo
    rep["seconds"] = time.perf_counter() - t0
    (out / "teacher_placed.glb").write_bytes(to_glb(merged, "omni_teacher", {"label": "ESTIMATED",
                                                                             "pipeline": "omni teacher",
                                                                             "isolation": isolation}))
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
        out = {"xy": np.asarray(k["xy"], float), "score": np.asarray(k["score"], float)}
        fit = Path(r["run_dir"]) / "fit.npz"
        if fit.is_file():
            f = np.load(fit)
            out["skeleton"] = Skeleton(tuple(str(n) for n in f["joint_names"]), np.asarray(f["joints_cm"], float),
                                       float(np.asarray(f["scale"]).reshape(-1)[0]))
        return out

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
                 size: int = 2048, names=None, load=None, isolation: str = "contour", reuse_body=None,
                 depth: str = "body", tidy: bool = True) -> dict:
    """Teacher on every human_v1 person from a size x size front render (human_v1.render_front).
    keypoints_for(image) -> {"xy", "score"} from that image only (the RTMW anatomy step on the PC).
    Copies body_placed.glb / teacher_placed.glb into bench_dir/<person>/ for human_v1.hand_scores.
    reuse_body: an earlier bench_dir; its <person>/body_placed.glb is reused instead of generating the body."""
    import shutil

    from .human_v1 import render_front

    data_dir, bench_dir = Path(data_dir), Path(bench_dir)
    from .mesh_io import load_mesh

    out = {"size": size, "isolation": isolation, "depth": depth, "tidy": tidy, "reuse_body": str(reuse_body) if reuse_body else None,
           "people": {}}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "gt.glb").is_file()):
        if names and d.name not in names:
            continue
        png, cam = render_front(d, size)
        dest = bench_dir / d.name
        try:
            body = load_mesh(Path(reuse_body) / d.name / "body_placed.glb") if reuse_body else None
            rep = run(png, cam, keypoints_for(png), generate, dest / "teacher_runs", load=load,
                      isolation=isolation, body=body, depth=depth, tidy=tidy)
        except Exception as e:  # recorded per person; the batch goes on
            out["people"][d.name] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
            continue
        for f in ("body_placed.glb", "teacher_placed.glb"):
            shutil.copy2(Path(rep["run_dir"]) / f, dest / f)
        out["people"][d.name] = {"status": "success", "run_dir": rep["run_dir"], "seconds": rep["seconds"]}
    (bench_dir / "teacher_batch.json").write_text(json.dumps(out, indent=1))
    return out
