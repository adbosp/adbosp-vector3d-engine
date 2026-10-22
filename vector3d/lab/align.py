"""Bring a generated mesh into the ground-truth frame before measuring it.

Foundation models emit meshes in their own canonical box and orientation.
We normalise scale and position from bounding boxes, then pick the axis
permutation and sign flip (24 proper rotations) with the lowest Chamfer
distance. The chosen transform is reported because it is ESTIMATED, not
observed.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from ..reconstruct.surface import Mesh
from .metrics import sample_surface


def normalize_box(mesh: Mesh, target_center=None, target_extent: float | None = None) -> tuple[Mesh, dict]:
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    c = (lo + hi) / 2
    ext = float((hi - lo).max())
    tc = np.zeros(3) if target_center is None else np.asarray(target_center, float)
    s = (target_extent or 1.0) / ext
    v = (mesh.vertices - c) * s + tc
    return Mesh(v, mesh.faces, mesh.colors), {"scale": s, "offset": (tc - c * s).tolist()}


def rotations24() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                r[i, p] = s
            if np.isclose(np.linalg.det(r), 1):
                out.append(r)
    return out


def align_to(pred: Mesh, gt: Mesh, search_rotation: bool = True, n: int = 4000) -> tuple[Mesh, dict]:
    lo, hi = gt.vertices.min(0), gt.vertices.max(0)
    gc, gext = (lo + hi) / 2, float((hi - lo).max())
    g = sample_surface(gt, n, 1)
    tree = cKDTree(g)
    best = None
    for r in (rotations24() if search_rotation else [np.eye(3)]):
        cand, info = normalize_box(Mesh(pred.vertices @ r.T, pred.faces, pred.colors), gc, gext)
        p = sample_surface(cand, n, 0)
        cd = tree.query(p)[0].mean() + cKDTree(p).query(g)[0].mean()
        if best is None or cd < best[0]:
            best = (cd, cand, r, info)
    _, mesh, r, info = best
    return mesh, {"rotation": r.tolist(), **info, "label": "ESTIMATED",
                  "method": "bbox normalisation + 24-rotation Chamfer search"}


def _mask_box(mask: np.ndarray, cam) -> tuple[float, float, float, float]:
    """World (u_min, u_max, v_min, v_max) of a mask in a camera's plane."""
    ys, xs = np.nonzero(mask)
    pix = 2 * cam.half_size / cam.size
    u = (np.array([xs.min(), xs.max() + 1]) * pix) - cam.half_size
    v = cam.half_size - (np.array([ys.max() + 1, ys.min()]) * pix)
    return u[0], u[1], v[0], v[1]


def silhouette_box(targets: dict, cameras: dict) -> tuple[np.ndarray, np.ndarray]:
    """World-space bounding box observed by the front and a side silhouette."""
    lo, hi = np.full(3, np.nan), np.full(3, np.nan)
    for view, mask in targets.items():
        if not mask.any():
            continue
        cam = cameras[view]
        u0, u1, v0, v1 = _mask_box(mask, cam)
        c = np.array(cam.center)
        for (a, b), axis in (((u0, u1), np.array(cam.right)), ((v0, v1), np.array(cam.up))):
            i = int(np.argmax(np.abs(axis)))
            if abs(axis[i]) < 0.999:
                continue  # oblique camera: not used for the box
            ends = sorted([c[i] + a * axis[i], c[i] + b * axis[i]])
            lo[i] = ends[0] if np.isnan(lo[i]) else min(lo[i], ends[0])
            hi[i] = ends[1] if np.isnan(hi[i]) else max(hi[i], ends[1])
    return lo, hi


def align_to_silhouettes(pred: Mesh, targets: dict, cameras: dict) -> tuple[Mesh, dict]:
    """Ground-truth-free placement: 24 rotations, uniform scale and offset from
    the observed silhouette box, scored by mean silhouette IoU."""
    from .metrics import iou
    from .render import silhouette

    lo, hi = silhouette_box(targets, cameras)
    if np.isnan(lo).any():
        raise ValueError("need at least two axis-aligned reference views to place the mesh")
    tc, text = (lo + hi) / 2, hi - lo
    best = None
    for r in rotations24():
        v = pred.vertices @ r.T
        plo, phi = v.min(0), v.max(0)
        s = float(np.median(text / np.maximum(phi - plo, 1e-12)))
        cand = Mesh((v - (plo + phi) / 2) * s + tc, pred.faces, pred.colors)
        score = float(np.mean([iou(silhouette(cand, cameras[k]), m) for k, m in targets.items()]))
        if best is None or score > best[0]:
            best = (score, cand, r, s)
    score, mesh, r, s = best
    return mesh, {"rotation": r.tolist(), "scale": s, "mean_ref_iou": score, "label": "ESTIMATED",
                  "method": "24 rotations + silhouette box, scored by reference IoU (no ground truth)"}
