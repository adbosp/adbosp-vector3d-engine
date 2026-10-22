"""Fit a generated mesh (AI_GENERATED) to the views the user drew or photographed.

The views are measurements, the model's shape is a guess: the mesh is moved and
scaled per axis onto the extents the views observe, then its silhouettes are
compared with each view's silhouette on the visual-hull grid. Everything here is
plain geometry; no model is called.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..normalize.normalizer import NormalizedScene
from .hull import HullResult
from .surface import Mesh


def drop_small_parts(mesh: Mesh, min_fraction: float = 0.005) -> tuple[Mesh, int]:
    """Remove connected pieces with fewer than min_fraction of the faces (floating fragments)."""
    n = len(mesh.vertices)
    f = mesh.faces
    adj = sparse.coo_matrix((np.ones(2 * len(f)), (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])), shape=(n, n))
    k, labels = connected_components(adj, directed=False)
    if k == 1:
        return mesh, 0
    face_label = labels[f[:, 0]]
    counts = np.bincount(face_label, minlength=k)
    keep_label = counts >= max(1, min_fraction * len(f))
    keep_label[np.argmax(counts)] = True
    faces = f[keep_label[face_label]]
    used = np.zeros(n, bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    colors = mesh.colors[used] if mesh.colors is not None else None
    removed = int((counts > 0).sum() - keep_label[counts > 0].sum())
    return Mesh(mesh.vertices[used], remap[faces], colors), removed


def fit_to_extent(mesh: Mesh, ext: dict[int, tuple[float, float]]) -> tuple[Mesh, list[float]]:
    """Scale and move the mesh so its bounding box matches the observed extent on each axis.

    Axes no view observes get the mean scale of the observed ones and stay centred on 0,
    as the visual hull assumes. Returns the mesh and the per-axis scale factors."""
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    size = np.maximum(hi - lo, 1e-12)
    k = np.array([(ext[a][1] - ext[a][0]) / size[a] if a in ext else np.nan for a in range(3)])
    k[np.isnan(k)] = np.nanmean(k)
    target_c = np.array([(ext[a][0] + ext[a][1]) / 2 if a in ext else 0.0 for a in range(3)])
    v = (mesh.vertices - (lo + hi) / 2) * k + target_c
    return Mesh(v, mesh.faces, mesh.colors), k.tolist()


def view_cameras(scene: NormalizedScene, hull: HullResult, size: int = 256) -> dict:
    """One orthographic camera per view, all framing the hull grid the same way."""
    from ..lab.render import view_camera

    g = hull.grid
    lo = g.lo
    hi = g.lo + g.h * np.array(g.shape)
    center = tuple(((lo + hi) / 2).tolist())
    half = float((hi - lo).max()) / 2 * 1.05
    return {v.value: view_camera(v.value, center, half, size) for v in scene.views}


def target_masks(scene: NormalizedScene, hull: HullResult, cameras: dict) -> dict[str, np.ndarray]:
    """Each view's hull silhouette resampled at the pixel centres of its camera."""
    g = hull.grid
    out = {}
    for v, nv in scene.views.items():
        cam = cameras[v.value]
        mask = hull.masks[v.value]
        if not mask.any():
            continue
        (wu, _), (wv, _) = nv.camera.axis_of(0), nv.camera.axis_of(1)
        s, half = cam.size, cam.half_size
        t = (np.arange(s) + 0.5) / s * 2 * half - half
        u, w = np.meshgrid(t, -t)           # columns go right, rows go down
        p = np.array(cam.center) + u[..., None] * np.array(cam.right) + w[..., None] * np.array(cam.up)
        ku = np.floor((p[..., wu] - g.lo[wu]) / g.h[wu]).astype(int)
        kv = np.floor((p[..., wv] - g.lo[wv]) / g.h[wv]).astype(int)
        ok = (ku >= 0) & (ku < mask.shape[0]) & (kv >= 0) & (kv < mask.shape[1])
        img = np.zeros((s, s), bool)
        img[ok] = mask[ku[ok], kv[ok]]
        out[v.value] = img
    return out


def view_ious(mesh: Mesh, targets: dict[str, np.ndarray], cameras: dict) -> dict[str, float]:
    from ..lab.metrics import iou
    from ..lab.render import silhouette

    return {v: float(iou(silhouette(mesh, cameras[v]), t)) for v, t in targets.items()}
