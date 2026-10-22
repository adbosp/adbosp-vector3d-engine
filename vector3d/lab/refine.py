"""Silhouette refinement: render, compare against the observed masks, move
the contour vertices, repeat.

The observed reference silhouettes are the only input; ground truth is never
read here. Each iteration:
  1. renders the mesh from every reference camera,
  2. computes the target mask's signed distance in pixels (positive inside),
  3. moves rim vertices (normal nearly perpendicular to the view) along the
     rendered silhouette's outward normal by that signed distance, and pulls
     any vertex that projects outside the target back inside,
  4. diffuses the displacements over the mesh graph so the surface moves as
     a whole instead of growing spikes, then applies a damped step.
The best iteration by mean reference IoU is kept, and the run stops early
once the IoU stops improving. Losses are recorded per iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse

from ..reconstruct.surface import Mesh, taubin_smooth
from .metrics import iou
from .render import OrthoCamera, render


@dataclass
class RefineParams:
    iterations: int = 40
    step: float = 0.6            # fraction of the measured correction applied per iteration
    rim_cos: float = 0.35        # |normal . view| below this marks a rim vertex
    diffusion: int = 12          # graph smoothing passes on the displacement field
    max_px: float = 6.0          # cap per-iteration motion, in pixels
    patience: int = 4
    smooth: int = 3              # Taubin passes per iteration (surface regularisation)
    blur: int = 4                # graph smoothing passes over the whole displacement field


@dataclass
class RefineResult:
    mesh: Mesh
    history: list[dict] = field(default_factory=list)
    best_iteration: int = 0


def _adjacency(n: int, faces: np.ndarray) -> sparse.csr_matrix:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float) + sparse.identity(n)
    deg = np.asarray(a.sum(axis=1)).ravel()
    return sparse.diags(1 / deg) @ a


def _sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    s = img.shape[0]
    x = np.clip(xy[:, 0].astype(int), 0, s - 1)
    y = np.clip(xy[:, 1].astype(int), 0, s - 1)
    return img[y, x]


def _view_displacement(mesh: Mesh, normals: np.ndarray, cam: OrthoCamera, target: np.ndarray, rim_cos: float):
    """Per-vertex world displacement and weight for one view, plus the current IoU."""
    r = render(mesh, cam)
    cur = r.mask
    phi = ndimage.distance_transform_edt(target) - ndimage.distance_transform_edt(~target)  # +inside
    # Outward normal of the rendered silhouette: minus the gradient of its own signed distance.
    psi = ndimage.distance_transform_edt(cur) - ndimage.distance_transform_edt(~cur)
    psi = ndimage.gaussian_filter(psi, 1.0)
    gy, gx = np.gradient(psi)
    px = cam.project(mesh.vertices)
    toward, right, up = (np.array(a) for a in (cam.toward, cam.right, cam.up))

    n2 = -np.column_stack([_sample(gx, px), _sample(gy, px)])
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
    d_phi = _sample(phi, px)
    near_edge = np.abs(_sample(psi, px)) < 2.0
    rim = (np.abs(normals @ toward) < rim_cos) & near_edge

    move = np.zeros((len(px), 2))
    move[rim] = n2[rim] * d_phi[rim, None]            # grow or shrink the contour to the target
    outside = d_phi < -1.0                             # clearly outside the target silhouette
    gpy, gpx = np.gradient(ndimage.gaussian_filter(phi, 1.0))
    g = np.column_stack([_sample(gpx, px), _sample(gpy, px)])
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    move[outside & ~rim] = g[outside & ~rim] * (-d_phi[outside & ~rim, None])
    w = (rim | outside).astype(float)

    pix = 2 * cam.half_size / cam.size
    world = (move[:, :1] * right - move[:, 1:] * up) * pix  # image y points down
    return world, w, iou(cur, target)


def refine_silhouettes(mesh: Mesh, targets: dict[str, np.ndarray], cameras: dict[str, OrthoCamera],
                       params: RefineParams | None = None) -> RefineResult:
    p = params or RefineParams()
    views = list(targets)
    A = _adjacency(len(mesh.vertices), mesh.faces)
    cur = Mesh(mesh.vertices.copy(), mesh.faces, mesh.colors)
    history, best, best_score, stale = [], cur, -1.0, 0
    for it in range(p.iterations + 1):
        normals = cur.normals()
        disp = np.zeros_like(cur.vertices)
        wsum = np.zeros(len(cur.vertices))
        ious = {}
        for v in views:
            d, w, ious[v] = _view_displacement(cur, normals, cameras[v], targets[v], p.rim_cos)
            disp += d * w[:, None]
            wsum += w
        score = float(np.mean(list(ious.values())))
        history.append({"iteration": it, "mean_iou": score, "iou": ious})
        if score > best_score + 1e-5:
            best, best_score, stale = cur, score, 0
        else:
            stale += 1
        if it == p.iterations or stale >= p.patience:
            break
        disp /= np.maximum(wsum, 1)[:, None]
        wt = (wsum > 0).astype(float)
        for _ in range(p.diffusion):  # spread rim corrections over the surface
            num = A @ (disp * wt[:, None])
            den = A @ wt
            disp = np.where(wt[:, None] > 0, disp, num / np.maximum(den, 1e-9)[:, None])
            wt = np.maximum(wt, (den > 0).astype(float))
        for _ in range(p.blur):  # then low-pass everything, so neighbours move together
            disp = A @ disp
        pix = 2 * cameras[views[0]].half_size / cameras[views[0]].size
        norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp *= np.minimum(1, p.max_px * pix / np.maximum(norm, 1e-12))
        cur = taubin_smooth(Mesh(cur.vertices + p.step * disp, cur.faces, cur.colors), p.smooth)
    return RefineResult(best, history, int(np.argmax([h["mean_iou"] for h in history])))
