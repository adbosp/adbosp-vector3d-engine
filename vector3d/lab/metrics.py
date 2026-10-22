"""Geometry and mesh-quality metrics with documented definitions.

Chamfer distance and F-score are computed on area-uniform surface samples
in the units of the inputs. Callers normalise both meshes first (see
align.py) so thresholds are fractions of the ground-truth bounding-box
diagonal.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..reconstruct.surface import Mesh, validate


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def sample_surface(mesh: Mesh, n: int = 20000, seed: int = 0) -> np.ndarray:
    v = mesh.vertices[mesh.faces]
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) / 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(area), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.column_stack([1 - s, s * (1 - r2), s * r2])
    return np.einsum("ij,ijk->ik", w, v[idx])


def chamfer_fscore(pred: Mesh, gt: Mesh, thresholds=(0.01, 0.02, 0.05), n: int = 20000, seed: int = 0) -> dict:
    """Symmetric mean Chamfer-L1 (mean of the two directed means) and F-score.

    Thresholds are fractions of the GT bounding-box diagonal.
    """
    p, g = sample_surface(pred, n, seed), sample_surface(gt, n, seed + 1)
    diag = float(np.linalg.norm(gt.vertices.max(0) - gt.vertices.min(0)))
    d_pg = cKDTree(g).query(p)[0] / diag
    d_gp = cKDTree(p).query(g)[0] / diag
    out = {"chamfer_l1": float((d_pg.mean() + d_gp.mean()) / 2), "units": "fraction of GT bbox diagonal",
           "samples": n}
    for t in thresholds:
        prec, rec = float((d_pg < t).mean()), float((d_gp < t).mean())
        out[f"fscore@{t}"] = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return out


def normal_consistency(mesh: Mesh) -> float:
    """Mean cosine between normals of faces sharing an edge (1 = perfectly smooth)."""
    v = mesh.vertices[mesh.faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    f = mesh.faces
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k, fo = key[order], fid[order]
    same = (k[1:] == k[:-1]).all(axis=1)
    if not same.any():
        return float("nan")
    return float(np.einsum("ij,ij->i", fn[fo[:-1][same]], fn[fo[1:][same]]).mean())


def mesh_quality(mesh: Mesh) -> dict:
    q = validate(mesh)
    q["finite"] = bool(np.isfinite(mesh.vertices).all())
    q["normal_consistency"] = normal_consistency(mesh)
    q["self_intersections"] = "skipped: no exact intersection test implemented yet"
    q["quad_ratio"] = 0.0  # triangle meshes only so far
    return q
