"""Occupancy grid -> triangle mesh, smoothing and mesh validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure

from .hull import Grid


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float
    faces: np.ndarray     # (M, 3) int, counter-clockwise seen from outside
    colors: np.ndarray | None = None  # (N, 3) float in [0, 1]

    def normals(self) -> np.ndarray:
        v, f = self.vertices, self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        n = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.where(n == 0, 1, n)

    def volume(self) -> float:
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6)


def occupancy_to_mesh(occ: np.ndarray, grid: Grid) -> Mesh:
    if not occ.any():
        raise ValueError("carved volume is empty: the views do not overlap")
    vol = np.pad(occ.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    verts = grid.lo + (verts - 1 + 0.5) * grid.h  # index space -> world, undoing the padding
    mesh = Mesh(verts, faces.astype(np.int64))
    if mesh.volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    return mesh


def field_to_mesh(field: np.ndarray, grid: Grid) -> Mesh:
    """Signed field (positive inside) -> mesh at the zero level, with sub-voxel accuracy."""
    if not (field > 0).any():
        raise ValueError("reconstructed volume is empty")
    vol = np.pad(field, 1, constant_values=-float(grid.h.max()))
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0)
    verts = grid.lo + (verts - 1 + 0.5) * grid.h
    mesh = Mesh(verts, faces.astype(np.int64))
    if mesh.volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    return mesh


def _neighbors(n: int, faces: np.ndarray):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(np.concatenate([e, e[:, ::-1]]), axis=1), axis=0)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    return e, deg


def taubin_smooth(mesh: Mesh, iterations: int = 10, lam: float = 0.5, mu: float = -0.53) -> Mesh:
    """Volume-preserving smoothing that removes voxel stair-steps."""
    if iterations <= 0:
        return mesh
    v = mesh.vertices.copy()
    e, deg = _neighbors(len(v), mesh.faces)
    deg = np.where(deg == 0, 1, deg)[:, None]
    for _ in range(iterations):
        for factor in (lam, mu):
            acc = np.zeros_like(v)
            np.add.at(acc, e[:, 0], v[e[:, 1]])
            np.add.at(acc, e[:, 1], v[e[:, 0]])
            v = v + factor * (acc / deg - v)
    return Mesh(v, mesh.faces, mesh.colors)


def validate(mesh: Mesh) -> dict:
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    v = mesh.vertices[f]
    area = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) / 2
    n_v, n_e, n_f = len(mesh.vertices), len(counts), len(f)
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return {
        "vertices": n_v,
        "triangles": n_f,
        "watertight": bool((counts == 2).all()),
        "non_manifold_edges": int((counts > 2).sum()),
        "boundary_edges": int((counts == 1).sum()),
        "degenerate_triangles": int((area < 1e-12 * max(area.max(), 1e-30)).sum()),
        "euler_characteristic": int(n_v - n_e + n_f),
        "volume": mesh.volume(),
        "bbox": [lo.tolist(), hi.tolist()],
    }
