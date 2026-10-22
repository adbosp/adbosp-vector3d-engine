"""Photo reconstruction: a closed solid from one image's mask and AI depth.

The front surface follows the estimated depth, modulated by a rounded
silhouette inflation so it meets the back surface exactly at the outline.
The back surface is not visible in a photo; it is mirrored from the front
(`back="symmetric"`) or flattened (`back="flat"`). Everything here is
AI_GENERATED / ESTIMATED shape, never measurement.

The solid is returned as a signed field (positive inside, world units) on
the visual-hull grid, so other views' silhouettes still carve it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..ai.depth import inflation
from ..normalize.normalizer import NormalizedView
from .hull import Grid
from .surface import Mesh

BACK_RATIO = {"symmetric": 1.0, "flat": 0.35}


@dataclass
class ReliefParams:
    thickness: float = 0.6   # full front-to-back depth as a fraction of the silhouette width
    detail: float = 0.7      # 0 = pure inflation, 1 = depth fully modulates the surface
    back: str = "symmetric"  # symmetric | flat


@dataclass
class Relief:
    front: np.ndarray        # (H, W) front height above the mid-plane, pixel units
    back: np.ndarray         # (H, W) back height (<= 0)
    half_thickness: float


def build_heights(mask: np.ndarray, depth: np.ndarray, params: ReliefParams) -> Relief:
    if params.back not in BACK_RATIO:
        raise ValueError(f"back must be one of {sorted(BACK_RATIO)}")
    cols = np.flatnonzero(mask.any(axis=0))
    width = float(cols[-1] - cols[0] + 1) if cols.size else 1.0
    half = params.thickness * width / 2
    infl = inflation(mask)
    # Light masked blur: keeps the shape from AI depth but drops pixel noise and JPEG artefacts.
    sigma = max(1.0, 0.004 * max(mask.shape))
    m = mask.astype(np.float32)
    depth = ndimage.gaussian_filter(depth * m, sigma) / np.maximum(ndimage.gaussian_filter(m, sigma), 1e-6)
    infl = infl / (infl.max() or 1.0)
    # AI depth sets the front surface directly (faces, folds); inflation only rounds it. A narrow
    # rim falloff closes the solid at the outline without flattening the interior detail.
    shape = (1 - params.detail) * infl + params.detail * depth
    rim = max(2.0, 0.04 * width)
    falloff = np.sqrt(np.clip(ndimage.distance_transform_edt(mask) / rim, 0, 1))
    front = half * shape * falloff
    # The back is unseen: a smooth rounded shell, never a mirror of the front's features.
    back = -half * BACK_RATIO[params.back] * infl * falloff
    front[~mask] = 0
    back[~mask] = 0
    return Relief(front.astype(np.float32), back.astype(np.float32), half)


def heightfield_mesh(relief: Relief, mask: np.ndarray, max_side: int = 384) -> tuple[Mesh, float]:
    """Closed mesh straight from the front/back height maps at image resolution.

    One vertex per pixel on each side; quads where four neighbouring pixels are
    inside the mask; side walls stitch front to back along the outline.
    Returns the mesh in pixel units (x right, y up, z toward the viewer) and
    the pixel stride used.
    """
    k = max(1, int(np.ceil(max(mask.shape) / max_side)))
    m = mask[::k, ::k]
    f, b = relief.front[::k, ::k], relief.back[::k, ::k]
    h, w = m.shape
    idx = -np.ones((h, w), dtype=np.int64)
    ys, xs = np.nonzero(m)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    px, py = xs * k, ys * k
    verts = np.concatenate([np.column_stack([px, -py, f[ys, xs]]), np.column_stack([px, -py, b[ys, xs]])]).astype(float)
    a, bb, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    full = (a >= 0) & (bb >= 0) & (c >= 0) & (d >= 0)
    a, bb, c, d = a[full], bb[full], c[full], d[full]
    # Image rows grow downward and y = -row, so (a, c, b) is counter-clockwise seen from +Z.
    front_tris = np.concatenate([np.column_stack([a, c, bb]), np.column_stack([bb, c, d])])
    back_tris = front_tris[:, ::-1] + n
    e = np.concatenate([front_tris[:, [0, 1]], front_tris[:, [1, 2]], front_tris[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = e[counts[inv.ravel()] == 1]          # directed outline edges of the front sheet
    u, v = border[:, 0], border[:, 1]
    walls = np.concatenate([np.column_stack([v, u, u + n]), np.column_stack([v, u + n, v + n])])
    faces = np.concatenate([front_tris, back_tris, walls])
    used = np.zeros(2 * n, bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    mesh = Mesh(verts[used], remap[faces])
    near = ndimage.distance_transform_edt(m)[ys, xs] <= 3
    _relax(mesh, np.concatenate([near, near])[used], iterations=12)
    return mesh, float(k)


def _relax(mesh: Mesh, movable: np.ndarray, iterations: int, lam: float = 0.5) -> None:
    """Laplacian smoothing of the selected vertices only: removes the pixel staircase along the outline."""
    f = mesh.faces
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(mesh.vertices)).astype(float)[:, None]
    v = mesh.vertices
    for _ in range(iterations):
        acc = np.zeros_like(v)
        np.add.at(acc, e[:, 0], v[e[:, 1]])
        step = acc / np.maximum(deg, 1) - v
        v = v + lam * step * movable[:, None]
    mesh.vertices = v


def z_extent(relief: Relief, scale: float) -> tuple[float, float]:
    """World Z range of the relief for a plane scale of `scale` world units per pixel."""
    return float(relief.back.min()) * scale, float(relief.front.max()) * scale


def relief_field(relief: Relief, mask: np.ndarray, nv: NormalizedView, grid: Grid) -> np.ndarray:
    """Signed field on the grid (x, y, z), positive inside, in world units.

    Assumes nv is the front view (u = +X, v = +Y), the image the relief came from.
    """
    inv = np.linalg.inv(nv.svg_to_plane)          # plane (u, v) -> pixel (x right, y down)
    xs, ys = grid.centers(0), grid.centers(1)
    U, V = np.meshgrid(xs, ys, indexing="ij")
    px = inv[0, 0] * U + inv[0, 1] * V + inv[0, 2]
    py = inv[1, 0] * U + inv[1, 1] * V + inv[1, 2]
    scale = abs(nv.svg_to_plane[0, 0])            # world units per pixel

    # Signed distance to the outline in world units: positive inside.
    sd = (ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)).astype(np.float32)
    coords = [py.ravel(), px.ravel()]
    sample = lambda img: ndimage.map_coordinates(img, coords, order=1, mode="nearest").reshape(U.shape)  # noqa: E731
    s_xy = sample(sd) * scale
    f = sample(relief.front) * scale
    b = sample(relief.back) * scale
    z = grid.centers(2)[None, None, :]
    s_z = np.minimum(f[:, :, None] - z, z - b[:, :, None])
    return np.minimum(s_xy[:, :, None], s_z).astype(np.float32)


def sample_colors(rgb: np.ndarray, nv: NormalizedView, vertices: np.ndarray) -> np.ndarray:
    """Per-vertex RGB in [0, 1], projected straight back onto the photo along Z."""
    inv = np.linalg.inv(nv.svg_to_plane)
    px = inv[0, 0] * vertices[:, 0] + inv[0, 1] * vertices[:, 1] + inv[0, 2]
    py = inv[1, 0] * vertices[:, 0] + inv[1, 1] * vertices[:, 1] + inv[1, 2]
    out = np.empty((len(vertices), 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = ndimage.map_coordinates(rgb[..., c].astype(np.float32), [py, px], order=1, mode="nearest")
    return out / 255.0
