"""Procedural human detail probes with exact geometry, for measuring detail ceilings.

No scans or third-party assets: a right hand built from an SDF (palm, five
fingers with three phalanges, a thumb), with joint creases and palm ridges of
known depth, and fingers separated by a known gap. Units are metres, +Y up
along the fingers, palm facing +Z. These are test probes, not anatomy
references: proportions are typical adult values, not measured from a person.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from ..reconstruct.hull import Grid
from ..reconstruct.surface import Mesh, field_to_mesh
from .dataset import sd_box, sd_capsule, smin


@dataclass(frozen=True)
class HandParams:
    gap: float = 0.004        # free space between neighbouring fingers
    radius: float = 0.0085    # proximal finger radius (tapers to 80 % at the tip)
    crease: float = 0.0012    # depth of the joint creases
    ridge: float = 0.0005     # amplitude of the palm ridges
    ridge_period: float = 0.006

    def to_dict(self) -> dict:
        return asdict(self)


PALM_HALF = (0.040, 0.048, 0.013)   # x, y, z half extents of the palm box
PALM_TOP = 0.096                     # palm spans y in [0, PALM_TOP]
FINGER_LENGTHS = (0.072, 0.080, 0.075, 0.058)   # index, middle, ring, little
PHALANX_SPLIT = (0.45, 0.30, 0.25)              # proximal, middle, distal share of a finger


def finger_axes(p: HandParams) -> list[tuple[np.ndarray, np.ndarray]]:
    """(base, tip) of the four fingers, side by side with `gap` between their surfaces."""
    pitch = 2 * p.radius + p.gap
    xs = (np.arange(4) - 1.5) * pitch
    return [(np.array([x, PALM_TOP - 0.004, 0.0]), np.array([x, PALM_TOP - 0.004 + L, 0.0]))
            for x, L in zip(xs, FINGER_LENGTHS)]


def thumb_axis() -> tuple[np.ndarray, np.ndarray]:
    return np.array([-0.036, 0.030, 0.006]), np.array([-0.078, 0.080, 0.018])


def joints(p: HandParams) -> list[np.ndarray]:
    """Joint centres where creases are cut (two per finger)."""
    out = []
    for a, b in finger_axes(p):
        t = np.cumsum(PHALANX_SPLIT)[:2]
        out += [a + (b - a) * s for s in t]
    return out


def hand_sdf(p: HandParams = HandParams()):
    """Signed distance (negative inside) of the hand, evaluated on points (..., 3)."""
    fingers, (ta, tb), jts = finger_axes(p), thumb_axis(), joints(p)

    def sdf(q):
        palm = sd_box(q - np.array([0.0, PALM_HALF[1], 0.0]), PALM_HALF, 0.010)
        # palm ridges: shallow grooves across the palm side (+Z) only
        palm = palm + p.ridge * 0.5 * (1 - np.cos(2 * np.pi * q[..., 1] / p.ridge_period)) * (q[..., 2] > 0)
        d = palm
        for a, b in fingers:
            seg = b - a
            t = np.clip(((q - a) @ seg) / (seg @ seg), 0, 1)
            r = p.radius * (1 - 0.2 * t)
            d = smin(d, sd_capsule(q, a, b, 0.0) - r, 0.003)
        d = smin(d, sd_capsule(q, ta, tb, p.radius * 1.1), 0.006)
        for j in jts:                       # creases: narrow bands that push the surface inward
            band = np.exp(-(((q[..., 1] - j[1]) / 0.0012) ** 2)) * (np.abs(q[..., 0] - j[0]) < p.radius * 1.3)
            d = d + p.crease * band
        return d

    return sdf


def hand_mesh(p: HandParams = HandParams(), voxel: float = 0.0005) -> Mesh:
    """Marching cubes of hand_sdf on a `voxel`-sized grid, evaluated slab by slab."""
    lo = np.array([-0.095, -0.012, -0.030])
    hi = np.array([0.050, PALM_TOP + max(FINGER_LENGTHS) + 0.012, 0.042])
    shape = tuple(int(np.ceil(s)) for s in (hi - lo) / voxel)
    grid = Grid(lo, np.full(3, voxel), shape)
    sdf = hand_sdf(p)
    xs, ys = grid.centers(0), grid.centers(1)
    field = np.empty(shape, np.float32)
    for k, z in enumerate(grid.centers(2)):
        q = np.stack(np.meshgrid(xs, ys, [z], indexing="ij"), -1)[:, :, 0]
        field[:, :, k] = -sdf(q)
    return field_to_mesh(field, grid)


def finger_region(points: np.ndarray, p: HandParams = HandParams()) -> np.ndarray:
    """Mask of points above the palm, where only the four fingers exist."""
    return points[:, 1] > PALM_TOP + 0.004


def count_fingers(mesh: Mesh, p: HandParams = HandParams(), voxel: float = 0.0005) -> int:
    """Separate solid regions in the horizontal slice through the middle phalanges (4 for a correct hand)."""
    y = PALM_TOP + 0.004 + 0.55 * min(FINGER_LENGTHS)
    tri = mesh.vertices[mesh.faces]
    above = tri[:, :, 1] > y
    tri = tri[above.any(1) & ~above.all(1)]          # triangles cut by the plane
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):            # the cut point on each crossing edge, NaN elsewhere
        e0, e1 = tri[:, i], tri[:, j]
        t = (y - e0[:, 1]) / np.where(e1[:, 1] == e0[:, 1], np.nan, e1[:, 1] - e0[:, 1])
        ok = (t >= 0) & (t <= 1)
        pts.append(np.where(ok[:, None], e0[:, [0, 2]] + (e1[:, [0, 2]] - e0[:, [0, 2]]) * t[:, None], np.nan))
    pts = np.stack(pts, 1)                           # (T, 3 edges, 2)
    ends = np.array([p[~np.isnan(p[:, 0])][:2] for p in pts if (~np.isnan(p[:, 0])).sum() >= 2])
    lo, hi = np.array([-0.06, -0.03]), np.array([0.06, 0.03])
    n = np.ceil((hi - lo) / voxel).astype(int)
    img = np.zeros(n, bool)
    if len(ends):                                    # draw each cross-section segment densely
        k = int(np.ceil(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).max() / (0.5 * voxel))) + 2
        w = np.linspace(0, 1, k)[None, :, None]
        seg = (ends[:, :1] * (1 - w) + ends[:, 1:] * w).reshape(-1, 2)
        idx = np.floor((seg - lo) / voxel).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < n), axis=1)]
        img[idx[:, 0], idx[:, 1]] = True
    lab, k = ndimage.label(ndimage.binary_fill_holes(img))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
    return int((np.asarray(sizes) >= 20).sum())      # slivers under 20 px (about 5 mm²) are mesh noise
