"""Coarse reconstruction: visual hull by silhouette carving.

Every view's filled silhouette is extruded along its viewing axis; the
object is the intersection of those prisms. The result is an upper bound
of the true shape: concavities invisible in every silhouette cannot be
recovered here and are left to later stages. Output is ESTIMATED geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.draw import polygon2mask

from ..core.geometry import Contour
from ..normalize.normalizer import AXIS_NAMES, NormalizedScene, NormalizedView
from ..report import scene_extent


@dataclass
class Grid:
    lo: np.ndarray      # world coordinate of the grid corner (3,)
    h: np.ndarray       # voxel edge length per axis (3,); nearly cubic
    shape: tuple[int, int, int]

    def centers(self, axis: int) -> np.ndarray:
        return self.lo[axis] + (np.arange(self.shape[axis]) + 0.5) * self.h[axis]


@dataclass
class HullResult:
    occupancy: np.ndarray
    grid: Grid
    masks: dict[str, np.ndarray]           # per view, indexed [k_u_axis, k_v_axis] in world-axis order
    assumptions: list[str] = field(default_factory=list)


def _is_hole(c: Contour) -> bool:
    tags = " ".join(str(c.semantic.get(k, "")) for k in ("data-v3d-part", "data-v3d-role", "class"))
    return "hole" in tags.lower()


def silhouette_polygons(nv: NormalizedView, samples: int = 24) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    """(filled polygons, hole polygons, even-odd) in plane coordinates.

    Raster views are traced boundaries, so even-odd filling is exact. In SVG
    line drawings an inner contour is usually an edge, not a hole, so SVG
    contours are united and only contours tagged as holes are subtracted.
    """
    closed = [c for c in nv.contours if c.closed and c.semantic.get("line_style") != "dashed"]
    evenodd = any(c.semantic.get("fill_rule") == "evenodd" for c in closed)
    if evenodd:
        return [c.sample(samples) for c in closed], [], True
    fills = [c.sample(samples) for c in closed if not _is_hole(c)]
    holes = [c.sample(samples) for c in closed if _is_hole(c)]
    return fills, holes, False


def _view_mask(nv: NormalizedView, grid: Grid) -> tuple[np.ndarray, int, int]:
    cam = nv.camera
    (wu, su), (wv, sv) = cam.axis_of(0), cam.axis_of(1)
    shape = (grid.shape[wu], grid.shape[wv])

    def to_index(poly: np.ndarray) -> np.ndarray:
        ku = (su * poly[:, 0] - grid.lo[wu]) / grid.h[wu] - 0.5
        kv = (sv * poly[:, 1] - grid.lo[wv]) / grid.h[wv] - 0.5
        return np.column_stack([ku, kv])

    fills, holes, evenodd = silhouette_polygons(nv)
    mask = np.zeros(shape, dtype=bool)
    for p in fills:
        m = polygon2mask(shape, to_index(p))
        mask = mask ^ m if evenodd else mask | m
    for p in holes:
        mask &= ~polygon2mask(shape, to_index(p))
    return mask, wu, wv


def carve(scene: NormalizedScene, resolution: int = 128, pad_voxels: int = 2,
          extent_overrides: dict[int, tuple[float, float]] | None = None) -> HullResult:
    """extent_overrides: world extent for axes no view observes (from hints)."""
    ext = scene_extent(scene)
    assumptions = []
    for a, rng in (extent_overrides or {}).items():
        if a in ext:
            assumptions.append(f"hint extent for {AXIS_NAMES[a]} ignored: the axis is observed by a view")
        else:
            ext[a] = rng
            assumptions.append(f"{AXIS_NAMES[a]} extent {rng[0]:.4g}..{rng[1]:.4g} taken from hints (AI_GENERATED)")
    known = {a: hi - lo for a, (lo, hi) in ext.items() if hi > lo}
    if not known:
        raise ValueError("views contain no closed silhouette")
    for a in range(3):
        if a not in ext:
            # Only one viewing direction: depth is unobservable. Assume it equals
            # the smaller visible extent, centred on the origin, and say so.
            depth = min(known.values())
            ext[a] = (-depth / 2, depth / 2)
            assumptions.append(f"{AXIS_NAMES[a]} depth not visible in any view; assumed {depth:.4g} "
                               f"(smaller visible extent)")
    lo = np.array([ext[a][0] for a in range(3)])
    size = np.array([ext[a][1] for a in range(3)]) - lo
    # Per-axis spacing chosen so the object's extent spans a whole number of
    # voxels: bounding faces then fall exactly between voxel centres, where
    # marching cubes places the surface.
    target = size.max() / resolution
    n = np.maximum(1, np.round(size / target)).astype(int)
    h = size / n
    lo = lo - pad_voxels * h
    shape = tuple(int(k) for k in n + 2 * pad_voxels)
    grid = Grid(lo, h, shape)

    occ = np.ones(shape, dtype=bool)
    masks = {}
    for v, nv in scene.views.items():
        mask, wu, wv = _view_mask(nv, grid)
        masks[v.value] = mask
        if not mask.any():
            assumptions.append(f"{v.value}: no closed silhouette; view ignored for carving")
            continue
        m = mask if wu < wv else mask.T             # axes now in increasing world order
        depth = nv.camera.depth_axis
        occ &= np.expand_dims(m, axis=depth)
    return HullResult(occ, grid, masks, assumptions)


def projection_iou(hull: HullResult, scene: NormalizedScene) -> dict[str, float]:
    """How well the carved solid reproduces each view's silhouette (1.0 = exact)."""
    out = {}
    for v, nv in scene.views.items():
        mask = hull.masks[v.value]
        if not mask.any():
            continue
        (wu, _), (wv, _) = nv.camera.axis_of(0), nv.camera.axis_of(1)
        proj = hull.occupancy.any(axis=nv.camera.depth_axis)
        proj = proj if wu < wv else proj.T
        inter = np.logical_and(proj, mask).sum()
        union = np.logical_or(proj, mask).sum()
        out[v.value] = float(inter / union) if union else 1.0
    return out
