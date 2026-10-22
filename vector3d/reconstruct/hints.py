"""Geometry hints: the contract between AI agents and the engine.

An agent (Claude, Codex, a person) never edits meshes. It writes a small
JSON document of *decisions* (unseen depth, symmetry, revolve, primitive
features) and the engine applies them deterministically on the voxel grid.
Everything a hint creates is labelled AI_GENERATED in the metadata.

Schema (all keys optional) is documented in docs/AI_PLAYBOOK.md:

{
  "version": 1,
  "author": "claude",
  "extent":   {"z": [-0.4, 0.4]},                      # world extent for an axis no view observes
  "revolve":  {"view": "front", "axis": "y", "center": 0.0},
  "symmetry": ["x"],                                   # mirror-union across the plane axis = 0
  "primitives": [
    {"op": "subtract", "type": "cylinder", "center": [0, 0, 0], "radius": 0.25, "height": 3.2, "axis": "z"},
    {"op": "add", "type": "box", "center": [0, 0.6, 0], "size": [0.5, 0.2, 0.5]},
    {"op": "add", "type": "sphere", "center": [0, 1, 0], "radius": 0.1}
  ],
  "notes": "why these decisions were made"
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..normalize.normalizer import NormalizedScene
from ..normalize.views import ViewName
from .hull import Grid, HullResult

AXES = {"x": 0, "y": 1, "z": 2}
PRIMITIVE_TYPES = {"box", "cylinder", "sphere"}


def load_hints(source: str | Path | dict | None) -> dict:
    if source is None:
        return {}
    if isinstance(source, dict):
        hints = source
    else:
        hints = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_hints(hints)
    return hints


def validate_hints(h: dict) -> None:
    if not isinstance(h, dict):
        raise ValueError("hints must be a JSON object")
    for ax, rng in (h.get("extent") or {}).items():
        if ax not in AXES or len(rng) != 2 or not rng[0] < rng[1]:
            raise ValueError(f"extent.{ax} must be [min, max] with min < max")
    rev = h.get("revolve")
    if rev:
        if rev.get("axis", "y") not in AXES:
            raise ValueError("revolve.axis must be x, y or z")
        ViewName(rev.get("view", "front"))
    for ax in h.get("symmetry") or []:
        if ax not in AXES:
            raise ValueError(f"symmetry axis must be x, y or z, got {ax!r}")
    for i, p in enumerate(h.get("primitives") or []):
        if p.get("op") not in ("add", "subtract"):
            raise ValueError(f"primitives[{i}].op must be add or subtract")
        if p.get("type") not in PRIMITIVE_TYPES:
            raise ValueError(f"primitives[{i}].type must be one of {sorted(PRIMITIVE_TYPES)}")
        if len(p.get("center", [])) != 3:
            raise ValueError(f"primitives[{i}].center must be [x, y, z]")


def _centers(grid: Grid):
    return np.meshgrid(grid.centers(0), grid.centers(1), grid.centers(2), indexing="ij")


def _primitive_mask(p: dict, grid: Grid) -> np.ndarray:
    X, Y, Z = _centers(grid)
    P = (X - p["center"][0], Y - p["center"][1], Z - p["center"][2])
    if p["type"] == "box":
        sx, sy, sz = (s / 2 for s in p["size"])
        return (np.abs(P[0]) <= sx) & (np.abs(P[1]) <= sy) & (np.abs(P[2]) <= sz)
    if p["type"] == "sphere":
        return P[0] ** 2 + P[1] ** 2 + P[2] ** 2 <= p["radius"] ** 2
    a = AXES[p.get("axis", "z")]
    others = [P[i] for i in range(3) if i != a]
    return (others[0] ** 2 + others[1] ** 2 <= p["radius"] ** 2) & (np.abs(P[a]) <= p["height"] / 2)


def _revolve(rev: dict, hull: HullResult, scene: NormalizedScene) -> np.ndarray:
    """Solid of revolution of one view's silhouette about a world axis lying in that view's plane."""
    view = ViewName(rev.get("view", "front"))
    if view not in scene.views:
        raise ValueError(f"revolve.view {view.value!r} was not supplied")
    cam = scene.views[view].camera
    axis = AXES[rev.get("axis", "y")]
    (wu, _), (wv, _) = cam.axis_of(0), cam.axis_of(1)
    if axis not in (wu, wv):
        raise ValueError(f"revolve axis {rev.get('axis', 'y')} is not in the {view.value} view plane")
    radial = wu if axis == wv else wv
    mask = hull.masks[view.value]                      # indexed [k_wu, k_wv]
    m = mask if wu < wv else mask.T                    # indexed in world-axis order
    m = m if radial < axis else m.T                    # -> [k_radial, k_axis]
    grid = hull.grid
    c = float(rev.get("center", 0.0))
    # Profile radius at each axis station: the silhouette's furthest extent from the axis.
    rc = grid.centers(radial) - c
    # Measure to the outer face of the furthest cell, so cells off the profile plane
    # (whose centres are up to half a voxel out of it) still reproduce the silhouette.
    r_max = np.where(m, np.abs(rc)[:, None] + grid.h[radial] / 2, -np.inf).max(axis=0)
    X = _centers(grid)
    perp = [i for i in range(3) if i != axis]
    r = np.sqrt((X[perp[0]] - (c if perp[0] == radial else 0.0)) ** 2 + (X[perp[1]] - (c if perp[1] == radial else 0.0)) ** 2)
    shape = [1, 1, 1]
    shape[axis] = -1
    return r <= r_max.reshape(shape)


def apply_hints(hints: dict, hull: HullResult, scene: NormalizedScene) -> list[str]:
    """Modify hull.occupancy in place; return a description of every applied hint."""
    applied = []
    occ = hull.occupancy
    if hints.get("revolve"):
        occ &= _revolve(hints["revolve"], hull, scene)
        applied.append(f"revolve {hints['revolve'].get('view', 'front')} silhouette about "
                       f"{hints['revolve'].get('axis', 'y')}")
    for ax in hints.get("symmetry") or []:
        a = AXES[ax]
        # Mirror across the plane through the world origin; grids are padded symmetrically only
        # when the object is centred, so resample by coordinate rather than flipping indices.
        coords = hull.grid.centers(a)
        src = np.clip(np.round((-coords - hull.grid.lo[a]) / hull.grid.h[a] - 0.5).astype(int), 0, occ.shape[a] - 1)
        occ |= np.take(occ, src, axis=a)
        applied.append(f"mirror symmetry across {ax} = 0")
    for p in hints.get("primitives") or []:
        m = _primitive_mask(p, hull.grid)
        if p["op"] == "add":
            occ |= m
        else:
            occ &= ~m
        applied.append(f"{p['op']} {p['type']} at {p['center']}")
    return applied



def extent_overrides(hints: dict, scene: NormalizedScene) -> dict[int, tuple[float, float]]:
    """World extents for axes no view observes, decided by the hints.

    An explicit `extent` wins. Otherwise a revolve makes the unseen depth
    equal to the revolved diameter.
    """
    from ..report import scene_extent

    ext = scene_extent(scene)
    out = {AXES[a]: (float(r[0]), float(r[1])) for a, r in (hints.get("extent") or {}).items()}
    rev = hints.get("revolve")
    if rev:
        axis = AXES[rev.get("axis", "y")]
        c = float(rev.get("center", 0.0))
        for a in range(3):
            if a == axis or a in ext or a in out:
                continue
            radial = next((b for b in ext if b != axis), None)
            if radial is not None:
                R = max(abs(ext[radial][0] - c), abs(ext[radial][1] - c))
                out[a] = (-R, R)
    return out
