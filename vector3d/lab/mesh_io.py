"""Read any mesh trimesh understands (GLB, OBJ, PLY) into the engine's Mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..reconstruct.surface import Mesh


def load_mesh(path: str | Path) -> Mesh:
    """All geometry of the file, merged, with scene transforms applied."""
    obj = trimesh.load(str(path), force="scene", process=False)
    geoms = [g for g in obj.dump() if isinstance(g, trimesh.Trimesh)] if isinstance(obj, trimesh.Scene) else [obj]
    if not geoms:
        raise ValueError(f"{path}: no triangle geometry")
    tm = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    colors = None
    vc = getattr(tm.visual, "vertex_colors", None)
    if tm.visual.kind == "vertex" and vc is not None and len(vc) == len(tm.vertices):
        colors = np.asarray(vc)[:, :3] / 255.0
    return Mesh(np.asarray(tm.vertices, dtype=float), np.asarray(tm.faces, dtype=np.int64), colors)


def to_trimesh(mesh: Mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
