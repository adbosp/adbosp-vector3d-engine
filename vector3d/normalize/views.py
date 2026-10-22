"""Orthographic view cameras.

World frame (right-handed, meters): +X right, +Y up, +Z toward the viewer of
the Front view (glTF convention: the model's front faces +Z).

Each camera maps view-plane coordinates (u right, v up on the drawing) to
world axes. SVG coordinates have y pointing down, so v = -y_svg.

Front/Back, Left/Right and Top/Bottom are opposite pairs: they share the same
two world axes, with the horizontal axis mirrored, and look along opposite
directions of the same depth axis. They are not independent viewing axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ViewName(str, Enum):
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class ViewCamera:
    view: ViewName
    position_dir: tuple[int, int, int]  # unit vector from object toward the camera
    right: tuple[int, int, int]         # world direction of +u on the drawing
    up: tuple[int, int, int]            # world direction of +v on the drawing

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction (camera looks along this)."""
        return -np.array(self.position_dir, dtype=float)

    @property
    def basis(self) -> np.ndarray:
        """3x3 matrix whose columns are right, up, toward-camera."""
        return np.column_stack([self.right, self.up, self.position_dir]).astype(float)

    def to_world(self, uv: np.ndarray, depth: np.ndarray | float = 0.0) -> np.ndarray:
        """(N,2) view-plane points plus depth toward the camera -> (N,3) world points."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        d = np.broadcast_to(np.asarray(depth, dtype=float), (len(uv),))
        return np.column_stack([uv, d]) @ self.basis.T

    def to_view(self, xyz: np.ndarray) -> np.ndarray:
        """(N,3) world points -> (N,3) as (u, v, depth toward camera)."""
        return np.atleast_2d(np.asarray(xyz, dtype=float)) @ self.basis

    def axis_of(self, plane_axis: int) -> tuple[int, int]:
        """World axis index (0=X,1=Y,2=Z) and sign for plane axis 0 (u) or 1 (v)."""
        vec = self.right if plane_axis == 0 else self.up
        idx = int(np.flatnonzero(vec)[0])
        return idx, int(vec[idx])

    @property
    def depth_axis(self) -> int:
        return int(np.flatnonzero(self.position_dir)[0])


CAMERAS: dict[ViewName, ViewCamera] = {
    ViewName.FRONT: ViewCamera(ViewName.FRONT, (0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ViewName.BACK: ViewCamera(ViewName.BACK, (0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    # Left/Right name the object's own sides. The object faces +Z, so its
    # right side is at -X (the viewer's left when looking at the Front view).
    ViewName.RIGHT: ViewCamera(ViewName.RIGHT, (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ViewName.LEFT: ViewCamera(ViewName.LEFT, (1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ViewName.TOP: ViewCamera(ViewName.TOP, (0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ViewName.BOTTOM: ViewCamera(ViewName.BOTTOM, (0, -1, 0), (1, 0, 0), (0, 0, 1)),
}

OPPOSITE: dict[ViewName, ViewName] = {
    ViewName.FRONT: ViewName.BACK, ViewName.BACK: ViewName.FRONT,
    ViewName.LEFT: ViewName.RIGHT, ViewName.RIGHT: ViewName.LEFT,
    ViewName.TOP: ViewName.BOTTOM, ViewName.BOTTOM: ViewName.TOP,
}


def shared_axes(a: ViewName, b: ViewName) -> list[tuple[int, int, int]]:
    """World axes visible in both views as (world_axis, plane_axis_in_a, plane_axis_in_b)."""
    ca, cb = CAMERAS[a], CAMERAS[b]
    out = []
    for pa in (0, 1):
        wa, _ = ca.axis_of(pa)
        for pb in (0, 1):
            if cb.axis_of(pb)[0] == wa:
                out.append((wa, pa, pb))
    return out


def mirror_to_opposite(view: ViewName, uv: np.ndarray) -> np.ndarray:
    """Express plane coordinates of `view` in the plane of its opposite view."""
    src, dst = CAMERAS[view], CAMERAS[OPPOSITE[view]]
    world = src.to_world(uv)
    return dst.to_view(world)[:, :2]
