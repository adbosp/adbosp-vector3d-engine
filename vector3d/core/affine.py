"""2D affine transforms as 3x3 homogeneous matrices (column vectors)."""
from __future__ import annotations

import math

import numpy as np

Affine = np.ndarray  # shape (3, 3)


def identity() -> Affine:
    return np.eye(3)


def matrix(a: float, b: float, c: float, d: float, e: float, f: float) -> Affine:
    """SVG matrix(a b c d e f): x' = a*x + c*y + e, y' = b*x + d*y + f."""
    return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])


def translate(tx: float, ty: float = 0.0) -> Affine:
    return matrix(1, 0, 0, 1, tx, ty)


def scale(sx: float, sy: float | None = None) -> Affine:
    return matrix(sx, 0, 0, sx if sy is None else sy, 0, 0)


def rotate(deg: float, cx: float = 0.0, cy: float = 0.0) -> Affine:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    rot = matrix(c, s, -s, c, 0, 0)
    return translate(cx, cy) @ rot @ translate(-cx, -cy)


def skew_x(deg: float) -> Affine:
    return matrix(1, 0, math.tan(math.radians(deg)), 1, 0, 0)


def skew_y(deg: float) -> Affine:
    return matrix(1, math.tan(math.radians(deg)), 0, 1, 0, 0)


def apply(m: Affine, pts: np.ndarray) -> np.ndarray:
    """Apply to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ m[:2, :2].T + m[:2, 2]
