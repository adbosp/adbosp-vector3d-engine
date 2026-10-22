"""Exact 2D curve primitives.

Curves are kept analytically (lines, quadratic and cubic Béziers). Affine
transforms map control points to control points, so no precision is lost
until a later stage explicitly asks for sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import affine
from .confidence import Confidence


@dataclass(frozen=True)
class Segment:
    """A polynomial Bézier segment of degree len(points) - 1 (1 = line, 2 = quad, 3 = cubic)."""

    points: np.ndarray  # (degree + 1, 2)

    @property
    def degree(self) -> int:
        return len(self.points) - 1

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def point_at(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
        pts = self.points
        # de Casteljau keeps this numerically stable for any degree.
        work = np.repeat(pts[None, :, :], len(t), axis=0)
        for k in range(self.degree):
            work = (1 - t[:, :, None]) * work[:, :-1] + t[:, :, None] * work[:, 1:]
        return work[:, 0, :]

    def sample(self, n: int = 16) -> np.ndarray:
        n = 1 if self.degree == 1 else n
        return self.point_at(np.linspace(0.0, 1.0, n + 1))

    def transformed(self, m: affine.Affine) -> "Segment":
        return Segment(affine.apply(m, self.points))

    def reversed(self) -> "Segment":
        return Segment(self.points[::-1].copy())


def line(p0, p1) -> Segment:
    return Segment(np.array([p0, p1], dtype=float))


def quad(p0, p1, p2) -> Segment:
    return Segment(np.array([p0, p1, p2], dtype=float))


def cubic(p0, p1, p2, p3) -> Segment:
    return Segment(np.array([p0, p1, p2, p3], dtype=float))


@dataclass
class Contour:
    """One open or closed chain of segments, with its provenance from the SVG."""

    segments: list[Segment]
    closed: bool
    element_id: str | None = None
    tag: str = "path"
    layer_path: tuple[str, ...] = ()
    semantic: dict[str, str] = field(default_factory=dict)
    primitive: dict[str, Any] | None = None  # e.g. {"type": "circle", ...} in source user units
    confidence: Confidence = Confidence.VERIFIED

    def transformed(self, m: affine.Affine) -> "Contour":
        return Contour(
            segments=[s.transformed(m) for s in self.segments],
            closed=self.closed,
            element_id=self.element_id,
            tag=self.tag,
            layer_path=self.layer_path,
            semantic=dict(self.semantic),
            primitive=self.primitive,
            confidence=self.confidence,
        )

    def sample(self, n_per_curve: int = 16) -> np.ndarray:
        if not self.segments:
            return np.zeros((0, 2))
        chunks = [self.segments[0].sample(n_per_curve)]
        chunks += [s.sample(n_per_curve)[1:] for s in self.segments[1:]]
        return np.vstack(chunks)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.sample()
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def anchor_points(self) -> np.ndarray:
        if not self.segments:
            return np.zeros((0, 2))
        return np.vstack([self.segments[0].start] + [s.end for s in self.segments])


def bbox_of(contours: list[Contour]) -> tuple[np.ndarray, np.ndarray] | None:
    boxes = [c.bbox() for c in contours if c.segments]
    if not boxes:
        return None
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
