"""Coordinate normalization across all orthographic views.

Input: parsed SVG documents keyed by view. Output: contours for every view in
a common view-plane frame (u right, v up, meters when scale is known) whose
origin coincides with the world origin projected into that view, so that
shared world axes line up between views.

Every inferred quantity carries a Confidence and a recorded source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import affine
from ..core.confidence import Confidence
from ..core.geometry import Contour, bbox_of
from ..svg.parser import ScaleInfo, SvgDocument
from .views import CAMERAS, OPPOSITE, ViewCamera, ViewName, shared_axes

AXIS_NAMES = "XYZ"


@dataclass
class NormalizedView:
    view: ViewName
    camera: ViewCamera
    contours: list[Contour]          # in view-plane coordinates (u, v)
    construction: list[Contour]
    scale: ScaleInfo
    origin_source: str               # marker | bbox_center
    origin_confidence: Confidence
    svg_to_plane: np.ndarray         # affine: SVG user units -> plane coordinates
    source: str

    def bbox(self):
        return bbox_of(self.contours)

    def world_extent(self) -> dict[int, tuple[float, float]]:
        """Extent along each visible world axis."""
        box = self.bbox()
        if box is None:
            return {}
        lo, hi = box
        out = {}
        for pa in (0, 1):
            w, sign = self.camera.axis_of(pa)
            a, b = sorted((lo[pa] * sign, hi[pa] * sign))
            out[w] = (float(a), float(b))
        return out


@dataclass
class ConsistencyReport:
    view_a: ViewName
    view_b: ViewName
    world_axis: str
    extent_a: float
    extent_b: float
    relative_error: float
    ok: bool


@dataclass
class NormalizedScene:
    views: dict[ViewName, NormalizedView]
    units: str                       # "m" or "drawing_units"
    checks: list[ConsistencyReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_view(name: str | ViewName, doc: SvgDocument) -> ViewName:
    view = ViewName(name.value if isinstance(name, ViewName) else str(name).lower())
    if doc.view_hint and doc.view_hint.lower() != view.value:
        raise ValueError(f"{doc.source}: declared data-v3d-view={doc.view_hint!r} but supplied as {view.value!r}")
    return view


def _extent_along(doc: SvgDocument, cam: ViewCamera, world_axis: int) -> float | None:
    box = doc.bbox()
    if box is None:
        return None
    for pa in (0, 1):
        if cam.axis_of(pa)[0] == world_axis:
            return float(box[1][pa] - box[0][pa])
    return None


def _propagate_scales(docs: dict[ViewName, SvgDocument], initial: dict[ViewName, ScaleInfo],
                      warnings: list[str]) -> dict[ViewName, ScaleInfo]:
    """Fill unknown scales from views that share a world axis with a known-scale view."""
    scales = dict(initial)
    changed = True
    while changed:
        changed = False
        for v, s in list(scales.items()):
            if s.known:
                continue
            for k, ks in scales.items():
                if not ks.known or k == v:
                    continue
                for w, _, _ in shared_axes(v, k):
                    ext_v = _extent_along(docs[v], CAMERAS[v], w)
                    ext_k = _extent_along(docs[k], CAMERAS[k], w)
                    if ext_v and ext_k:
                        mpu = ks.meters_per_unit * ext_k / ext_v
                        scales[v] = ScaleInfo(mpu, "propagated", Confidence.ESTIMATED,
                                              [f"matched {AXIS_NAMES[w]} extent with {k.value} view"])
                        warnings.append(f"{v.value}: scale estimated from {k.value} view via shared "
                                        f"{AXIS_NAMES[w]} extent")
                        changed = True
                        break
                if scales[v].known:
                    break
    return scales


def normalize_views(docs: dict[str | ViewName, SvgDocument], tolerance: float = 0.01) -> NormalizedScene:
    """Bring all views into one consistent frame.

    tolerance: relative extent mismatch allowed between views before a check fails.
    """
    warnings: list[str] = []
    typed = {_resolve_view(k, d): d for k, d in docs.items()}
    if not typed:
        raise ValueError("no views supplied")
    for v, d in typed.items():
        warnings.extend(f"{v.value}: {w}" for w in d.warnings)
        if not d.contours:
            warnings.append(f"{v.value}: no geometry contours")

    scales = _propagate_scales(typed, {v: d.scale for v, d in typed.items()}, warnings)
    all_known = all(s.known for s in scales.values())
    if not all_known:
        # Without a physical scale, keep drawing units but make every view
        # relative to the first one so shared axes still agree.
        ref = next(iter(typed))
        initial = {v: ScaleInfo(None, "unknown", None) for v in typed}
        initial[ref] = ScaleInfo(1.0, "reference", Confidence.ESTIMATED, [f"{ref.value} view defines drawing units"])
        scales = _propagate_scales(typed, initial, warnings)
        unknown = [v.value for v, s in scales.items() if not s.known]
        if unknown:
            warnings.append(f"views with no shared axis to the reference view: {', '.join(unknown)}")
            scales = {v: (s if s.known else ScaleInfo(1.0, "unknown", None)) for v, s in scales.items()}
        warnings.append("no physical scale available; output is in drawing units, not meters")

    views: dict[ViewName, NormalizedView] = {}
    for v, d in typed.items():
        cam = CAMERAS[v]
        s = scales[v].meters_per_unit
        if d.origin is not None:
            origin, origin_src, origin_conf = d.origin, "marker", Confidence.VERIFIED
        else:
            box = d.bbox()
            origin = np.zeros(2) if box is None else (box[0] + box[1]) / 2
            origin_src, origin_conf = "bbox_center", Confidence.ESTIMATED
        # SVG (x right, y down) -> plane (u right, v up), scaled, origin at (0, 0).
        m = affine.scale(s, -s) @ affine.translate(-origin[0], -origin[1])
        views[v] = NormalizedView(
            view=v, camera=cam,
            contours=[c.transformed(m) for c in d.contours],
            construction=[c.transformed(m) for c in d.construction],
            scale=scales[v], origin_source=origin_src, origin_confidence=origin_conf,
            svg_to_plane=m, source=d.source,
        )

    scene = NormalizedScene(views=views, units="m" if all_known else "drawing_units", warnings=warnings)
    scene.checks = check_consistency(scene, tolerance)
    for c in scene.checks:
        if not c.ok:
            warnings.append(f"{c.view_a.value}/{c.view_b.value}: {c.world_axis} extent differs by "
                            f"{c.relative_error:.2%}")
    return scene


def check_consistency(scene: NormalizedScene, tolerance: float = 0.01) -> list[ConsistencyReport]:
    """Compare extents of every world axis seen by two views (includes opposite pairs)."""
    out = []
    names = list(scene.views)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea, eb = scene.views[a].world_extent(), scene.views[b].world_extent()
            for w, _, _ in shared_axes(a, b):
                if w not in ea or w not in eb:
                    continue
                la, lb = ea[w][1] - ea[w][0], eb[w][1] - eb[w][0]
                err = abs(la - lb) / max(la, lb, 1e-12)
                out.append(ConsistencyReport(a, b, AXIS_NAMES[w], la, lb, err, err <= tolerance))
    return out


__all__ = ["normalize_views", "check_consistency", "NormalizedScene", "NormalizedView", "OPPOSITE"]
