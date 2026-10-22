"""Structured scene description for AI agents (Claude, Codex).

`python -m vector3d describe ...` prints this as JSON. It gives an agent the
measured facts it needs to write hints.json (docs/AI_PLAYBOOK.md) without
looking at pixels: extents, simplified contours, detected circles, mirror
symmetry and a width profile per view.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import approximate_polygon

from .normalize.normalizer import AXIS_NAMES, NormalizedScene
from .reconstruct.hull import carve
from .report import scene_extent, summary


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def describe(scene: NormalizedScene, resolution: int = 96, max_points: int = 64) -> dict:
    hull = carve(scene, resolution=resolution)
    ext = scene_extent(scene)
    out = {
        "units": scene.units,
        "world_extent": {AXIS_NAMES[a]: [round(lo, 6), round(hi, 6)] for a, (lo, hi) in sorted(ext.items())},
        "unobserved_axes": [AXIS_NAMES[a] for a in range(3) if a not in ext],
        "hull_assumptions": hull.assumptions,
        "checks": summary(scene)["checks"],
        "views": {},
    }
    for v, nv in scene.views.items():
        cam = nv.camera
        (wu, su), (wv, sv) = cam.axis_of(0), cam.axis_of(1)
        mask = hull.masks[v.value]                       # [k along wu, k along wv]
        # Mirror symmetry about the view's vertical / horizontal centre line (u = 0 / v = 0).
        sym_u = _iou(mask, mask[::-1, :])
        sym_v = _iou(mask, mask[:, ::-1])
        # Width of the silhouette at 16 stations along v (bottom to top), as fractions of max width.
        cols = mask.any(axis=0)
        widths = mask.sum(axis=0).astype(float)
        stations = np.linspace(0, len(widths) - 1, 16).round().astype(int)
        if sv < 0:
            stations = stations[::-1]
        wmax = widths.max() or 1.0
        contours = []
        for c in nv.contours:
            pts = c.sample(12)
            span = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 0.0
            simple = approximate_polygon(pts, tolerance=max(span, 1e-12) * 0.004) if len(pts) > 2 else pts
            if len(simple) > max_points:
                simple = simple[np.linspace(0, len(simple) - 1, max_points).round().astype(int)]
            contours.append({
                "id": c.element_id, "closed": c.closed,
                "part": c.semantic.get("data-v3d-part"), "dashed": c.semantic.get("line_style") == "dashed",
                "uv": np.round(simple, 6).tolist(),
            })
        circles = []
        for c in nv.contours:
            if c.primitive and c.primitive.get("type") in ("circle", "ellipse"):
                pts = c.sample(16)
                centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
                circles.append({"id": c.element_id, "center_uv": np.round(centre, 6).tolist(),
                                "radius_uv": np.round((pts.max(axis=0) - pts.min(axis=0)) / 2, 6).tolist(),
                                "center_world": np.round(cam.to_world(centre[None])[0], 6).tolist()})
        out["views"][v.value] = {
            "source": nv.source,
            "u_axis": f"{'+' if su > 0 else '-'}{AXIS_NAMES[wu]}",
            "v_axis": f"{'+' if sv > 0 else '-'}{AXIS_NAMES[wv]}",
            "camera_at": f"{'+' if cam.position_dir[cam.depth_axis] > 0 else '-'}{AXIS_NAMES[cam.depth_axis]}",
            "scale": {"meters_per_unit": nv.scale.meters_per_unit, "source": nv.scale.source,
                      "confidence": nv.scale.confidence.value if nv.scale.confidence else None},
            "silhouette_fill_ratio": round(float(mask.sum() / max(mask.size, 1)), 4),
            "mirror_symmetry": {"about_vertical_centerline": round(sym_u, 4),
                                "about_horizontal_centerline": round(sym_v, 4)},
            "width_profile_bottom_to_top": [round(float(widths[i] / wmax), 3) if cols[i] else 0.0 for i in stations],
            "circles": circles,
            "contours": contours,
        }
    return out
