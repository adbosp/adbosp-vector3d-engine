"""JSON-ready reports of a NormalizedScene, shared by the CLI and the web UI."""
from __future__ import annotations

from .normalize.normalizer import AXIS_NAMES, NormalizedScene


def summary(scene: NormalizedScene) -> dict:
    out = {"units": scene.units, "views": {}, "checks": [], "warnings": scene.warnings}
    for v, nv in scene.views.items():
        box = nv.bbox()
        out["views"][v.value] = {
            "source": nv.source,
            "contours": len(nv.contours),
            "segments": sum(len(c.segments) for c in nv.contours),
            "scale": {"meters_per_unit": nv.scale.meters_per_unit, "source": nv.scale.source,
                      "confidence": nv.scale.confidence.value if nv.scale.confidence else None},
            "origin": {"source": nv.origin_source, "confidence": nv.origin_confidence.value},
            "world_extent": {AXIS_NAMES[w]: [round(a, 6), round(b, 6)] for w, (a, b) in nv.world_extent().items()},
            "bbox_uv": None if box is None else [box[0].round(6).tolist(), box[1].round(6).tolist()],
        }
    for c in scene.checks:
        out["checks"].append({"views": [c.view_a.value, c.view_b.value], "axis": c.world_axis,
                              "extents": [round(c.extent_a, 6), round(c.extent_b, 6)],
                              "relative_error": round(c.relative_error, 6), "ok": c.ok})
    return out


def scene_extent(scene: NormalizedScene) -> dict[int, tuple[float, float]]:
    """Union of every view's extent per world axis."""
    ext: dict[int, tuple[float, float]] = {}
    for nv in scene.views.values():
        for w, (a, b) in nv.world_extent().items():
            lo, hi = ext.get(w, (a, b))
            ext[w] = (min(lo, a), max(hi, b))
    return ext


def geometry(scene: NormalizedScene, samples: int = 24) -> dict:
    """Sampled contours per view, in plane (u, v) and in world space.

    In world space each view is placed on the face of the scene's bounding
    box that faces its camera (a "glass box"), so aligned views can be
    checked visually.
    """
    ext = scene_extent(scene)
    out = {}
    for v, nv in scene.views.items():
        cam = nv.camera
        axis = cam.depth_axis
        sign = cam.position_dir[axis]
        lo, hi = ext.get(axis, (0.0, 0.0))
        depth = max(sign * lo, sign * hi)
        items = []
        for kind, contours in (("geometry", nv.contours), ("construction", nv.construction)):
            for c in contours:
                uv = c.sample(samples)
                items.append({
                    "kind": kind,
                    "id": c.element_id,
                    "part": c.semantic.get("data-v3d-part"),
                    "dashed": c.semantic.get("line_style") == "dashed",
                    "closed": c.closed,
                    "uv": uv.round(6).tolist(),
                    "world": cam.to_world(uv, depth).round(6).tolist(),
                })
        out[v.value] = {"contours": items, "depth": depth,
                        "right": list(cam.right), "up": list(cam.up), "toward_camera": list(cam.position_dir)}
    return out
