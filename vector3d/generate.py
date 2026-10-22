"""One-click generation: input views -> validated mesh -> GLB / OBJ / metadata.

Runs the implemented pipeline stages in order and reports each step through
a progress callback, so a UI or agent can follow along. Stages that are not
implemented yet (semantics, AI fill) are reported as skipped with the reason,
never silently faked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .core.confidence import Confidence
from .export.writers import to_glb, to_obj
from .normalize.normalizer import normalize_views
from .normalize.views import ViewName
from .raster.image_input import is_svg, raster_document
from .reconstruct.hints import apply_hints, extent_overrides, validate_hints
from .reconstruct.hull import carve, projection_iou
from .reconstruct.surface import occupancy_to_mesh, taubin_smooth, validate
from .report import summary
from .svg.parser import SvgDocument, parse_svg

STEPS: tuple[tuple[str, str], ...] = (
    ("load", "Read input views"),
    ("normalize", "Normalize coordinates"),
    ("semantics", "Semantic understanding"),
    ("carve", "Carve visual hull"),
    ("ai_fill", "AI fill for unseen geometry"),
    ("surface", "Extract surface"),
    ("smooth", "Smooth surface"),
    ("validate", "Validate mesh"),
    ("export", "Export GLB / OBJ / metadata"),
)

Progress = Callable[[str, str, str], None]  # (step, status: running|done|skipped|error, detail)


@dataclass
class GenerateOptions:
    resolution: int = 128
    smooth_iterations: int = 10
    tolerance: float = 0.02
    hints: dict | None = None  # see vector3d/reconstruct/hints.py and docs/AI_PLAYBOOK.md


@dataclass
class GenerateResult:
    glb: bytes
    obj: str
    metadata: dict
    timings: dict[str, float] = field(default_factory=dict)


def load_view(view: str, name: str, data: bytes) -> SvgDocument:
    v = ViewName(view.lower()).value
    if is_svg(data):
        doc = parse_svg(data.decode("utf-8", errors="replace"), is_text=True)
    else:
        doc = raster_document(data, view_hint=v)
    doc.source = name or f"{v}"
    return doc


def generate(inputs: dict[str, tuple[str, bytes]], options: GenerateOptions | None = None,
             progress: Progress | None = None) -> GenerateResult:
    opts = options or GenerateOptions()
    report = progress or (lambda *_: None)
    timings: dict[str, float] = {}
    current = ""

    def step(key: str):
        nonlocal current
        current = key
        report(key, "running", "")
        timings[key] = time.perf_counter()

    def done(key: str, detail: str = "", status: str = "done"):
        timings[key] = round(time.perf_counter() - timings.get(key, time.perf_counter()), 3)
        report(key, status, detail)

    try:
        step("load")
        if not inputs:
            raise ValueError("no input views")
        docs = {v: load_view(v, name, data) for v, (name, data) in inputs.items()}
        kinds = {v: ("svg" if d.contours and d.contours[0].tag != "raster" else "image") for v, d in docs.items()}
        done("load", ", ".join(f"{v}: {kinds[v]}, {len(d.contours)} contours" for v, d in docs.items()))

        step("normalize")
        scene = normalize_views(docs, tolerance=opts.tolerance)
        bad = [c for c in scene.checks if not c.ok]
        done("normalize", f"units: {scene.units}; {len(scene.checks) - len(bad)}/{len(scene.checks)} cross-view checks ok")

        step("semantics")
        done("semantics", "not implemented yet; no AI used", "skipped")

        step("carve")
        hints = opts.hints or {}
        if hints:
            validate_hints(hints)
        hull = carve(scene, resolution=opts.resolution, extent_overrides=extent_overrides(hints, scene))
        filled = int(hull.occupancy.sum())
        done("carve", f"grid {'x'.join(map(str, hull.grid.shape))}, {filled:,} voxels filled")

        step("ai_fill")
        applied: list[str] = []
        if hints:
            applied = apply_hints(hints, hull, scene)
            done("ai_fill", f"{len(applied)} hint(s) from {hints.get('author', 'unknown')}: " + "; ".join(applied))
        elif hull.assumptions:
            done("ai_fill", "no hints supplied; " + "; ".join(hull.assumptions), "skipped")
        else:
            done("ai_fill", "every axis is observed; nothing to fill", "skipped")

        step("surface")
        mesh = occupancy_to_mesh(hull.occupancy, hull.grid)
        done("surface", f"{len(mesh.faces):,} triangles")

        step("smooth")
        mesh = taubin_smooth(mesh, opts.smooth_iterations)
        done("smooth", f"Taubin, {opts.smooth_iterations} iterations" if opts.smooth_iterations else "off",
             "done" if opts.smooth_iterations else "skipped")

        step("validate")
        checks = validate(mesh)
        iou = projection_iou(hull, scene)
        checks["silhouette_iou"] = iou
        worst = min(iou.values()) if iou else 1.0
        done("validate", f"watertight: {'yes' if checks['watertight'] else 'no'}; worst silhouette match {worst:.1%}")

        step("export")
        geometry_confidence = Confidence.AI_GENERATED if applied else Confidence.ESTIMATED
        metadata = {
            "generator": "Vector3D Engine",
            "units": scene.units,
            "coordinate_system": {"right_handed": True, "up": "+Y", "front": "+Z", "unit": "meter" if scene.units == "m" else "drawing_unit"},
            "geometry_confidence": geometry_confidence.value,
            "method": "visual_hull",
            "notes": ["Visual hull is an upper bound: concavities hidden in every view are not reconstructed."],
            "assumptions": hull.assumptions,
            "hints": {"author": hints.get("author"), "applied": applied, "notes": hints.get("notes")} if hints else None,
            "options": {"resolution": opts.resolution, "smooth_iterations": opts.smooth_iterations},
            "views": summary(scene),
            "validation": checks,
        }
        glb = to_glb(mesh, extras={"vector3d": {"geometry_confidence": geometry_confidence.value, "units": scene.units}})
        obj = to_obj(mesh)
        done("export", f"GLB {len(glb) / 1024:.0f} KB, OBJ {len(obj) / 1024:.0f} KB")
        metadata["timings_s"] = timings
        return GenerateResult(glb, obj, metadata, timings)
    except Exception as exc:
        report(current or "load", "error", str(exc))
        raise
