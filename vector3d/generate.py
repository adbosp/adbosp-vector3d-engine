"""One-click generation: input views -> validated mesh -> GLB / OBJ / metadata.

Two modes share one progress protocol:

* drawing: orthographic drawings (SVG or flat images). Visual hull from the
  silhouettes, optional agent hints. Geometry is ESTIMATED.
* photo: any photo. AI segmentation and monocular depth, a normal map, a
  closed relief solid (front from depth, back mirrored or flattened) carved
  by any other views, coloured from the photo. Geometry is AI_GENERATED.

Each step is reported through a progress callback so a UI or agent can follow
along. Steps that are not implemented (semantics) are reported as skipped
with the reason, never silently faked.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .core.confidence import Confidence
from .export.writers import to_glb, to_obj
from .normalize.normalizer import normalize_views
from .normalize.views import ViewName
from .raster.image_input import is_svg, load_image, mask_document, raster_document
from .reconstruct.hints import apply_hints, extent_overrides, validate_hints
from .reconstruct.hull import carve, projection_iou
from .reconstruct.surface import Mesh, field_to_mesh, occupancy_to_mesh, taubin_smooth, validate
from .report import scene_extent, summary
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

PHOTO_STEPS: tuple[tuple[str, str], ...] = (
    ("load", "Read input images"),
    ("segment", "Remove background"),
    ("depth", "Estimate depth (AI)"),
    ("normals", "Normal map"),
    ("normalize", "Normalize coordinates"),
    ("carve", "Carve with other views"),
    ("relief", "Build solid from depth"),
    ("ai_fill", "Apply hints"),
    ("surface", "Extract surface"),
    ("smooth", "Smooth surface"),
    ("color", "Colour from photo"),
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
    mode: str = "auto"         # auto | drawing | photo
    # photo mode
    depth_backend: str = "auto"    # auto | depth-anything | inflation
    segmenter: str = "auto"        # auto | rembg | classic
    thickness: float = 0.6         # front-to-back depth / silhouette width
    detail: float = 0.7            # how strongly AI depth shapes the front surface
    back: str = "symmetric"        # symmetric | flat
    size_m: float = 1.0            # largest dimension of the photo model, meters (assumed)


@dataclass
class GenerateResult:
    glb: bytes
    obj: str
    metadata: dict
    timings: dict[str, float] = field(default_factory=dict)
    images: dict[str, bytes] = field(default_factory=dict)  # PNGs of intermediate maps (photo mode)


def resolve_mode(inputs: dict[str, tuple[str, bytes]], mode: str = "auto") -> str:
    if mode in ("drawing", "photo"):
        return mode
    if mode != "auto":
        raise ValueError(f"unknown mode {mode!r}")
    return "drawing" if any(is_svg(data) for _, data in inputs.values()) else "photo"


def plan_steps(mode: str) -> tuple[tuple[str, str], ...]:
    return PHOTO_STEPS if mode == "photo" else STEPS


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


class _Runner:
    def __init__(self, progress: Progress | None):
        self.report = progress or (lambda *_: None)
        self.timings: dict[str, float] = {}
        self.current = ""

    def step(self, key: str) -> None:
        self.current = key
        self.report(key, "running", "")
        self.timings[key] = time.perf_counter()

    def done(self, key: str, detail: str = "", status: str = "done") -> None:
        self.timings[key] = round(time.perf_counter() - self.timings.get(key, time.perf_counter()), 3)
        self.report(key, status, detail)


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
    run = _Runner(progress)
    try:
        if not inputs:
            raise ValueError("no input views")
        if opts.hints:
            validate_hints(opts.hints)
        mode = resolve_mode(inputs, opts.mode)
        return (_generate_photo if mode == "photo" else _generate_drawing)(inputs, opts, run)
    except Exception as exc:
        run.report(run.current or "load", "error", str(exc))
        raise


def _generate_drawing(inputs, opts: GenerateOptions, run: _Runner) -> GenerateResult:
    step, done, timings = run.step, run.done, run.timings
    step("load")
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
        "mode": "drawing",
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


def _generate_photo(inputs, opts: GenerateOptions, run: _Runner) -> GenerateResult:
    from .ai.depth import DepthAnythingONNX, estimate_depth, normal_map
    from .ai.segment import segment
    from .reconstruct.relief import ReliefParams, build_heights, relief_field, sample_colors

    step, done, timings = run.step, run.done, run.timings
    hints = opts.hints or {}

    step("load")
    primary = "front" if "front" in inputs else next(iter(inputs))
    name, data = inputs[primary]
    if is_svg(data):
        raise ValueError(f"photo mode needs a raster image for the {primary} view; use drawing mode for SVG")
    rgba = load_image(data)
    rgb = np.ascontiguousarray(rgba[..., :3])
    others = {v: load_view(v, n, d) for v, (n, d) in inputs.items() if v != primary}
    note = "" if primary == "front" else f" (treated as front: a photo is always the primary view)"
    done("load", f"{name}: {rgb.shape[1]}x{rgb.shape[0]} px{note}" + (f"; +{len(others)} other view(s)" if others else ""))

    step("segment")
    mask, seg_method = segment(rgba, opts.segmenter)
    if not mask.any():
        raise ValueError("no foreground found; use an image with a clear subject")
    done("segment", f"{seg_method}; subject covers {mask.mean():.0%} of the image")

    step("depth")
    if opts.depth_backend != "inflation" and not DepthAnythingONNX().model_path.exists():
        run.report("depth", "running", "downloading Depth Anything V2 Small (~100 MB, first run only)")
    dres = estimate_depth(rgb, mask, opts.depth_backend)
    done("depth", dres.backend + (f"; {dres.note}" if dres.note else ""))

    step("normals")
    params = ReliefParams(thickness=opts.thickness, detail=opts.detail, back=opts.back)
    relief = build_heights(mask, dres.depth, params)
    normals = normal_map(relief.front, mask)
    images = {
        "mask.png": _png((mask * 255).astype(np.uint8)),
        "depth.png": _png((dres.depth * 255).astype(np.uint8)),
        "normal.png": _png(normals),
    }
    done("normals", "normal map from the front surface")

    step("normalize")
    docs = {"front": mask_document(mask, seg_method, name, "front")}
    docs.update({v: d for v, d in others.items() if v != "front"})
    scene = normalize_views(docs, tolerance=opts.tolerance)
    front = scene.views[ViewName.FRONT]
    scale = abs(float(front.svg_to_plane[0, 0]))   # plane units per pixel
    done("normalize", f"{len(scene.views)} view(s); {sum(c.ok for c in scene.checks)}/{len(scene.checks)} checks ok")

    step("carve")
    ext = scene_extent(scene)
    if 2 in ext:
        # A side or top view measures the depth: fit the relief's thickness to it.
        target = (ext[2][1] - ext[2][0]) / scale
        current = float(relief.front.max() - relief.back.min()) or 1.0
        relief.front *= target / current
        relief.back *= target / current
        depth_note = "depth taken from other views"
    else:
        depth_note = f"depth assumed: thickness {opts.thickness:.2f} x width ({opts.back} back)"
    zr = (float(relief.back.min()) * scale, float(relief.front.max()) * scale)
    hull = carve(scene, resolution=opts.resolution, extent_overrides={} if 2 in ext else {2: zr})
    done("carve", f"grid {'x'.join(map(str, hull.grid.shape))}; {depth_note}")

    step("relief")
    fld = relief_field(relief, mask, front, hull.grid)
    hvox = float(hull.grid.h.max())
    fld = np.where(hull.occupancy, fld, np.minimum(fld, -0.5 * hvox))
    hull.occupancy = fld > 0
    done("relief", f"{int(hull.occupancy.sum()):,} voxels inside")

    step("ai_fill")
    applied: list[str] = []
    if hints:
        before = hull.occupancy.copy()
        applied = apply_hints(hints, hull, scene)
        fld = np.where(hull.occupancy & ~before, 0.5 * hvox, np.where(~hull.occupancy & before, -0.5 * hvox, fld))
        done("ai_fill", "; ".join(applied))
    else:
        done("ai_fill", "no hints supplied", "skipped")

    step("surface")
    mesh = field_to_mesh(fld, hull.grid)
    done("surface", f"{len(mesh.faces):,} triangles")

    step("smooth")
    mesh = taubin_smooth(mesh, opts.smooth_iterations)
    done("smooth", f"Taubin, {opts.smooth_iterations} iterations" if opts.smooth_iterations else "off",
         "done" if opts.smooth_iterations else "skipped")

    step("color")
    mesh.colors = sample_colors(rgb, front, mesh.vertices)
    done("color", "vertex colours projected from the photo")

    step("validate")
    checks = validate(mesh)
    checks["silhouette_iou"] = projection_iou(hull, scene)
    # Photos carry no physical scale: size the model to the requested largest dimension.
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    k = opts.size_m / float((hi - lo).max())
    mesh = Mesh(mesh.vertices * k, mesh.faces, mesh.colors)
    checks.update({"volume": checks["volume"] * k ** 3, "bbox": [(lo * k).tolist(), (hi * k).tolist()]})
    worst = min(checks["silhouette_iou"].values()) if checks["silhouette_iou"] else 1.0
    done("validate", f"watertight: {'yes' if checks['watertight'] else 'no'}; worst silhouette match {worst:.1%}")

    step("export")
    conf = Confidence.AI_GENERATED
    metadata = {
        "generator": "Vector3D Engine",
        "mode": "photo",
        "units": "m",
        "scale": {"source": "assumed", "largest_dimension_m": opts.size_m, "confidence": Confidence.ESTIMATED.value},
        "coordinate_system": {"right_handed": True, "up": "+Y", "front": "+Z", "unit": "meter"},
        "geometry_confidence": conf.value,
        "method": "relief_from_monocular_depth",
        "depth_backend": dres.backend,
        "segmentation": seg_method,
        "notes": [
            "Front surface follows AI depth (relative, no metric scale); the back is not visible and is "
            + ("mirrored from the front." if opts.back == "symmetric" else "a flattened mirror of the front."),
            "A photo is treated as an orthographic front view; strong perspective distorts the result.",
        ] + ([dres.note] if dres.note else []),
        "assumptions": hull.assumptions + [depth_note],
        "hints": {"author": hints.get("author"), "applied": applied, "notes": hints.get("notes")} if hints else None,
        "options": {"resolution": opts.resolution, "smooth_iterations": opts.smooth_iterations,
                    "thickness": opts.thickness, "detail": opts.detail, "back": opts.back, "size_m": opts.size_m},
        "views": summary(scene),
        "validation": checks,
    }
    glb = to_glb(mesh, extras={"vector3d": {"geometry_confidence": conf.value, "mode": "photo"}})
    obj = to_obj(mesh)
    done("export", f"GLB {len(glb) / 1024:.0f} KB, OBJ {len(obj) / 1024:.0f} KB")
    metadata["timings_s"] = timings
    return GenerateResult(glb, obj, metadata, timings, images)
