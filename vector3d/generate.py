"""One-click generation: input views -> validated mesh -> GLB / OBJ / metadata.

Three modes share one progress protocol:

* ai: a 3D foundation model (TripoSG, run on the local NVIDIA GPU) generates
  the shape from the image; the engine prepares the input, places, colours
  and validates the result. Chosen by auto when TripoSG is installed.

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

AI_STEPS: tuple[tuple[str, str], ...] = (
    ("load", "Read input image"),
    ("segment", "Remove background"),
    ("ai_generate", "Generate 3D shape (TripoSG, GPU)"),
    ("cleanup", "Remove floating fragments"),
    ("normalize", "Normalize all views"),
    ("fit_views", "Fit model to the views' extents"),
    ("refine", "Refine against every view's silhouette"),
    ("place", "Place and scale model"),
    ("color", "Colour from image"),
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
    mode: str = "auto"         # auto | ai | drawing | photo
    ai_profile: str = "balanced"   # ai mode: fast | balanced | quality
    ai_lora: bool = False          # ai mode: TripoSG with the fine-tuned LoRA adapter (lab finetune)
    seed: int = 42
    # photo mode
    depth_backend: str = "auto"    # auto | depth-anything | inflation
    segmenter: str = "auto"        # auto | birefnet | birefnet-lite | isnet | u2net | classic
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


def ai_available() -> tuple[bool, str]:
    from .lab.backends import get_backend

    return get_backend("triposg").check()


def resolve_mode(inputs: dict[str, tuple[str, bytes]], mode: str = "auto") -> str:
    if mode in ("ai", "drawing", "photo"):
        return mode
    if mode != "auto":
        raise ValueError(f"unknown mode {mode!r}")
    if any(is_svg(data) for _, data in inputs.values()):
        return "drawing"
    return "ai" if ai_available()[0] else "photo"


def plan_steps(mode: str) -> tuple[tuple[str, str], ...]:
    return {"photo": PHOTO_STEPS, "ai": AI_STEPS}.get(mode, STEPS)


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
        return {"photo": _generate_photo, "ai": _generate_ai}.get(mode, _generate_drawing)(inputs, opts, run)
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
    from .reconstruct.relief import ReliefParams, build_heights, heightfield_mesh, relief_field, sample_colors

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

    if not others and not hints:
        # One image: mesh the depth maps directly at image resolution, no voxel grid.
        step("carve")
        done("carve", "single image: no other views to carve with", "skipped")
        depth_note = f"depth assumed: thickness {opts.thickness:.2f} x width ({opts.back} back)"
        step("relief")
        hf, stride = heightfield_mesh(relief, mask)
        # pixel (x right, y up) -> front plane: apply the same affine as the outline
        T = front.svg_to_plane
        xy = np.column_stack([hf.vertices[:, 0], -hf.vertices[:, 1], np.ones(len(hf.vertices))]) @ T.T
        mesh = Mesh(np.column_stack([xy[:, 0], xy[:, 1], hf.vertices[:, 2] * scale]), hf.faces)
        done("relief", f"height field from depth, 1 vertex per {int(stride)} px")
        step("ai_fill")
        done("ai_fill", "no hints supplied", "skipped")
        step("surface")
        done("surface", f"{len(mesh.faces):,} triangles")
        hull, applied = None, []
    else:
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
    iters = opts.smooth_iterations if hull is not None else min(opts.smooth_iterations, 2)  # keep depth detail
    mesh = taubin_smooth(mesh, iters)
    done("smooth", f"Taubin, {iters} iterations" if iters else "off", "done" if iters else "skipped")

    step("color")
    mesh.colors = sample_colors(rgb, front, mesh.vertices)
    done("color", "vertex colours projected from the photo")

    step("validate")
    checks = validate(mesh)
    checks["silhouette_iou"] = projection_iou(hull, scene) if hull is not None else \
        {"front": front_silhouette_iou(mesh, mask)}
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
        "method": "relief_from_monocular_depth" if hull is not None else "heightfield_from_monocular_depth",
        "depth_backend": dres.backend,
        "segmentation": seg_method,
        "notes": [
            "Front surface follows AI depth (relative, no metric scale); the back is not visible and is "
            + ("mirrored from the front." if opts.back == "symmetric" else "a flattened mirror of the front."),
            "A photo is treated as an orthographic front view; strong perspective distorts the result.",
        ] + ([dres.note] if dres.note else []),
        "assumptions": (hull.assumptions if hull is not None else []) + [depth_note],
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


def front_silhouette_iou(mesh: Mesh, mask: np.ndarray, size: int = 384) -> float:
    """IoU of the mesh seen from the front against the image mask, each framed by its own bounding box."""
    from .lab.metrics import iou
    from .lab.render import render, view_camera

    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    sil = render(mesh, view_camera("front", tuple(((lo + hi) / 2).tolist()), float((hi - lo).max()) * 0.55, size)).mask
    sy, sx = np.nonzero(sil)
    ys, xs = np.nonzero(mask)
    if not len(sy) or not len(ys):
        return 0.0
    sil = sil[sy.min():sy.max() + 1, sx.min():sx.max() + 1]
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    target = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize(sil.shape[::-1])) > 127
    return iou(sil, target)


def _generate_ai(inputs, opts: GenerateOptions, run: _Runner) -> GenerateResult:
    """TripoSG on the local GPU. Fails loudly when it cannot run: never a stand-in mesh."""
    import tempfile
    from pathlib import Path

    from .lab.backends import get_backend
    from .lab.mesh_io import load_mesh

    step, done, timings = run.step, run.done, run.timings
    step("load")
    primary = "front" if "front" in inputs else next(iter(inputs))
    name, data = inputs[primary]
    if is_svg(data):
        raise ValueError("AI mode needs a raster image (PNG/JPG); use drawing mode for SVG")
    rgba = load_image(data)
    ignored = [v for v in inputs if v != primary]
    done("load", f"{name}: {rgba.shape[1]}x{rgba.shape[0]} px" +
         (f"; TripoSG sees this image, {', '.join(ignored)} fit the result" if ignored else ""))

    step("segment")
    from .ai.segment import cutout

    mask, alpha, seg_method = cutout(rgba, opts.segmenter)
    if not mask.any():
        raise ValueError("no foreground found; use an image with a clear subject")
    cut = rgba.copy()
    cut[..., 3] = np.where(mask, np.maximum(alpha, 1), alpha)  # soft edges from the matting model
    done("segment", f"{seg_method}; subject covers {mask.mean():.0%} of the image")

    step("ai_generate")
    from .lab.backends.triposg import lora_path

    lora = lora_path() if opts.ai_lora else None
    if opts.ai_lora and lora is None:
        raise RuntimeError("LoRA adapter not found: run `lab finetune` or set V3D_TRIPOSG_LORA")
    backend = get_backend("triposg", lora=str(lora) if lora else None)
    ok, why = backend.check()
    if not ok:
        raise RuntimeError(f"TripoSG is not available: {why}. Install it (docs/LAB.md) or choose photo mode.")
    run.report("ai_generate", "running", "TripoSG" + (" + LoRA" if lora else "") + " on the GPU, usually 1-2 minutes")
    run_dir = Path(tempfile.mkdtemp(prefix="v3d_ai_", dir=Path("runs").resolve() if Path("runs").is_dir() else None))
    src = run_dir / "input.png"
    Image.fromarray(cut, "RGBA").save(src)
    rec = backend.generate({"front": src}, quality_profile=opts.ai_profile, out_dir=run_dir, seed=opts.seed)
    if rec.status != "success":
        raise RuntimeError(f"TripoSG {rec.status}: {rec.reason}")
    raw = load_mesh(rec.output_path)
    done("ai_generate", f"{len(raw.faces):,} triangles in {rec.duration_s:.0f} s, peak VRAM {rec.peak_vram_mb:.0f} MB")

    from .lab.metrics import normal_consistency
    from .lab.refine import RefineParams, refine_silhouettes
    from .reconstruct.fit import drop_small_parts, fit_to_extent, target_masks, view_cameras, view_ious
    from .reconstruct.relief import sample_colors

    step("cleanup")
    mesh, removed = drop_small_parts(raw)
    done("cleanup", f"removed {removed} floating piece(s)" if removed else "one connected piece",
         "done" if removed else "skipped")

    step("normalize")
    # Every view is a measurement: the front image (as its silhouette) plus any other drawings.
    docs = {"front": mask_document(mask, seg_method, name, "front")}
    docs.update({v: load_view(v, n, d) for v, (n, d) in inputs.items() if v != primary and v != "front"})
    scene = normalize_views(docs, tolerance=opts.tolerance)
    hull = carve(scene, resolution=128)
    cams = view_cameras(scene, hull)
    targets = target_masks(scene, hull, cams)
    bad = [c for c in scene.checks if not c.ok]
    done("normalize", f"{len(scene.views)} view(s), units: {scene.units}; "
                      f"{len(scene.checks) - len(bad)}/{len(scene.checks)} cross-view checks ok")

    step("fit_views")
    # TripoSG outputs +Y up with the front toward +Z, so only scale and position change.
    ext = scene_extent(scene)
    mesh, k_axes = fit_to_extent(mesh, ext)
    iou_raw = view_ious(mesh, targets, cams)
    observed = "".join("XYZ"[a] for a in sorted(ext))
    done("fit_views", f"bounding box fitted on {observed}" + ("" if 2 in ext else "; depth Z scaled like X/Y (not observed)")
         + "; IoU " + ", ".join(f"{v} {x:.0%}" for v, x in iou_raw.items()))

    step("refine")
    iou_final, refined_used = iou_raw, False
    nc_raw = normal_consistency(mesh)
    if len(targets) < 2:
        done("refine", "one view only: nothing beyond the front silhouette to fit", "skipped")
        nc_final = nc_raw
    else:
        ref = refine_silhouettes(mesh, targets, cams, RefineParams(iterations=12, step=0.4, patience=2))
        iou_ref = view_ious(ref.mesh, targets, cams)
        nc_ref = normal_consistency(ref.mesh)
        gain = np.mean(list(iou_ref.values())) - np.mean(list(iou_raw.values()))
        if gain >= 0.02 and nc_raw - nc_ref < 0.01:
            mesh, iou_final, nc_final, refined_used = ref.mesh, iou_ref, nc_ref, True
            done("refine", f"accepted: mean IoU +{gain:.1%}; " + ", ".join(f"{v} {x:.0%}" for v, x in iou_ref.items()))
        else:
            nc_final = nc_raw
            done("refine", f"rejected (mean IoU {gain:+.1%}, smoothness {nc_ref - nc_raw:+.3f}): kept the fitted model",
                 "skipped")

    step("place")
    if scene.units == "m":
        to_model = 1.0
        done("place", "drawings carry a physical scale: kept in metres")
    else:
        lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        to_model = opts.size_m / float((hi - lo).max())
        done("place", f"largest dimension {opts.size_m:g} m (assumed scale)")

    step("color")
    mesh.colors = sample_colors(rgba[..., :3], scene.views[ViewName.FRONT], mesh.vertices)
    mesh = Mesh(mesh.vertices * to_model, mesh.faces, mesh.colors)
    done("color", "vertex colours projected from the front image (back side reuses them)")

    step("validate")
    checks = validate(mesh)
    checks["silhouette_iou"] = iou_final
    checks["silhouette_iou_before_refine"] = iou_raw
    checks["normal_consistency"] = nc_final
    worst = min(iou_final.values()) if iou_final else 1.0
    done("validate", f"watertight: {'yes' if checks['watertight'] else 'no'}; worst view silhouette match {worst:.1%}")

    step("export")
    conf = Confidence.ESTIMATED if refined_used else Confidence.AI_GENERATED
    metadata = {
        "generator": "Vector3D Engine",
        "mode": "ai",
        "units": "m",
        "scale": {"source": "assumed", "largest_dimension_m": opts.size_m, "confidence": Confidence.ESTIMATED.value},
        "coordinate_system": {"right_handed": True, "up": "+Y", "front": "+Z", "unit": "meter"},
        "geometry_confidence": conf.value,
        "method": "triposg_image_to_3d",
        "model": {"name": rec.model, "version": rec.model_version, "license": rec.license,
                  "profile": rec.quality_profile, "seed": rec.seed, "device": rec.device,
                  "lora": str(lora) if lora else None,
                  "duration_s": rec.duration_s, "load_s": rec.load_s, "peak_vram_mb": rec.peak_vram_mb,
                  "raw_output": rec.output_path},
        "segmentation": seg_method,
        "fit": {"views": sorted(targets), "observed_axes": observed, "axis_scale": k_axes,
                "refined": refined_used, "removed_fragments": removed,
                "drawing_to_model_scale": to_model,
                "label": "ESTIMATED where the views constrain the shape; the rest is the model's guess"},
        "notes": ["Shape generated by TripoSG from the front image only; the other views fit its size and outline.",
                  "Colours are projected from the front image, so the back repeats the front."]
                 + hull.assumptions + rec.warnings,
        "views": summary(scene),
        "validation": checks,
    }
    glb = to_glb(mesh, extras={"vector3d": {"geometry_confidence": conf.value, "mode": "ai", "model": rec.model}})
    obj = to_obj(mesh)
    done("export", f"GLB {len(glb) / 1024:.0f} KB, OBJ {len(obj) / 1024:.0f} KB")
    metadata["timings_s"] = timings
    return GenerateResult(glb, obj, metadata, timings, {"mask.png": _png((mask * 255).astype(np.uint8))})
