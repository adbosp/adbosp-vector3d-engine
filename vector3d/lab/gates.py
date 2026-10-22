"""Acceptance gates for a generated mesh file.

Each gate is "pass", "fail" or "skipped" with a reason. A result is
"accepted" only when no gate fails; skipped gates are listed so a missing
tool (e.g. Blender) is visible rather than counted as a pass.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .mesh_io import load_mesh
from .metrics import mesh_quality
from .render import contact_sheet, frame, render, shade, view_camera

PREVIEW_VIEWS = ("front", "back", "left", "right")
REQUIRED_METADATA = ("model", "version")

BLENDER_SCRIPT = """
import bpy, sys, json
path, out = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
tris = sum(len(o.data.polygons) for o in objs)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB')
print('V3D_BLENDER ' + json.dumps({'meshes': len(objs), 'polygons': tris}))
"""


def _gate(status: str, detail="") -> dict:
    return {"status": status, "detail": detail}


def blender_roundtrip(path: Path, work: Path, blender: str | None = None, timeout: int = 180) -> dict:
    exe = blender or shutil.which("blender")
    if not exe:
        return _gate("skipped", "blender not found on PATH")
    script = work / "_blender_roundtrip.py"
    script.write_text(BLENDER_SCRIPT)
    out = work / "blender_roundtrip.glb"
    try:
        p = subprocess.run([exe, "-b", "--python", str(script), "--", str(path), str(out)],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return _gate("fail", f"blender did not finish: {e}")
    line = next((l for l in p.stdout.splitlines() if l.startswith("V3D_BLENDER ")), None)
    if p.returncode != 0 or line is None or not out.exists():
        return _gate("fail", (p.stderr or p.stdout)[-500:])
    info = json.loads(line.split(" ", 1)[1])
    return _gate("pass" if info["meshes"] > 0 and info["polygons"] > 0 else "fail", info)


def run_gates(mesh_path: str | Path, work_dir: str | Path, metadata: dict | None = None,
              reference_masks: dict | None = None, render_previews: bool = True,
              min_reference_iou: float = 0.5, blender: str | None = None) -> dict:
    mesh_path, work = Path(mesh_path), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    gates: dict[str, dict] = {}
    try:
        mesh = load_mesh(mesh_path)
        gates["readable_mesh"] = _gate("pass", f"{len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    except Exception as e:  # noqa: BLE001 - any loader error is a failed gate
        gates["readable_mesh"] = _gate("fail", str(e))
        return {"accepted": False, "gates": gates}
    gates["actual_geometry"] = _gate("pass" if len(mesh.faces) > 0 and mesh_quality(mesh)["volume"] != 0 else "fail")
    finite = bool(np.isfinite(mesh.vertices).all())
    gates["finite_coordinates"] = _gate("pass" if finite else "fail", "" if finite else "NaN/Inf vertices")
    q = mesh_quality(mesh) if finite else {}
    (work / "validation.json").write_text(json.dumps(q, indent=1, default=str))
    gates["validation_report"] = _gate("pass", "validation.json")
    meta = metadata or {}
    missing = [k for k in REQUIRED_METADATA if not meta.get(k)]
    gates["model_version_recorded"] = _gate("fail" if missing else "pass", f"missing {missing}" if missing else meta)
    if "duration_s" in meta or "peak_vram_mb" in meta:
        gates["runtime_recorded"] = _gate("pass", {k: meta.get(k) for k in ("duration_s", "peak_vram_mb")})
    else:
        gates["runtime_recorded"] = _gate("skipped", "no runtime in metadata (not a model run)")
    if render_previews and finite:
        center, half = frame(mesh)
        tiles = [shade(mesh, render(mesh, view_camera(v, center, half, 384))) for v in PREVIEW_VIEWS]
        for v, t in zip(PREVIEW_VIEWS, tiles):
            Image.fromarray(t, "RGBA").save(work / f"preview_{v}.png")
        Image.fromarray(contact_sheet(tiles, list(PREVIEW_VIEWS))).save(work / "previews.png")
        gates["previews_rendered"] = _gate("pass", [f"preview_{v}.png" for v in PREVIEW_VIEWS])
    else:
        gates["previews_rendered"] = _gate("skipped", "previews not requested")
    if reference_masks:
        from .metrics import iou

        scores = {v: iou(render(mesh, cam).mask, m) for v, (cam, m) in reference_masks.items()}
        ok = min(scores.values()) >= min_reference_iou
        gates["input_view_comparison"] = _gate("pass" if ok else "fail", scores)
    else:
        gates["input_view_comparison"] = _gate("skipped", "no calibrated reference masks")
    gates["blender_roundtrip"] = blender_roundtrip(mesh_path, work, blender)
    return {"accepted": not any(g["status"] == "fail" for g in gates.values()), "gates": gates}
