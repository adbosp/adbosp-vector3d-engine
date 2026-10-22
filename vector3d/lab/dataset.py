"""Synthetic benchmark set with exact ground truth.

Each object is a signed distance function (negative inside) built from
primitives, meshed with marching cubes and rendered from controlled
orthographic cameras. The reference PNGs are therefore pixel-exact
projections of the ground truth, and every number measured against them is
reproducible from this file alone. The set covers the categories the lab
targets: hard-surface props, curved furniture, a stylized character,
concavities, asymmetry and thin structures. It does not include human
faces or real scanned assets; see research/benchmark_report.md.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..reconstruct.hull import Grid
from ..reconstruct.surface import Mesh, field_to_mesh
from .render import frame, orbit_camera, render, shade, view_camera

REFERENCE_VIEWS = ("front", "back", "left", "right")
# Never shown to a refinement method; used to catch overfitting to the references.
HELDOUT_VIEWS = (("az45_el20", 45, 20), ("az135_el20", 135, 20), ("az225_el-15", 225, -15), ("top", None, None))


# --- SDF primitives (p: (...,3) points) ---------------------------------
def sd_box(p, half, r=0.0):
    q = np.abs(p) - (np.asarray(half) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0) - r


def sd_sphere(p, r):
    return np.linalg.norm(p, axis=-1) - r


def sd_capsule(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def sd_cyl_y(p, r, half_h):
    d = np.stack([np.linalg.norm(p[..., [0, 2]], axis=-1) - r, np.abs(p[..., 1]) - half_h], -1)
    return np.minimum(d.max(-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


def sd_torus_z(p, R, r):
    """Torus in the XY plane (axis along Z)."""
    q = np.stack([np.linalg.norm(p[..., :2], axis=-1) - R, p[..., 2]], -1)
    return np.linalg.norm(q, axis=-1) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _t(p, x=0.0, y=0.0, z=0.0):
    return p - np.array([x, y, z])


# --- Objects -------------------------------------------------------------
def crate(p):
    """Hard-surface prop: bevelled box with a recessed front panel."""
    d = sd_box(p, (0.45, 0.35, 0.35), 0.03)
    return np.maximum(d, -sd_box(_t(p, z=0.36), (0.33, 0.23, 0.04)))


def chair(p):
    """Curved furniture: seat, curved backrest shell, four thin legs."""
    seat = sd_box(_t(p, y=-0.05), (0.32, 0.04, 0.30), 0.02)
    q = _t(p, y=0.25, z=0.55)
    shell = np.abs(np.linalg.norm(q[..., [0, 2]], axis=-1) - 0.85) - 0.025
    back = np.maximum(shell, np.maximum(np.abs(q[..., 1]) - 0.26, np.abs(q[..., 0]) - 0.30))
    back = np.maximum(back, q[..., 2] + 0.6)  # keep only the rear arc of the ring
    legs = np.min([sd_cyl_y(_t(p, sx * 0.27, -0.32, sz * 0.25), 0.025, 0.25)
                   for sx in (-1, 1) for sz in (-1, 1)], axis=0)
    return np.minimum(np.minimum(seat, back), legs)


def mug(p):
    """Concavity plus asymmetry: open cup with a handle on the object's left (+X)."""
    body = sd_cyl_y(p, 0.30, 0.35)
    inner = sd_cyl_y(_t(p, y=0.06), 0.25, 0.35)
    cup = np.maximum(body, -inner)
    handle = sd_torus_z(_t(p, x=0.33, y=0.02), 0.17, 0.045)
    handle = np.maximum(handle, -(p[..., 0] - 0.30))
    return np.minimum(cup, handle)


def character(p):
    """Stylized humanoid: big head, capsule limbs, smooth unions, slight pose asymmetry."""
    head = sd_sphere(_t(p, y=0.52), 0.20)
    body = sd_capsule(p, (0, -0.05, 0), (0, 0.25, 0), 0.17)
    arm_l = sd_capsule(p, (0.17, 0.22, 0), (0.42, 0.02, 0.05), 0.06)
    arm_r = sd_capsule(p, (-0.17, 0.22, 0), (-0.38, 0.30, 0.12), 0.06)
    leg_l = sd_capsule(p, (0.08, -0.15, 0), (0.10, -0.50, 0.02), 0.07)
    leg_r = sd_capsule(p, (-0.08, -0.15, 0), (-0.10, -0.50, -0.02), 0.07)
    nose = sd_sphere(_t(p, y=0.52, z=0.20), 0.04)
    d = smin(head, body, 0.06)
    for part in (arm_l, arm_r, leg_l, leg_r):
        d = smin(d, part, 0.05)
    return np.minimum(d, nose)


def lamp(p):
    """Thin structures: disc base, 2.5 cm pole, cone shade shell."""
    base = sd_cyl_y(_t(p, y=-0.45), 0.22, 0.03)
    pole = sd_cyl_y(_t(p, y=-0.05), 0.0125, 0.40)
    q = _t(p, y=0.38)
    rad = np.linalg.norm(q[..., [0, 2]], axis=-1)
    cone_r = 0.12 + (0.15 - q[..., 1]) * 0.6  # wider at the bottom
    shade_ = np.maximum(np.abs(rad - cone_r) - 0.012, np.abs(q[..., 1]) - 0.15)
    return np.minimum(np.minimum(base, pole), shade_)


def bracket(p):
    """Asymmetric hard-surface: L bracket with a through hole and a knob."""
    a = sd_box(_t(p, y=-0.30), (0.45, 0.08, 0.25), 0.015)
    b = sd_box(_t(p, x=-0.37, y=0.05), (0.08, 0.35, 0.25), 0.015)
    d = np.minimum(a, b)
    hole = np.linalg.norm(_t(p, x=0.15, z=0.0)[..., [0, 2]], axis=-1) - 0.09
    d = np.maximum(d, -hole)
    return np.minimum(d, sd_sphere(_t(p, x=-0.37, y=0.25, z=0.27), 0.08))


@dataclass(frozen=True)
class SyntheticObject:
    name: str
    category: str
    sdf: Callable


OBJECTS = (
    SyntheticObject("crate", "hard_surface", crate),
    SyntheticObject("chair", "curved_furniture", chair),
    SyntheticObject("mug", "concavity_asymmetric", mug),
    SyntheticObject("character", "stylized_character", character),
    SyntheticObject("lamp", "thin_structure", lamp),
    SyntheticObject("bracket", "asymmetric_hard_surface", bracket),
)


def mesh_sdf(sdf: Callable, resolution: int = 112, bound: float = 0.75) -> Mesh:
    h = 2 * bound / resolution
    c = -bound + (np.arange(resolution) + 0.5) * h
    p = np.stack(np.meshgrid(c, c, c, indexing="ij"), -1)
    grid = Grid(np.full(3, -bound), np.full(3, h), (resolution,) * 3)
    return field_to_mesh(-sdf(p).astype(np.float32), grid)


def cameras_for(mesh: Mesh, size: int = 256) -> dict:
    center, half = frame(mesh)
    cams = {v: view_camera(v, center, half, size) for v in REFERENCE_VIEWS}
    for name, az, el in HELDOUT_VIEWS:
        cams[name] = view_camera("top", center, half, size) if az is None else orbit_camera(az, el, center, half, size, name)
    return cams


def build(out_dir: str | Path, resolution: int = 112, size: int = 256, names=None) -> dict:
    """Write gt.glb, reference PNGs (RGBA), masks and cameras.json per object."""
    from ..export.writers import to_glb

    out = Path(out_dir)
    manifest = {"version": "synthetic_v1", "resolution": resolution, "image_size": size,
                "reference_views": list(REFERENCE_VIEWS), "heldout_views": [h[0] for h in HELDOUT_VIEWS],
                "license": "procedurally generated by vector3d/lab/dataset.py; no third-party assets",
                "objects": []}
    for obj in OBJECTS:
        if names and obj.name not in names:
            continue
        d = out / obj.name
        (d / "views").mkdir(parents=True, exist_ok=True)
        mesh = mesh_sdf(obj.sdf, resolution)
        glb = to_glb(mesh, obj.name, {"label": "GROUND_TRUTH", "source": "synthetic SDF"})
        (d / "gt.glb").write_bytes(glb)
        cams = cameras_for(mesh, size)
        for name, cam in cams.items():
            r = render(mesh, cam)
            Image.fromarray(shade(mesh, r), "RGBA").save(d / "views" / f"{name}.png")
        (d / "cameras.json").write_text(json.dumps({k: c.to_dict() for k, c in cams.items()}, indent=1))
        manifest["objects"].append({"name": obj.name, "category": obj.category, "triangles": int(len(mesh.faces)),
                                    "gt_sha256": hashlib.sha256(glb).hexdigest()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_masks(obj_dir: str | Path, views) -> dict[str, np.ndarray]:
    d = Path(obj_dir) / "views"
    return {v: np.asarray(Image.open(d / f"{v}.png"))[..., 3] > 127 for v in views}


def load_cameras(obj_dir: str | Path):
    from .render import OrthoCamera

    raw = json.loads((Path(obj_dir) / "cameras.json").read_text())
    return {k: OrthoCamera.from_dict(v) for k, v in raw.items()}
