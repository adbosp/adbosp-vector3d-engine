"""Fine-tuning TripoSG with the engine's geometry: data, training run, evaluation hook.

Measured weakness (research/benchmark_report_rtx3060.md): from one image
TripoSG flattens boxy hard-surface objects (the crate became a thin plate)
and misjudges depth. The engine can generate unlimited objects with exact
geometry, so it can teach that missing depth prior:

  build_dataset  procedural objects (hard-surface boxes, panels, holes, L
                 brackets, plates, cups, limbs) -> per sample cond.png (front
                 view, random tint) + surface.npz (points, outward normals)
  train          runs backends/triposg_finetune.py in the TripoSG
                 environment: LoRA on the flow transformer, base weights frozen
  evaluation     `lab benchmark --lora DIR` runs the same synthetic_v1
                 benchmark with the adapter, so base and fine-tuned are
                 compared on identical inputs. synthetic_v1 objects are
                 hand-written SDFs never produced by these generators.

All training data is generated here (no third-party assets). The adapter is
a derivative of TripoSG (MIT).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..reconstruct.surface import Mesh
from . import dataset as ds
from .distill import random_sdf
from .render import frame, orbit_camera, render, shade

TRAINER = Path(__file__).with_name("backends") / "triposg_finetune.py"


def hard_surface_sdf(rng: np.random.Generator):
    """Boxy object with depth the front view cannot show: crates, cabinets, brackets, plates."""
    kind = rng.choice(["crate", "bracket", "plate", "cabinet", "stack"])
    sx, sy, sz = rng.uniform(0.15, 0.45, 3)
    bevel = rng.uniform(0, 0.04)
    if kind == "plate":                                        # thin in a random axis
        dims = [sx, sy, sz]
        dims[int(rng.integers(3))] = rng.uniform(0.03, 0.08)
        sx, sy, sz = dims
    rec = rng.random() < 0.6
    rec_face = rng.choice(["front", "side", "top"])
    holes = int(rng.integers(0, 3))
    hole_xy = rng.uniform(-0.5, 0.5, (holes, 2))
    hole_r = rng.uniform(0.03, 0.08, holes)
    knob = rng.random() < 0.35
    knob_at = rng.uniform(-1, 1, 3)
    leg_kind = rng.random() < 0.3

    def sdf(p):
        d = ds.sd_box(p, (sx, sy, sz), min(bevel, 0.5 * min(sx, sy, sz)))
        if kind == "bracket":
            arm = ds.sd_box(ds._t(p, x=-sx + 0.06, y=sy * 1.4), (0.06, sy * 1.4, sz), bevel)
            d = np.minimum(np.maximum(d, p[..., 1] - (-sy + 2 * min(0.08, sy))), arm)
        if kind == "stack":
            top = ds.sd_box(ds._t(p, y=sy + 0.5 * sy), (0.7 * sx, 0.5 * sy, 0.7 * sz), bevel)
            d = np.minimum(d, top)
        if rec:
            if rec_face == "front":
                cut = ds.sd_box(ds._t(p, z=sz), (0.75 * sx, 0.7 * sy, min(0.05, 0.4 * sz)))
            elif rec_face == "side":
                cut = ds.sd_box(ds._t(p, x=sx), (min(0.05, 0.4 * sx), 0.7 * sy, 0.75 * sz))
            else:
                cut = ds.sd_box(ds._t(p, y=sy), (0.75 * sx, min(0.05, 0.4 * sy), 0.75 * sz))
            d = np.maximum(d, -cut)
        for (hx, hz), r in zip(hole_xy, hole_r):           # vertical through holes
            d = np.maximum(d, -(np.linalg.norm(ds._t(p, x=hx * sx, z=hz * sz)[..., [0, 2]], axis=-1) - r))
        if knob:
            c = knob_at * np.array([sx, sy, sz])
            c[int(np.argmax(np.abs(knob_at)))] = np.sign(knob_at[int(np.argmax(np.abs(knob_at)))]) * \
                [sx, sy, sz][int(np.argmax(np.abs(knob_at)))]
            d = np.minimum(d, ds.sd_sphere(p - c, 0.06))
        if leg_kind and kind in ("cabinet", "crate"):
            for lx in (-1, 1):
                for lz in (-1, 1):
                    d = np.minimum(d, ds.sd_cyl_y(ds._t(p, x=lx * 0.8 * sx, y=-sy - 0.1, z=lz * 0.8 * sz), 0.025, 0.1))
        return d

    return sdf


def surface_samples(mesh: Mesh, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted points on the mesh with outward face normals."""
    v, f = mesh.vertices, mesh.faces
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    nrm = cross / np.maximum(2 * area[:, None], 1e-30)
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)
    if vol < 0:                                                # make the normals point outwards
        nrm = -nrm
    k = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.column_stack([1 - s, s * (1 - r2), s * r2])
    pts = np.einsum("ni,nij->nj", w, tri[k])
    return pts.astype(np.float32), nrm[k].astype(np.float32)


def build_dataset(out_dir, count: int, seed: int = 0, image_size: int = 512, points: int = 40960,
                  hard_fraction: float = 0.6) -> dict:
    """Write `count` samples (cond.png + surface.npz) and a manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.perf_counter()
    for i in range(count):
        name = f"s{seed}_{i:05d}"
        d = out / name
        if (d / "surface.npz").exists():
            rows.append({"name": name, "status": "exists"})
            continue
        rng = np.random.default_rng([seed, i, 7])
        hard = rng.random() < hard_fraction
        try:
            mesh = ds.mesh_sdf((hard_surface_sdf if hard else random_sdf)(rng), 112)
        except ValueError:
            rows.append({"name": name, "status": "empty"})
            continue
        mesh = Mesh(mesh.vertices, mesh.faces, np.tile(rng.uniform(0.35, 0.95, 3), (len(mesh.vertices), 1)))
        center, half = frame(mesh)
        az = float(rng.normal(0, 8))                           # small yaw jitter; still the front view
        cam = orbit_camera(az, float(rng.normal(0, 5)), center, half, image_size, "front")
        r = render(mesh, cam)
        d.mkdir(exist_ok=True)
        Image.fromarray(shade(mesh, r), "RGBA").save(d / "cond.png")
        pts, nrm = surface_samples(mesh, points, rng)
        yaw = np.radians(az)                                   # rotate GT into the camera's frame
        rot = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
        np.savez_compressed(d / "surface.npz", points=pts @ rot.T, normals=nrm @ rot.T)
        rows.append({"name": name, "status": "ok", "kind": "hard_surface" if hard else "generic", "yaw_deg": az})
    manifest = {"version": "v3d_finetune_v1", "seed": seed, "count": count, "image_size": image_size,
                "points": points, "license": "procedurally generated by vector3d/lab; no third-party assets",
                "seconds": time.perf_counter() - t0, "samples": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def train(data_dir, out_dir, steps: int = 1500, rank: int = 16, lr: float = 1e-4, accum: int = 4,
          repo=None, python=None, weights=None, tiny: bool = False, vram_fraction: float = 0.9,
          timeout: int = 6 * 3600) -> dict:
    """Run the trainer in the TripoSG environment; returns its result JSON."""
    from .backends.triposg import TripoSGBackend

    be = TripoSGBackend(repo, python, weights)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not tiny:
        ok, why = be.check()
        if not ok:
            return {"status": "blocked", "reason": why}
    result = out / "train_result.json"
    cmd = [sys.executable if tiny else be.python, str(TRAINER),
           "--repo", str(be.repo), "--data", str(data_dir), "--out", str(out / "lora"), "--result", str(result),
           "--steps", str(steps), "--rank", str(rank), "--lr", str(lr), "--accum", str(accum),
           "--vram-fraction", str(vram_fraction)]
    cmd += ["--tiny"] if tiny else ["--weights", str(be.weights)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=timeout)
    (out / "train.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    if not result.exists():
        return {"status": "failed", "reason": f"trainer exited {p.returncode} without a result"}
    return json.loads(result.read_text())
