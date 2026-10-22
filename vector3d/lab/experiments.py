"""Experiment 001: silhouette refinement on a simulated degraded baseline.

No foundation-model output exists yet in this environment (no GPU, model
hosts blocked), so the "initial geometry" here is the ground truth after a
seeded, smooth, low-frequency deformation that imitates typical shape
errors: wrong proportions, bulges and lost thin detail. It is labelled
SIMULATED_BASELINE everywhere and is never presented as AI output.

The refinement only sees the four reference silhouettes. Ground truth is
used afterwards for Chamfer/F-score, and four held-out views (never given
to the optimiser) show whether the change generalises or just overfits the
reference views.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..export.writers import to_glb
from ..reconstruct.surface import Mesh, taubin_smooth
from . import dataset
from .gates import run_gates
from .mesh_io import load_mesh
from .metrics import chamfer_fscore, iou, mesh_quality
from .refine import RefineParams, refine_silhouettes
from .render import contact_sheet, render, shade, silhouette


def degrade(mesh: Mesh, seed: int = 0, strength: float = 1.0) -> tuple[Mesh, dict]:
    """Seeded smooth deformation: anisotropic scale, three sine bulges, oversmoothing."""
    rng = np.random.default_rng(seed)
    v = mesh.vertices
    c = (v.max(0) + v.min(0)) / 2
    ext = float((v.max(0) - v.min(0)).max())
    scale = 1 + strength * rng.uniform(-0.12, 0.12, 3)
    out = (v - c) * scale + c
    waves = []
    for _ in range(3):
        k = rng.normal(size=3)
        k = k / np.linalg.norm(k) * rng.uniform(2.0, 4.0) * np.pi / ext
        amp = strength * rng.uniform(0.015, 0.03) * ext
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        phase = rng.uniform(0, 2 * np.pi)
        out = out + amp * np.sin((v - c) @ k + phase)[:, None] * direction
        waves.append({"k": k.tolist(), "amp": amp, "dir": direction.tolist()})
    smooth = int(round(8 * strength))
    deg = taubin_smooth(Mesh(out, mesh.faces), smooth, lam=0.6, mu=-0.4)  # deliberately not volume preserving
    return deg, {"label": "SIMULATED_BASELINE", "seed": seed, "strength": strength,
                 "anisotropic_scale": scale.tolist(), "waves": waves, "oversmooth_passes": smooth}


def evaluate(mesh: Mesh, gt: Mesh, cams, targets, heldout_targets) -> dict:
    ref = {v: iou(silhouette(mesh, cams[v]), m) for v, m in targets.items()}
    held = {v: iou(silhouette(mesh, cams[v]), m) for v, m in heldout_targets.items()}
    q = mesh_quality(mesh)
    return {"ref_iou": ref, "ref_iou_mean": float(np.mean(list(ref.values()))),
            "heldout_iou": held, "heldout_iou_mean": float(np.mean(list(held.values()))),
            **chamfer_fscore(mesh, gt),
            "normal_consistency": q["normal_consistency"], "non_manifold_edges": q["non_manifold_edges"],
            "watertight": q["watertight"]}


def run_silhouette_experiment(data_dir: str | Path, out_dir: str | Path, seeds=(0, 1),
                              params: RefineParams | None = None, names=None) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    params = params or RefineParams()
    rows = []
    for obj in manifest["objects"]:
        if names and obj["name"] not in names:
            continue
        od = data_dir / obj["name"]
        gt = load_mesh(od / "gt.glb")
        cams = dataset.load_cameras(od)
        targets = dataset.load_masks(od, manifest["reference_views"])
        held = dataset.load_masks(od, manifest["heldout_views"])
        for seed in seeds:
            base, deg_info = degrade(gt, seed)
            t0 = time.perf_counter()
            res = refine_silhouettes(base, targets, cams, params)
            dt = time.perf_counter() - t0
            before, after = evaluate(base, gt, cams, targets, held), evaluate(res.mesh, gt, cams, targets, held)
            run = out_dir / f"{obj['name']}_seed{seed}"
            run.mkdir(exist_ok=True)
            (run / "baseline_simulated.glb").write_bytes(to_glb(base, "baseline", {"label": "SIMULATED_BASELINE"}))
            (run / "refined.glb").write_bytes(to_glb(res.mesh, "refined", {"label": "ESTIMATED",
                                                                           "source": "silhouette refinement"}))
            views = ["front", "left", "az45_el20", "az135_el20"]
            tiles = [shade(m, render(m, cams[v])) for m in (gt, base, res.mesh) for v in views]
            labels = [f"{k} {v}" for k in ("GT", "simulated", "refined") for v in views]
            Image.fromarray(contact_sheet(tiles, labels)).save(run / "compare.png")
            gates = run_gates(run / "refined.glb", run, metadata={"model": "silhouette_refine",
                                                                 "version": "exp001"}, render_previews=False)
            row = {"object": obj["name"], "category": obj["category"], "seed": seed, "degradation": deg_info,
                   "before": before, "after": after, "refine_seconds": dt, "best_iteration": res.best_iteration,
                   "history": res.history, "gates": gates}
            (run / "result.json").write_text(json.dumps(row, indent=1))
            rows.append(row)
    summary = {"experiment": "exp001_silhouette_refinement", "params": vars(params),
               "baseline": "SIMULATED_BASELINE (deformed ground truth, not AI output)",
               "dataset": manifest["version"], "runs": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
