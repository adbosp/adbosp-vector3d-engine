"""Benchmark a real backend on the synthetic set.

For each object the backend gets the front reference PNG (single-image
task). Its raw mesh is placed in the reference frame without ground truth
(align_to_silhouettes), then measured as:
  A  raw foundation-model output
  B  A + silhouette refinement against the four reference views
Ground truth is used only for scoring. Meshy or any other external output is
never estimated; it enters only as real files through `external`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..export.writers import to_glb
from . import dataset
from .align import align_to_silhouettes
from .backends import get_backend
from .experiments import evaluate
from .gates import run_gates
from .mesh_io import load_mesh
from .refine import refine_silhouettes
from .render import contact_sheet, render, shade, turntable


def run_benchmark(data_dir, out_dir, backend: str = "triposg", profile: str = "balanced", seed: int = 42,
                  names=None, backend_kwargs: dict | None = None) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    be = get_backend(backend, **(backend_kwargs or {}))
    rows = []
    for obj in manifest["objects"]:
        if names and obj["name"] not in names:
            continue
        od, run = data_dir / obj["name"], out_dir / obj["name"]
        rec = be.generate({"front": od / "views" / "front.png"}, quality_profile=profile, out_dir=run, seed=seed)
        row = {"object": obj["name"], "category": obj["category"], "record": rec.to_dict()}
        if rec.status != "success":
            rows.append(row)
            continue
        gt = load_mesh(od / "gt.glb")
        cams = dataset.load_cameras(od)
        targets = dataset.load_masks(od, manifest["reference_views"])
        held = dataset.load_masks(od, manifest["heldout_views"])
        raw = load_mesh(rec.output_path)
        placed, place_info = align_to_silhouettes(raw, targets, cams)
        (run / "A_placed.glb").write_bytes(to_glb(placed, "A", {"label": "AI_GENERATED", "placement": place_info}))
        t0 = time.perf_counter()
        ref = refine_silhouettes(placed, targets, cams)
        (run / "B_refined.glb").write_bytes(to_glb(ref.mesh, "B", {"label": "ESTIMATED", "from": "A_placed.glb"}))
        row.update(placement=place_info,
                   A=evaluate(placed, gt, cams, targets, held),
                   B=evaluate(ref.mesh, gt, cams, targets, held),
                   refine_seconds=time.perf_counter() - t0, refine_history=ref.history,
                   gates=run_gates(run / "A_placed.glb", run / "gates", metadata={
                       "model": rec.model, "version": rec.model_version, "duration_s": rec.duration_s,
                       "peak_vram_mb": rec.peak_vram_mb},
                       reference_masks={v: (cams[v], targets[v]) for v in targets}),
                   output_bytes=Path(rec.output_path).stat().st_size)
        frames = turntable(placed, 8)
        Image.fromarray(contact_sheet([f for _, f in frames], [n for n, _ in frames])).save(run / "turntable_A.png")
        views = ["front", "left", "back", "az45_el20"]
        tiles = [shade(m, render(m, cams[v])) for m in (gt, placed, ref.mesh) for v in views]
        Image.fromarray(contact_sheet(tiles, [f"{k} {v}" for k in ("GT", "A raw", "B refined") for v in views])
                        ).save(run / "compare.png")
        rows.append(row)
    summary = {"benchmark": "synthetic_v1 single-image", "backend": backend, "profile": profile, "seed": seed,
               "dataset": manifest["version"], "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


METRICS = (("chamfer_l1", "Chamfer-L1", False), ("fscore@0.02", "F@0.02", True),
           ("heldout_iou_mean", "held-out IoU", True), ("normal_consistency", "normal cons.", True))


def compare_summaries(a_dir, b_dir, labels=("base", "lora")) -> str:
    """Markdown table of the raw-model (A) metrics of two benchmark runs on the same objects."""
    runs = [json.loads((Path(d) / "summary.json").read_text()) for d in (a_dir, b_dir)]
    rows = [{r["object"]: r for r in s["rows"]} for s in runs]
    names = [n for n in rows[0] if n in rows[1]]
    head = "| object | " + " | ".join(f"{m} {l}" for _, m, _ in METRICS for l in labels) + " | s " + \
           " | s ".join(labels) + " |"
    lines = [head, "|" + "---|" * (1 + 2 * len(METRICS) + 2)]
    sums = {(k, i): [] for k, _, _ in METRICS for i in range(2)}
    wins = {k: 0 for k, _, _ in METRICS}
    for n in names:
        a, b = rows[0][n], rows[1][n]
        if "A" not in a or "A" not in b:
            lines.append(f"| {n} | " + " | ".join("failed" for _ in range(2 * len(METRICS) + 2)) + " |")
            continue
        cells = []
        for k, _, higher in METRICS:
            va, vb = a["A"][k], b["A"][k]
            sums[(k, 0)].append(va)
            sums[(k, 1)].append(vb)
            wins[k] += (vb > va) if higher else (vb < va)
            cells += [f"{va:.4f}", f"{vb:.4f}"]
        cells += [f"{(x['record'].get('duration_s') or 0):.0f}" for x in (a, b)]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    mean = [f"{np.mean(sums[(k, i)]):.4f}" if sums[(k, i)] else "-" for k, _, _ in METRICS for i in range(2)]
    lines.append("| **mean** | " + " | ".join(mean) + " | | |")
    lines.append("")
    lines.append(f"{labels[1]} better on: " + ", ".join(f"{m} {wins[k]}/{len(names)}" for k, m, _ in METRICS))
    return "\n".join(lines)
