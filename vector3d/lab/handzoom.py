"""Hand close-up path prototype on the synthetic hand (humanoid.py), three methods, one metric set:
  whole_body   TripoSG on a front render of the 1.75 m figure with the hand beside it; hand region cropped
  closeup      TripoSG on a front render of the hand alone (hand fills the frame)
  closeup_hi3d closeup output refined by Hi3D v1.0 at 512 with the same close-up image
Every output is aligned to the ground truth the same way before measuring: bbox fit, then similarity ICP
(scale + rotation + translation). That alignment uses the GT and is for evaluation only; it is not part
of any generation path. Results are observations on one synthetic hand, not claims about real people.
"""
from __future__ import annotations

import json
import shlex
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..reconstruct.surface import Mesh
from . import humanoid
from .mesh_io import load_mesh
from .render import contact_sheet, frame, orbit_camera, render, shade
from .vae_ceiling import HAND_VIEW_CENTRE, bbox_fit, compare_hand, crop_to_hand, figure_with_hand, hand_pictures


def render_rgba(mesh: Mesh, size: int = 1024, margin: float = 1.15) -> Image.Image:
    """Front (az 0) Lambert render with a real alpha channel, framed on the mesh."""
    c, h = frame(mesh, margin)
    r = render(mesh, orbit_camera(0, 0, c, h, size))
    rgb = shade(mesh, r)[..., :3]
    return Image.fromarray(np.dstack([rgb, (r.face_id >= 0).astype(np.uint8) * 255]), "RGBA")


def similarity_icp(src: Mesh, dst: Mesh, iters: int = 30, n: int = 20000, seed: int = 0,
                   scale: bool = True) -> tuple[Mesh, dict]:
    """Point-to-point ICP (Umeyama per step) from src onto dst; scale=False keeps it rigid."""
    rng = np.random.default_rng(seed)
    sp = src.vertices[rng.choice(len(src.vertices), min(n, len(src.vertices)), replace=False)]
    tree = cKDTree(dst.vertices[rng.choice(len(dst.vertices), min(n, len(dst.vertices)), replace=False)])
    S, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iters):
        cur = sp @ (S * R).T + t
        _, i = tree.query(cur)
        tgt = tree.data[i]
        s2, R2, t2 = umeyama(cur, tgt)
        if not scale:
            s2, t2 = 1.0, tgt.mean(0) - R2 @ cur.mean(0)
        S, R, t = s2 * S, R2 @ R, s2 * (R2 @ t) + t2
    v = src.vertices @ (S * R).T + t
    return Mesh(v, src.faces), {"scale": float(S), "rotation": R.tolist(), "translation": t.tolist(), "iters": iters}


def umeyama(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R a + t - b|^2."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, D, Vt = np.linalg.svd(B.T @ A / len(a))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / (A ** 2).sum(1).mean())
    return s, R, mb - s * R @ ma


def align_eval(pred: Mesh, ref: Mesh) -> tuple[Mesh, dict]:
    """Evaluation-only alignment: bbox fit then similarity ICP onto the reference."""
    m, b = bbox_fit(pred, ref)
    m, i = similarity_icp(m, ref)
    return m, {"bbox": b, "icp": i, "note": "uses GT; evaluation only"}


def run(out_dir, body_path, hi3d_wsl=None, triposg_kw: dict | None = None, profile: str = "quality",
        seed: int = 42) -> dict:
    """Generate and measure all three methods into a new run folder. hi3d_wsl: callable(image, coarse, out) ->
    runner record, or None to skip Hi3D (recorded as skipped)."""
    from .backends.triposg import TripoSGBackend

    out = Path(out_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True)
    p = humanoid.HandParams()
    hand = humanoid.hand_mesh(p)
    fig, off = figure_with_hand(hand, load_mesh(body_path))
    render_rgba(fig).save(out / "whole_body.png")
    render_rgba(hand).save(out / "closeup.png")
    be = TripoSGBackend(**(triposg_kw or {}))
    report = {"run_dir": str(out), "profile": profile, "seed": seed, "methods": {}}
    pics = {"input": hand}

    for name, img, ref, offset in (("whole_body", "whole_body.png", fig, off), ("closeup", "closeup.png", hand, None)):
        rec = be.generate({"front": out / img}, quality_profile=profile, out_dir=out / name, seed=seed)
        row = {"record": {k: rec.to_dict()[k] for k in ("status", "reason", "duration_s", "peak_vram_mb", "model_version")}}
        if rec.status == "success":
            aligned, info = align_eval(load_mesh(rec.output_path), ref)
            local = Mesh(aligned.vertices - offset, aligned.faces) if offset is not None else aligned
            row.update(alignment=info, measured=compare_hand(crop_to_hand(local, hand), hand, p))
            pics[name] = local
        report["methods"][name] = row

    if hi3d_wsl is None:
        report["methods"]["closeup_hi3d"] = {"status": "skipped", "reason": "no Hi3D runner given"}
    elif report["methods"]["closeup"]["record"]["status"] == "success":
        raw = out / "closeup" / "raw_triposg.glb"
        rr = hi3d_wsl(out / "closeup.png", raw, out / "closeup_hi3d.glb")
        row = {"runner": {k: rr.get(k) for k in ("status", "reason", "duration_s", "peak_vram_mb", "returncode")}}
        if rr.get("status") == "success":
            aligned, info = align_eval(load_mesh(out / "closeup_hi3d.glb"), hand)
            row.update(alignment=info, measured=compare_hand(aligned, hand, p))
            pics["closeup_hi3d"] = aligned
        report["methods"]["closeup_hi3d"] = row

    Image.fromarray(hand_pictures(pics)).save(out / "hand_region.png")
    (out / "handzoom.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def rescore(run_dir, body_path, margins=(0.0, 0.01, 0.02, 0.05), local_margin: float = 0.01) -> dict:
    """Recompute a run's metrics from its raw outputs, two ways (both use GT, evaluation only):
    global = whole output aligned (bbox + similarity ICP), then the hand region measured: placement + shape;
    local  = the global result's hand crop (local_margin around the GT hand box) rigidly ICP-aligned onto the
             GT hand: shape only. The local number depends on the crop margin (more of the arm in the crop
             pulls the ICP), so it is also reported for every value in `margins`."""
    run_dir = Path(run_dir)
    p = humanoid.HandParams()
    hand = humanoid.hand_mesh(p)
    fig, off = figure_with_hand(hand, load_mesh(body_path))
    raws = {"whole_body": (run_dir / "whole_body" / "raw_triposg.glb", fig, off),
            "closeup": (run_dir / "closeup" / "raw_triposg.glb", hand, None),
            "closeup_hi3d": (run_dir / "closeup_hi3d.glb", hand, None)}
    out = {"run_dir": str(run_dir), "alignment": "evaluation only, uses GT", "methods": {}}
    for name, (f, ref, offset) in raws.items():
        if not f.is_file():
            out["methods"][name] = {"status": "missing"}
            continue
        aligned, _ = align_eval(load_mesh(f), ref)
        if offset is not None:
            aligned = Mesh(aligned.vertices - offset, aligned.faces)
        sens = {}
        for m in sorted(set(margins) | {local_margin}):
            local, info = similarity_icp(crop_to_hand(aligned, hand, margin=m), hand, scale=False)
            sens[m] = (compare_hand(local, hand, p), float(np.linalg.norm(info["translation"]) * 1000))
        out["methods"][name] = {"global": compare_hand(crop_to_hand(aligned, hand), hand, p),
                                "local": sens[local_margin][0], "local_margin_m": local_margin,
                                "local_icp_shift_mm": sens[local_margin][1],
                                "local_by_margin": {str(m): {"chamfer_fingers_mm": r["chamfer_fingers_mm"],
                                                             "icp_shift_mm": s} for m, (r, s) in sens.items()}}
    return out


def hi3d_via_wsl(resolution: int = 512, distro: str = "Ubuntu-24.04"):
    """Callable for run(): Hi3D refine_mesh through hi3d_runner.py in the WSL env of scripts/setup_hi3d_env.sh."""
    from .anatomy import wsl_path
    from .backends.base import invoke_runner

    runner = Path(__file__).parent / "backends" / "hi3d_runner.py"

    def call(image, coarse, output):
        result = Path(output).with_suffix(".json")
        q = lambda p: shlex.quote(wsl_path(p))  # noqa: E731 - Windows paths may contain spaces
        script = ("export CC=~/micromamba/envs/hi3d/bin/x86_64-conda-linux-gnu-gcc; "
                  f"~/micromamba/envs/hi3d/bin/python {q(runner)} --repo ~/ai3d/Hi3Dv1.0 "
                  f"--result {q(result)} -- --model ~/ai3d/weights/Hi3Dv1.0 --local-files-only "
                  "--dino-model ~/ai3d/weights/dinov2-large --clip-model ~/ai3d/weights/clip-vit-large-patch14 "
                  f"--image {q(image)} --coarse-mesh {q(coarse)} --output {q(output)} "
                  f"--resolution {int(resolution)} --seed 0 --sequential-cpu-offload")
        return invoke_runner(["wsl", "-d", distro, "--", "bash", "-lc", script], result, Path(output), timeout=3600)

    return call
