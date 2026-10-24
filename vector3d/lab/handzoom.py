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


def hi3d_via_wsl(resolution: int = 512, distro: str = "Ubuntu-24.04"):
    """Callable for run(): Hi3D refine_mesh through hi3d_runner.py in the WSL env of scripts/setup_hi3d_env.sh."""
    from .anatomy import wsl_path
    from .backends.base import invoke_runner

    runner = Path(__file__).parent / "backends" / "hi3d_runner.py"

    def call(image, coarse, output):
        result = Path(output).with_suffix(".json")
        script = ("export CC=~/micromamba/envs/hi3d/bin/x86_64-conda-linux-gnu-gcc; "
                  f"~/micromamba/envs/hi3d/bin/python {wsl_path(runner)} --repo ~/ai3d/Hi3Dv1.0 "
                  f"--result {wsl_path(result)} -- --model ~/ai3d/weights/Hi3Dv1.0 --local-files-only "
                  "--dino-model ~/ai3d/weights/dinov2-large --clip-model ~/ai3d/weights/clip-vit-large-patch14 "
                  f"--image {wsl_path(image)} --coarse-mesh {wsl_path(coarse)} --output {wsl_path(output)} "
                  f"--resolution {resolution} --seed 0 --sequential-cpu-offload")
        return invoke_runner(["wsl", "-d", distro, "--", "bash", "-lc", script], result, Path(output), timeout=3600)

    return call
