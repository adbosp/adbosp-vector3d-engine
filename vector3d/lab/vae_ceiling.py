"""VAE ceiling gate: how much hand detail survives TripoSG's VAE alone.

Two scenes built from the procedural hand (humanoid.py), sampled as surface
points + normals, normalised to TripoSG model space, encoded and decoded by
`triposg_vae_roundtrip.py` in the TripoSG env, then measured back in metres:
  closeup  the hand fills the model cube (best case for detail)
  in_body  the hand at real scale beside a 1.75 m figure (what a full-body
           image gives the model)
If fingers fuse or creases vanish here, no DiT fine-tune can bring them back.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ..reconstruct.surface import Mesh
from . import humanoid
from .mesh_io import load_mesh
from .metrics import sample_surface
from .render import contact_sheet, orbit_camera, render, shade

RUNNER = Path(__file__).parent / "backends" / "triposg_vae_roundtrip.py"
FIGURE_HEIGHT = 1.75


def surface_points(mesh: Mesh, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform points with the normal of the face they lie on."""
    rng = np.random.default_rng(seed)
    v, f = mesh.vertices, mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    i = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    p = v[f[i, 0]] * (1 - s)[:, None] + v[f[i, 1]] * (s * (1 - r2))[:, None] + v[f[i, 2]] * (s * r2)[:, None]
    return p, fn[i] / area[i, None]


def figure_with_hand(hand: Mesh, body: Mesh) -> tuple[Mesh, np.ndarray]:
    """Body scaled to FIGURE_HEIGHT with the hand beside it at hip height; returns mesh and hand offset."""
    lo, hi = body.vertices.min(0), body.vertices.max(0)
    bv = (body.vertices - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * (FIGURE_HEIGHT / (hi[1] - lo[1]))
    off = np.array([bv[:, 0].max() + 0.12, 0.80, 0.0])
    v = np.concatenate([bv, hand.vertices + off])
    f = np.concatenate([body.faces, hand.faces + len(bv)])
    return Mesh(v, f), off


def to_model_space(points: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre the bbox and scale its largest half extent to `half`; returns points, centre, scale."""
    lo, hi = points.min(0), points.max(0)
    c, s = (lo + hi) / 2, half / float((hi - lo).max() / 2)
    return ((points - c) * s).astype(np.float32), c, s


def compare_hand(pred: Mesh, gt: Mesh, p: humanoid.HandParams = humanoid.HandParams(), n: int = 60000) -> dict:
    """Metrics in hand coordinates (metres) on the region the hand occupies."""
    lo, hi = gt.vertices.min(0) - 0.01, gt.vertices.max(0) + 0.01
    keep = np.all((pred.vertices >= lo) & (pred.vertices <= hi), axis=1)
    fk = keep[pred.faces].all(1)
    remap = -np.ones(len(keep), int)
    remap[keep] = np.arange(keep.sum())
    local = Mesh(pred.vertices[keep], remap[pred.faces[fk]])
    if len(local.faces) == 0:
        return {"hand_faces": 0, "fingers": 0}
    gp, gn = surface_points(gt, n, 1)
    pp, pn = surface_points(local, n, 2)
    d_gp, i_gp = cKDTree(pp).query(gp)
    d_pg, i_pg = cKDTree(gp).query(pp)
    fin_g, fin_p = humanoid.finger_region(gp, p), humanoid.finger_region(pp, p)
    palm_g = (gp[:, 2] > 0.005) & (gp[:, 1] > 0.01) & (gp[:, 1] < humanoid.PALM_TOP - 0.01) & (np.abs(gp[:, 0]) < 0.03)
    return {
        "hand_faces": int(len(local.faces)),
        "chamfer_mm": float((d_gp.mean() + d_pg.mean()) / 2 * 1000),
        "chamfer_fingers_mm": float((d_gp[fin_g].mean() + d_pg[fin_p].mean()) / 2 * 1000),
        "chamfer_palm_ridges_mm": float(d_gp[palm_g].mean() * 1000),
        "normal_agreement": float(np.abs((gn * pn[i_gp]).sum(1)).mean()),
        "normal_agreement_fingers": float(np.abs((gn[fin_g] * pn[i_gp[fin_g]]).sum(1)).mean()),
        "fingers": humanoid.count_fingers(local, p),
        "crease_depth_mm": p.crease * 1000, "ridge_amplitude_mm": p.ridge * 1000, "finger_gap_mm": p.gap * 1000,
    }


def hand_pictures(meshes: dict[str, Mesh], size: int = 384) -> np.ndarray:
    """Front (palm side) and 3/4 views of each mesh, framed on the hand."""
    tiles, labels = [], []
    centre = (-0.02, 0.09, 0.0)
    for name, m in meshes.items():
        for cam_name, cam in (("palm", orbit_camera(0, 0, centre, 0.11, size)),
                              ("az30", orbit_camera(30, 15, centre, 0.11, size))):
            tiles.append(shade(m, render(m, cam)))
            labels.append(f"{name} {cam_name}")
    return contact_sheet(tiles, labels)


def run(out_dir, python: str, repo: str, weights: str, body_path, half: float = 0.9913,
        n_points: int = 200000, tokens=(2048,)) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = humanoid.HandParams()
    hand = humanoid.hand_mesh(p)
    fig, off = figure_with_hand(hand, load_mesh(body_path))
    scenes = {"closeup": (hand, np.zeros(3)), "in_body": (fig, off)}
    report = {"hand_params": p.to_dict(), "gt_half": half, "n_points": n_points, "runs": {}}
    pictures = {"input": hand}
    for name, (mesh, offset) in scenes.items():
        pts, nrm = surface_points(mesh, n_points)
        mp, c, s = to_model_space(pts, half)
        np.savez(out_dir / f"{name}_surface.npz", points=mp, normals=nrm.astype(np.float32))
        for k in tokens:
            tag = f"{name}_t{k}"
            cmd = [python, str(RUNNER), "--repo", repo, "--weights", weights, "--surface",
                   str(out_dir / f"{name}_surface.npz"), "--out", str(out_dir / f"{tag}.glb"),
                   "--result", str(out_dir / f"{tag}_runner.json"), "--tokens", str(k)]
            subprocess.run(cmd, check=False)
            rr = json.loads((out_dir / f"{tag}_runner.json").read_text())
            row = {"runner": {k2: v for k2, v in rr.items() if k2 != "traceback"}}
            if rr.get("status") == "success":
                dec = load_mesh(out_dir / f"{tag}.glb")
                back = Mesh(dec.vertices / s + c - offset, dec.faces)
                row.update(compare_hand(back, hand, p))
                if k == tokens[0]:
                    pictures[name] = back
            else:
                row["reason"] = rr.get("reason")
            report["runs"][tag] = row
    Image.fromarray(hand_pictures(pictures)).save(out_dir / "hand_region.png")
    report["reference"] = {"input_hand": compare_hand(hand, hand, p)}
    (out_dir / "vae_ceiling.json").write_text(json.dumps(report, indent=1, default=str))
    return report
