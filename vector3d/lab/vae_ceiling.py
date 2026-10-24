"""VAE reconstruction probe: how much of a synthetic hand TripoSG's VAE reproduces.

Two scenes built from the procedural hand (humanoid.py), sampled as surface
points + normals, normalised to TripoSG model space, encoded and decoded by
`triposg_vae_roundtrip.py` in the TripoSG env, then measured back in metres:
  closeup  the hand fills the model cube
  in_body  the hand at real scale beside a 1.75 m figure
This measures reconstruction error of one synthetic probe with the encoder's
posterior mean and one extraction depth. It is evidence for a hypothesis
(capacity limited by the hand's scale in the cube), not a proof of a ceiling
for every latent a DiT could produce: sampled latents, real scans and a
fine-tuned decoder are untested.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
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


def reference_params(p: humanoid.HandParams) -> dict:
    """Generator settings of the ground-truth hand. Inputs, not measurements: recovered crease depth,
    ridge amplitude and finger gap are NOT measured by compare_hand."""
    return {"crease_depth_mm": p.crease * 1000, "ridge_amplitude_mm": p.ridge * 1000, "finger_gap_mm": p.gap * 1000}


def compare_hand(pred: Mesh, gt: Mesh, p: humanoid.HandParams = humanoid.HandParams(), n: int = 60000) -> dict:
    """Measured metrics in hand coordinates (metres) on the region the hand occupies.
    `fingers` is a one-slice heuristic (humanoid.count_fingers), not a topological count."""
    local = crop_to_hand(pred, gt)
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
    }


def hand_pictures(meshes: dict[str, Mesh], size: int = 384) -> np.ndarray:
    """Front (palm side) and 3/4 views of each mesh, framed on the hand."""
    tiles, labels = [], []
    centre = HAND_VIEW_CENTRE
    for name, m in meshes.items():
        for cam_name, cam in (("palm", orbit_camera(0, 0, centre, 0.11, size)),
                              ("az30", orbit_camera(30, 15, centre, 0.11, size))):
            tiles.append(shade(m, render(m, cam)))
            labels.append(f"{name} {cam_name}")
    return contact_sheet(tiles, labels)


def invoke(cmd: list[str], result: Path, mesh: Path, timeout: int = 3600) -> dict:
    """Run a child runner and accept its result only if it is fresh: exit code 0, a result JSON and a
    mesh both written after the start, and status success. Anything else is a failure record."""
    for f in (result, mesh):
        if f.exists():
            raise FileExistsError(f"{f} already exists; every invocation needs fresh output paths")
    t0 = time.time()
    try:
        code = subprocess.run(cmd, check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": f"timed out after {timeout}s"}
    fresh = result.is_file() and result.stat().st_mtime >= t0 - 1
    rr = json.loads(result.read_text()) if fresh else {}
    rr = {k: v for k, v in rr.items() if k != "traceback"}
    if code != 0 or not fresh or rr.get("status") != "success" or not mesh.is_file():
        why = rr.get("reason") or ("no result file written" if not fresh else "mesh missing")
        return {**rr, "status": "failed", "returncode": code, "reason": f"exit {code}: {why}"}
    return {**rr, "returncode": code}


def run(out_dir, python: str, repo: str, weights: str, body_path, half: float = 0.9913,
        n_points: int = 200000, tokens=(2048,), octree_depth: int = 9) -> dict:
    """One invocation = one new folder out_dir/run_<time>_<id>; earlier runs are never read or overwritten."""
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True)
    p = humanoid.HandParams()
    hand = humanoid.hand_mesh(p)
    fig, off = figure_with_hand(hand, load_mesh(body_path))
    scenes = {"closeup": (hand, np.zeros(3)), "in_body": (fig, off)}
    report = {"run_id": run_id, "reference_params": reference_params(p), "hand_params": p.to_dict(),
              "gt_half": half, "n_points": n_points, "octree_depth": octree_depth, "runs": {}}
    pictures = {"input": hand}
    for name, (mesh, offset) in scenes.items():
        pts, nrm = surface_points(mesh, n_points)
        mp, c, s = to_model_space(pts, half)
        np.savez(out_dir / f"{name}_surface.npz", points=mp, normals=nrm.astype(np.float32))
        for k in tokens:
            tag = f"{name}_t{k}"
            cmd = [python, str(RUNNER), "--repo", repo, "--weights", weights, "--surface",
                   str(out_dir / f"{name}_surface.npz"), "--out", str(out_dir / f"{tag}.glb"),
                   "--result", str(out_dir / f"{tag}_runner.json"), "--tokens", str(k),
                   "--octree-depth", str(octree_depth)]
            rr = invoke(cmd, out_dir / f"{tag}_runner.json", out_dir / f"{tag}.glb")
            row = {"runner": rr}
            if rr["status"] == "success":
                dec = load_mesh(out_dir / f"{tag}.glb")
                back = Mesh(dec.vertices / s + c - offset, dec.faces)
                row["measured"] = compare_hand(back, hand, p)
                if k == tokens[0]:
                    pictures[name] = back
            report["runs"][tag] = row
    Image.fromarray(hand_pictures(pictures)).save(out_dir / "hand_region.png")
    report["noise_floor"] = compare_hand(hand, hand, p)
    (out_dir / "vae_ceiling.json").write_text(json.dumps(report, indent=1, default=str))
    return report


# --- Hi3D refinement of the in-body hand: inputs and evaluation -------------
HAND_VIEW_CENTRE = (-0.02, 0.09, 0.0)


def crop_to_hand(mesh: Mesh, hand: Mesh, margin: float = 0.01) -> Mesh:
    """Faces whose three vertices lie in the hand's bounding box (plus margin)."""
    lo, hi = hand.vertices.min(0) - margin, hand.vertices.max(0) + margin
    keep = np.all((mesh.vertices >= lo) & (mesh.vertices <= hi), axis=1)
    fk = keep[mesh.faces].all(1)
    remap = -np.ones(len(keep), int)
    remap[keep] = np.arange(keep.sum())
    return Mesh(mesh.vertices[keep], remap[mesh.faces[fk]])


def bbox_fit(pred: Mesh, ref: Mesh) -> tuple[Mesh, dict]:
    """Uniform scale + translation mapping pred's bbox onto ref's (largest side, centres). Approximate:
    Hi3D renormalises its output, and this is not a surface registration."""
    pl, ph = pred.vertices.min(0), pred.vertices.max(0)
    rl, rh = ref.vertices.min(0), ref.vertices.max(0)
    s = float((rh - rl).max() / (ph - pl).max())
    t = (rl + rh) / 2 - (pl + ph) / 2 * s
    return Mesh(pred.vertices * s + t, pred.faces), {"scale": s, "translation": t.tolist(),
                                                      "method": "bbox fit (approximate)"}


def prepare_hi3d_hand(vae_run_dir, body_path, out_dir, half: float = 0.9913, n_points: int = 200000) -> dict:
    """From a VAE probe run: the in-body decoded hand cropped to hand coordinates (Hi3D coarse mesh),
    and an RGBA palm-view render of the reference hand (Hi3D condition image)."""
    import trimesh

    vae_run_dir, out_dir = Path(vae_run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hand = humanoid.hand_mesh()
    fig, off = figure_with_hand(hand, load_mesh(body_path))
    pts, _ = surface_points(fig, n_points)
    _, c, s = to_model_space(pts, half)
    dec = load_mesh(vae_run_dir / "in_body_t2048.glb")
    coarse = crop_to_hand(Mesh(dec.vertices / s + c - off, dec.faces), hand)
    trimesh.Trimesh(coarse.vertices, coarse.faces, process=False).export(out_dir / "hand_coarse.glb")
    r = render(hand, orbit_camera(0, 0, HAND_VIEW_CENTRE, 0.11, 1024))
    rgb = shade(hand, r)[..., :3]
    Image.fromarray(np.dstack([rgb, (r.face_id >= 0).astype(np.uint8) * 255])).save(out_dir / "hand_image.png")
    return {"coarse": str(out_dir / "hand_coarse.glb"), "image": str(out_dir / "hand_image.png"),
            "coarse_faces": int(len(coarse.faces))}


def evaluate_hi3d_hand(hi3d_dir, resolutions=(512, 1024)) -> dict:
    """Measured metrics of the coarse hand and each Hi3D output (bbox-fitted onto the coarse mesh)."""
    hi3d_dir = Path(hi3d_dir)
    p = humanoid.HandParams()
    hand = humanoid.hand_mesh(p)
    coarse = load_mesh(hi3d_dir / "hand_coarse.glb")
    out = {"reference_params": reference_params(p),
           "alignment_note": "Hi3D outputs bbox-fitted onto the coarse mesh; Chamfer is approximate",
           "finger_count_note": "one-slice heuristic (humanoid.count_fingers)",
           "coarse_in_body_vae": {"measured": compare_hand(coarse, hand, p)}}
    pics = {"input": hand, "coarse": coarse}
    for res in resolutions:
        f = hi3d_dir / f"hand_{res}_offload.glb"
        if not f.is_file():
            out[f"hi3d_{res}"] = {"status": "missing", "path": str(f)}
            continue
        fitted, info = bbox_fit(load_mesh(f), coarse)
        out[f"hi3d_{res}"] = {"alignment": info, "measured": compare_hand(fitted, hand, p)}
        pics[f"hi3d{res}"] = fitted
    Image.fromarray(hand_pictures(pics)).save(hi3d_dir / "hand_region.png")
    (hi3d_dir / "hand_metrics.json").write_text(json.dumps(out, indent=1))
    return out
