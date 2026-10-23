"""Distilling shape knowledge into a small student (level 1).

The imagination loop (imagine.py) keeps the silhouettes exact but cannot see
what no reference view shows: open cavities, shells, parts fused with the
body. A large model such as TripoSG has learned those from data. Level 1
distillation trains a small 3D network to predict a teacher's full shape
from what the engine can see:

  input  (7, G, G, G) on the reference camera cube:
         visual hull occupancy, imagination-loop field, and the luminance of
         the four reference views swept along their viewing axis
  target (G, G, G) occupancy of the teacher shape

Teachers: `procedural` objects (exact SDF ground truth, generated here, no
third-party assets) or TripoSG meshes collected on a GPU machine with
`collect_teacher` (label AI_GENERATED). The student output is always cut by
the visual hull, so it cannot contradict a silhouette (VERIFIED); what it adds
inside is ESTIMATED.

Training needs torch (CPU is enough at G=48..64); data generation does not.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..reconstruct.surface import Mesh, field_to_mesh, taubin_smooth
from . import dataset as ds
from .imagine import ImagineParams, hull_field, imagine
from .render import render, shade

CHANNELS = ("hull", "imagined", "lum_front", "lum_back", "lum_left", "lum_right", "bias")


# --- procedural teachers --------------------------------------------------
def random_sdf(rng: np.random.Generator):
    """A random object: 1-4 primitives, optional open cavity, shell or handle."""
    parts = []
    body_kind = rng.choice(["box", "cyl", "sphere", "capsule"])
    sx, sy, sz = rng.uniform(0.18, 0.45, 3)
    if body_kind == "box":
        r = rng.uniform(0, 0.6) * min(sx, sy, sz)
        parts.append(lambda p, h=(sx, sy, sz), r=r: ds.sd_box(p, h, r))
    elif body_kind == "cyl":
        parts.append(lambda p, r=sx, h=sy: ds.sd_cyl_y(p, r, h))
    elif body_kind == "sphere":
        parts.append(lambda p, r=sx: ds.sd_sphere(p, r))
    else:
        parts.append(lambda p, a=(0, -sy, 0), b=(0, sy, 0), r=sx * 0.7: ds.sd_capsule(p, a, b, r))
    for _ in range(rng.integers(0, 4)):                       # limbs, legs, knobs
        a = rng.uniform(-0.3, 0.3, 3)
        b = a + rng.normal(0, 0.25, 3)
        r = rng.uniform(0.02, 0.08)
        parts.append(lambda p, a=a, b=b, r=r: ds.sd_capsule(p, a, b, r))
    blend = rng.uniform(0, 0.05)
    cavity = rng.random() < 0.35 and body_kind in ("box", "cyl")
    shell = not cavity and rng.random() < 0.15
    handle = rng.random() < 0.25
    hx, hR, hr = (sx if body_kind != "box" else sx) + 0.02, rng.uniform(0.08, 0.16), rng.uniform(0.02, 0.04)

    def sdf(p):
        d = parts[0](p)
        if cavity:
            wall = rng_wall
            inner = (ds.sd_box(ds._t(p, y=wall), (sx - wall, sy, sz - wall)) if body_kind == "box"
                     else ds.sd_cyl_y(ds._t(p, y=wall), sx - wall, sy))
            d = np.maximum(d, -inner)
        if shell:
            d = np.maximum(d, -(d + rng_wall))
            d = np.maximum(d, p[..., 1] - sy * 0.6)          # cut open so the shell is visible
        for f in parts[1:]:
            d = ds.smin(d, f(p), blend) if blend > 1e-3 else np.minimum(d, f(p))
        if handle:
            h = ds.sd_torus_z(ds._t(p, x=hx), hR, hr)
            d = np.minimum(d, np.maximum(h, -(p[..., 0] - hx + 0.02)))
        return d

    rng_wall = float(rng.uniform(0.025, 0.06))
    return sdf


def mesh_occupancy(mesh: Mesh, center, half: float, G: int) -> np.ndarray:
    """(G,G,G) bool occupancy (index order x, y, z) of a closed mesh on the camera cube."""
    v, f = mesh.vertices, mesh.faces
    tri = v[f]
    h = 2 * half / G
    n = max(2, int(np.ceil(np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1).max() / (0.5 * h))) + 1)
    w = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
    pts = (tri[:, None, 0] * (1 - w[:, 0] - w[:, 1])[None, :, None] + tri[:, None, 1] * w[None, :, 0, None]
           + tri[:, None, 2] * w[None, :, 1, None]).reshape(-1, 3)
    idx = np.floor((pts - (np.asarray(center) - half)) / h).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < G), axis=1)]
    surf = np.zeros((G, G, G), bool)
    surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return ndimage.binary_fill_holes(surf)


# --- samples -----------------------------------------------------------------
def _swept_lum(images: dict, G: int) -> list[np.ndarray]:
    """Luminance of each reference view, block-averaged to G and swept along its viewing axis."""
    out = []
    for v in ds.REFERENCE_VIEWS:
        lum = images[v][..., :3].astype(np.float32).mean(-1) / 255 * (images[v][..., 3] > 127)
        S = lum.shape[0]
        small = lum.reshape(G, S // G, G, S // G).mean((1, 3))  # [row, col]
        grid_yx = small[::-1]                                     # row 0 is the top: y index G-1
        if v == "front":
            vol = np.broadcast_to(grid_yx.T[:, :, None], (G, G, G))                 # col -> x
        elif v == "back":
            vol = np.broadcast_to(grid_yx.T[::-1, :, None], (G, G, G))
        elif v == "left":                                                            # col -> -z
            vol = np.broadcast_to(grid_yx[:, ::-1][None, :, :], (G, G, G))
        else:                                                                         # col -> +z
            vol = np.broadcast_to(grid_yx[None, :, :], (G, G, G))
        out.append(np.ascontiguousarray(vol, dtype=np.float32))
    return out


SEARCH = ImagineParams(exponents=(1.4, 2.0, 3.0, 6.0, 16.0, 60.0))


def features(images: dict, masks: dict, cams: dict, G: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Student input (7, G, G, G) and the imagination-loop exponent per image row."""
    S = masks["front"].shape[0]
    res = imagine(images, masks, cams, SEARCH)
    hull_f, grid = hull_field(masks, cams, np.full(S, 1e3), G)
    im_f, _ = hull_field(masks, cams, res.exponent_per_row, G)
    h = float(grid.h[0])
    x = np.stack([(hull_f > 0).astype(np.float32), np.clip(im_f / (2 * h), -1, 1)]
                 + _swept_lum(images, G) + [np.ones((G, G, G), np.float32)])
    return x, res.exponent_per_row


def make_sample(teacher: Mesh, size: int = 128, G: int = 64) -> dict:
    """Render the teacher's four reference views and build (input, target) on the camera cube."""
    cams = ds.cameras_for(teacher, size)
    images, masks = {}, {}
    for v in ds.REFERENCE_VIEWS:
        r = render(teacher, cams[v])
        images[v] = shade(teacher, r)
        masks[v] = r.mask
    x, exps = features(images, masks, cams, G)
    c, half = np.array(cams["front"].center), cams["front"].half_size
    y = mesh_occupancy(teacher, c, half, G)
    return {"x": x.astype(np.float16), "y": np.packbits(y), "G": G, "size": size,
            "center": c, "half": half, "exponents": exps}


def generate_procedural(out_dir, count: int, seed: int = 0, G: int = 64, size: int = 128) -> dict:
    """Write `count` procedural teacher samples as npz (label: ground truth of a synthetic SDF)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done, t0 = 0, time.perf_counter()
    for i in range(count):
        path = out / f"proc_{seed}_{i:05d}.npz"
        if path.exists():
            done += 1
            continue
        rng = np.random.default_rng([seed, i])
        try:
            mesh = ds.mesh_sdf(random_sdf(rng), 96)
            s = make_sample(mesh, size, G)
        except ValueError:                       # empty object; skip, it is not data
            continue
        np.savez_compressed(path, teacher="procedural", **s)
        done += 1
    info = {"teacher": "procedural", "count": done, "seed": seed, "G": G, "size": size,
            "seconds": time.perf_counter() - t0}
    (out / f"info_proc_{seed}.json").write_text(json.dumps(info, indent=1))
    return info


def collect_teacher(mesh_paths, out_dir, G: int = 64, size: int = 128) -> dict:
    """Turn teacher meshes (e.g. TripoSG raw outputs) into training samples; label AI_GENERATED."""
    from .mesh_io import load_mesh

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in map(Path, mesh_paths):
        try:
            s = make_sample(load_mesh(p), size, G)
        except Exception as exc:  # noqa: BLE001 - one bad mesh is reported, not fatal
            rows.append({"mesh": str(p), "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        np.savez_compressed(out / f"teacher_{p.parent.name}_{p.stem}.npz", teacher="AI_GENERATED", **s)
        rows.append({"mesh": str(p), "status": "ok"})
    (out / "collect.json").write_text(json.dumps(rows, indent=1))
    return {"ok": sum(r["status"] == "ok" for r in rows), "rows": rows}


def load_sample(path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    G = int(d["G"])
    return d["x"].astype(np.float32), np.unpackbits(d["y"])[: G ** 3].reshape(G, G, G).astype(np.float32)


# --- student -------------------------------------------------------------------
@dataclass
class TrainParams:
    epochs: int = 30
    batch: int = 4
    lr: float = 2e-3
    width: int = 16
    seed: int = 0
    val_fraction: float = 0.1


def make_student(width: int = 16):
    """3D U-Net, two levels, about 0.3M parameters at width 16."""
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(),
                             nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(len(CHANNELS), w), block(w, 2 * w), block(2 * w, 4 * w)
            self.u2, self.d2 = nn.ConvTranspose3d(4 * w, 2 * w, 2, 2), block(4 * w, 2 * w)
            self.u1, self.d1 = nn.ConvTranspose3d(2 * w, w, 2, 2), block(2 * w, w)
            self.out = nn.Conv3d(w, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(nn.functional.max_pool3d(e1, 2))
            e3 = self.e3(nn.functional.max_pool3d(e2, 2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            # residual on the imagination field: the student only has to learn the correction
            return self.out(d1)[:, 0] + 4.0 * x[:, 1]

    return Student()


def train(sample_dir, out_path, params: TrainParams | None = None, log=print) -> dict:
    import torch

    p = params or TrainParams()
    torch.manual_seed(p.seed)
    files = sorted(Path(sample_dir).glob("*.npz"))
    if len(files) < 4:
        raise ValueError(f"need at least 4 samples in {sample_dir}, found {len(files)}")
    rng = np.random.default_rng(p.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * p.val_fraction))
    val, tr = files[:n_val], files[n_val:]
    data = {f: load_sample(f) for f in files}
    net = make_student(p.width)
    opt = torch.optim.AdamW(net.parameters(), lr=p.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, p.lr, total_steps=p.epochs * ((len(tr) + p.batch - 1) // p.batch))
    history = []

    def batch_of(fs):
        x = torch.from_numpy(np.stack([data[f][0] for f in fs]))
        y = torch.from_numpy(np.stack([data[f][1] for f in fs]))
        return x, y

    def loss_fn(logit, x, y):
        inside = x[:, 0] > 0.5                    # outside the hull is VERIFIED empty: not learned
        return torch.nn.functional.binary_cross_entropy_with_logits(logit[inside], y[inside])

    t0 = time.perf_counter()
    for ep in range(p.epochs):
        net.train()
        order = rng.permutation(len(tr))
        tl = []
        for i in range(0, len(tr), p.batch):
            x, y = batch_of([tr[j] for j in order[i:i + p.batch]])
            loss = loss_fn(net(x), x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl.append(float(loss))
        net.eval()
        with torch.no_grad():
            vi = []
            for f in val:
                x, y = batch_of([f])
                pred = (net(x) > 0) & (x[:, 0] > 0.5)
                base = x[:, 1] > 0                # imagination loop alone
                yb = y > 0.5
                vi.append((float((pred & yb).sum() / (pred | yb).sum()), float((base & yb).sum() / (base | yb).sum())))
        row = {"epoch": ep, "train_loss": float(np.mean(tl)), "val_iou_student": float(np.mean([a for a, _ in vi])),
               "val_iou_imagination": float(np.mean([b for _, b in vi])), "seconds": time.perf_counter() - t0}
        history.append(row)
        log(json.dumps(row))
    torch.save({"state": net.state_dict(), "width": p.width, "channels": CHANNELS, "G": int(data[files[0]][1].shape[0]),
                "params": asdict(p), "train_files": len(tr), "val_files": len(val), "history": history}, out_path)
    return {"history": history, "train": len(tr), "val": len(val)}


def load_student(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = make_student(ck["width"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def predict_mesh(net, images: dict, masks: dict, cams: dict, G: int = 64, size: int = 128,
                 smooth: int = 6) -> tuple[Mesh, dict]:
    """Student shape at the image resolution: student probability upsampled, cut by the exact hull."""
    from dataclasses import replace

    import torch

    S = masks["front"].shape[0]
    small_m = {v: np.asarray(Image.fromarray(masks[v].astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)) > 127
               for v in masks}
    small_i = {v: np.asarray(Image.fromarray(images[v]).resize((size, size), Image.BILINEAR)) for v in images}
    x, _ = features(small_i, small_m, {k: replace(c, size=size) for k, c in cams.items()}, G)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(x[None]))).numpy()[0]
    hull_hi, grid = hull_field(masks, cams, np.full(S, 1e3), S)
    up = ndimage.zoom(prob, S / G, order=1)
    field = np.minimum(hull_hi, (up - 0.5) * float(grid.h[0]) * 8)
    mesh = field_to_mesh(field.astype(np.float32), grid)
    if smooth:
        mesh = taubin_smooth(mesh, smooth)
    return mesh, {"label": "ESTIMATED", "method": "distilled student over imagination loop v0",
                  "verified": "cut by the visual hull of the four reference silhouettes"}
