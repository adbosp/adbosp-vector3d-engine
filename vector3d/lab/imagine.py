"""Imagination loop: imagine -> simulate -> compare -> revise (prototype v0).

Idea (brain-inspired, see research/imagination_engine.md): perception gives
hard facts (the silhouettes), imagination proposes what the views cannot
show (the shape of each horizontal cross-section), mental simulation renders
every proposal, and the prediction error against the observed shading
decides which proposal survives. A held-out check (metacognition) keeps the
revision only when it does not contradict evidence it was not fitted to.

Key fact used: with front/back and left/right silhouettes, every horizontal
slice of the visual hull is a set of rectangles (x-interval by z-interval).
Any convex cross-section that touches all four sides of its rectangle leaves
both silhouettes unchanged. Superellipses |x'|^n + |z'|^n <= 1 do exactly
that for every n, from a diamond (n=1) through a circle (n=2) to the box
(n -> inf). So the choice of n per slice changes the unseen shape while the
silhouettes stay VERIFIED; n is ESTIMATED from shading.

Shading model: lum = alpha + beta*|n.t| + gamma*max(n.L, 0), with alpha,
beta, gamma fitted by least squares and L picked from a fixed direction set,
per view and per hypothesis. Lighting is not given to the method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

from ..reconstruct.hull import Grid
from ..reconstruct.surface import Mesh, field_to_mesh, taubin_smooth
from .render import OrthoCamera, render

EXPONENTS = (1.4, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 16.0, 25.0, 60.0)
VIEWS = ("front", "back", "left", "right")


@dataclass
class ImagineParams:
    exponents: tuple = EXPONENTS
    resolution: int | None = None        # voxels per side while searching (default: half the image size)
    final_resolution: int | None = None  # default: the image size, so the hull is pixel exact
    smooth_cost: float = 0.02      # DP penalty per step in log(n)
    search_sigma: float = 0.8      # blur of the search field, for clean normals
    prior_cost: float = 0.01       # per row, x |log(n/2)|: with no evidence, imagine round sections
    by_part: bool = True           # one decision per part instead of per row
    final_smooth: int = 6          # Taubin passes on the result: removes 1-px slice terraces
    rounds: int = 2                # light fit <-> slice choice alternations


@dataclass
class ImagineResult:
    mesh: Mesh
    exponent_per_row: np.ndarray
    history: list = field(default_factory=list)
    parts: list = field(default_factory=list)


def _intervals(row: np.ndarray):
    """Per pixel: (start, end) of the run of True it belongs to, -1 if outside."""
    idx = np.arange(len(row))
    start = np.where(row, idx, -1)
    if not row.any():
        return start, start
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    lo, hi = np.full(len(row), -1), np.full(len(row), -1)
    for a, b in zip(s, e):
        lo[a:b + 1], hi[a:b + 1] = a, b
    return lo, hi


def _world_coord(cam: OrthoCamera, pixel: np.ndarray, axis: int) -> np.ndarray:
    """World coordinate along `axis` of pixel column centres (pixel may be fractional)."""
    s = cam.size / (2 * cam.half_size)
    u = pixel / s - cam.half_size
    return cam.center[axis] + u * cam.right[axis]


def hull_field(masks: dict, cams: dict, n_per_row: np.ndarray, res: int, thin_rows=None):
    """Signed field (positive inside) of the superellipse-sliced hull, and its grid.

    thin_rows[r]: in a row with several x-runs but one z-run, the narrower runs
    become round tubes (depth = width) instead of spanning the full depth. The
    widest run still spans the z-run, so the side silhouette is unchanged.
    """
    front, side = masks["front"], masks["left"]
    cf, cs = cams["front"], cams["left"]
    S = cf.size
    c, hs = np.array(cf.center), cf.half_size
    h = 2 * hs / res
    lo = c - hs
    coords = [lo[a] + (np.arange(res) + 0.5) * h for a in range(3)]
    # voxel -> pixel column / row (all reference cameras share centre, scale and up = +Y)
    px = (coords[0] - c[0]) * cf.right[0] * S / (2 * hs) + S / 2
    pz = (coords[2] - c[2]) * cs.right[2] * S / (2 * hs) + S / 2
    py = (hs - (coords[1] - c[1])) * S / (2 * hs)
    ix, iz, iy = (np.clip(p.astype(int), 0, S - 1) for p in (px, pz, py))
    out = np.full((res, res, res), -1.0, np.float32)
    for j, r in enumerate(iy):
        flo, fhi = _intervals(front[r])
        slo, shi = _intervals(side[r])
        fx, sz = flo[ix] >= 0, slo[iz] >= 0
        if not fx.any() or not sz.any():
            continue
        x0, x1 = _world_coord(cf, flo[ix] - 0.0, 0), _world_coord(cf, fhi[ix] + 1.0, 0)
        z0, z1 = _world_coord(cs, slo[iz] - 0.0, 2), _world_coord(cs, shi[iz] + 1.0, 2)
        xc, a = (x0 + x1) / 2, np.abs(x1 - x0) / 2
        zc, b = (z0 + z1) / 2, np.abs(z1 - z0) / 2
        bb = np.broadcast_to(b[None, :], (res, res))
        runs = np.unique(flo[flo >= 0])
        if thin_rows is not None and thin_rows[r] and len(runs) > 1 and len(np.unique(slo[slo >= 0])) == 1:
            widths = {s: fhi[flo == s][0] - s for s in runs}
            main = max(widths, key=widths.get)
            minor = fx & (flo[ix] != main)
            bb = np.where(minor[:, None], np.minimum(bb, a[:, None]), bb)
        xr = np.abs(coords[0] - xc) / np.maximum(a, 1e-9)
        zr = np.abs(coords[2] - zc)[None, :] / np.maximum(bb, 1e-9)
        n = float(n_per_row[r])
        xr, zr = np.minimum(xr, 2.0), np.minimum(zr, 2.0)    # outside the cell anyway; avoids overflow
        rad = (xr[:, None] ** n + zr ** n) ** (1 / n)
        val = (1 - rad) * np.minimum(a[:, None], bb)
        val[~fx, :] = -h
        val[:, ~sz] = -h
        out[:, j, :] = np.maximum(val, -h)
    return out, Grid(lo, np.full(3, h), (res,) * 3)


def build_mesh(masks, cams, n_per_row, res, thin_rows=None, sigma=0.0) -> Mesh:
    f, grid = hull_field(masks, cams, n_per_row, res, thin_rows)
    if sigma:
        f = ndimage.gaussian_filter(f, sigma)
    return field_to_mesh(f, grid)


def _light_dirs(k: int = 40) -> np.ndarray:
    """Directions on the camera-facing hemisphere (camera frame: x right, y up, z toward)."""
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - i / k)           # 0..90 deg from the view axis
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)])


def pixel_normals(mesh: Mesh, cam: OrthoCamera):
    """Per-pixel unit normals in the camera frame, and the coverage mask."""
    r = render(mesh, cam)
    v = mesh.vertices[mesh.faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    basis = np.array([cam.right, cam.up, cam.toward])
    n = fn[np.maximum(r.face_id, 0)] @ basis.T
    n[n[..., 2] < 0] *= -1                      # two-sided, like the observed renders
    return n, r.mask


def fit_light(normals, lum, mask, dirs):
    """Best (L, coeffs) for lum ~ a + b*n_z + c*max(n.L,0); returns residual image too."""
    nz = normals[mask][:, 2]
    y = lum[mask]
    best = None
    for L in dirs:
        A = np.column_stack([np.ones_like(nz), nz, np.clip(normals[mask] @ L, 0, None)])
        coef, _ = nnls(A, y)                      # light adds brightness: no negative terms
        err = float(np.mean((A @ coef - y) ** 2))
        if best is None or err < best[0]:
            best = (err, L, coef)
    return best[1], best[2]


def predict(normals, L, coef):
    return coef[0] + coef[1] * normals[..., 2] + coef[2] * np.clip(normals @ L, 0, None)


def row_errors(mesh, cams, images, masks, views=VIEWS, lights=None):
    """(rows,) shading error summed over views, plus the light per view."""
    dirs = _light_dirs()
    total, fitted = None, {}
    for v in views:
        nrm, cover = pixel_normals(mesh, cams[v])
        lum = images[v][..., :3].astype(np.float32).mean(-1) / 255
        m = cover & masks[v]
        L, coef = lights[v] if lights else fit_light(nrm, lum, m, dirs)
        fitted[v] = (L, coef)
        e = np.where(m, (predict(nrm, L, coef) - lum) ** 2, 0.0)
        e = e + np.where(cover ^ masks[v], 0.05, 0.0)    # silhouette must stay put
        total = e.sum(1) if total is None else total + e.sum(1)
    return total, fitted


def choose_rows(cost: np.ndarray, exps, smooth: float) -> np.ndarray:
    """DP over image rows: per-row exponent index minimising cost + smooth*|dlog n|."""
    R, K = cost.shape
    ln = np.log(np.asarray(exps))
    jump = smooth * np.abs(ln[:, None] - ln[None, :]) * cost.max(initial=1e-9)
    acc, back = cost[0].copy(), np.zeros((R, K), int)
    for r in range(1, R):
        t = acc[:, None] + jump
        back[r] = t.argmin(0)
        acc = t.min(0) + cost[r]
    k = np.empty(R, int)
    k[-1] = int(acc.argmin())
    for r in range(R - 1, 0, -1):
        k[r - 1] = back[r, k[r]]
    return k


def row_parts(masks: dict, jump: float = 0.15) -> np.ndarray:
    """Label image rows by part: a new part starts where the slice structure changes.

    Structure = number of runs in the front and side masks; a width change of
    more than `jump` (relative) between rows also starts a new part. Rows
    without foreground get -1. Parts are judged as a whole (recognition by
    components), so a few rows with weak evidence cannot flip on their own.
    """
    front, side = masks["front"], masks["left"]
    labels = np.full(front.shape[0], -1)
    prev, part = None, -1
    for r in range(front.shape[0]):
        if not front[r].any() or not side[r].any():
            prev = None
            continue
        sig = (int(np.count_nonzero(np.diff(front[r].astype(np.int8)) == 1) + front[r][0]),
               int(np.count_nonzero(np.diff(side[r].astype(np.int8)) == 1) + side[r][0]),
               int(front[r].sum()), int(side[r].sum()))
        if prev is None or sig[:2] != prev[:2] or abs(sig[2] - prev[2]) > jump * prev[2] + 1 \
                or abs(sig[3] - prev[3]) > jump * prev[3] + 1:
            part += 1
        labels[r], prev = part, sig
    return labels


def imagine(images: dict, masks: dict, cams: dict, params: ImagineParams | None = None) -> ImagineResult:
    """Run the loop on the four reference views (RGBA images, masks, cameras)."""
    p = params or ImagineParams()
    S = masks["front"].shape[0]
    search_res = p.resolution or S // 2
    exps = np.asarray(p.exponents, float)
    cands = [(n, False) for n in exps] + [(n, True) for n in exps]
    cand_n = np.array([c[0] for c in cands])
    cand_thin = np.array([c[1] for c in cands])
    current, thin = np.full(S, exps[-1]), np.zeros(S, bool)     # start from the hull (box slices)
    history = []
    meshes = [build_mesh(masks, cams, np.full(S, n), search_res, np.full(S, th), p.search_sigma)
              for n, th in cands]
    # one light per view for the whole scene: from the single hypothesis that explains the images best
    own = [row_errors(m, cams, images, masks) for m in meshes]
    lights = own[int(np.argmin([c.sum() for c, _ in own]))][1]
    for rnd in range(p.rounds):
        if rnd:
            _, lights = row_errors(build_mesh(masks, cams, current, search_res, thin, p.search_sigma),
                                   cams, images, masks)
        cost = np.stack([row_errors(m, cams, images, masks, lights=lights)[0] for m in meshes], 1)
        prior = p.prior_cost * (np.abs(np.log(cand_n / 2.0)) + 0.1 * cand_thin)[None, :]
        total = cost + prior
        if p.by_part:
            parts = row_parts(masks)
            pick = np.full(S, int(np.argmin(np.abs(cand_n - 2.0) + cand_thin)))
            for q in np.unique(parts[parts >= 0]):
                rows_q = parts == q
                pick[rows_q] = int(total[rows_q].sum(0).argmin())
        else:
            pick = choose_rows(total, cand_n, p.smooth_cost)
        current, thin = cand_n[pick], cand_thin[pick]
        rows = masks["front"].any(1)
        history.append({"round": rnd, "exponents_used": sorted({float(x) for x in current[rows]}),
                        "median_exponent": float(np.median(current[rows])) if rows.any() else None,
                        "thin_rows": int(thin[rows].sum()),
                        "cost": float(cost[np.arange(S), pick].sum())})
    mesh = build_mesh(masks, cams, current, p.final_resolution or S, thin)
    if p.final_smooth:
        mesh = taubin_smooth(mesh, p.final_smooth)
    parts = row_parts(masks)
    summary = [{"part": int(q), "rows": [int(np.flatnonzero(parts == q)[0]), int(np.flatnonzero(parts == q)[-1])],
                "exponent": float(current[parts == q][0]), "thin_minor_runs": bool(thin[parts == q][0]),
                "label": "ESTIMATED"} for q in np.unique(parts[parts >= 0])]
    return ImagineResult(mesh, current, history, summary)


def run_benchmark(data_dir, out_dir, params: ImagineParams | None = None, names=None) -> dict:
    """Visual hull vs the imagination loop on a lab dataset; GT is used only for scoring."""
    import json
    import time
    from pathlib import Path

    from PIL import Image

    from ..export.writers import to_glb
    from . import dataset
    from .experiments import evaluate
    from .mesh_io import load_mesh
    from .render import contact_sheet, shade

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows = []
    for obj in manifest["objects"]:
        if names and obj["name"] not in names:
            continue
        od = data_dir / obj["name"]
        gt, cams = load_mesh(od / "gt.glb"), dataset.load_cameras(od)
        ref = dataset.load_masks(od, dataset.REFERENCE_VIEWS)
        held = dataset.load_masks(od, [h[0] for h in dataset.HELDOUT_VIEWS])
        imgs = {v: np.asarray(Image.open(od / "views" / f"{v}.png")) for v in VIEWS}
        S = ref["front"].shape[0]
        t0 = time.perf_counter()
        hull = build_mesh(ref, cams, np.full(S, 1e3), S)
        t_hull = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = imagine(imgs, ref, cams, params)
        t_im = time.perf_counter() - t0
        od_out = out_dir / obj["name"]
        od_out.mkdir(exist_ok=True)
        meta = {"label": "ESTIMATED", "method": "imagination loop v0", "parts": res.parts}
        (od_out / "imagined.glb").write_bytes(to_glb(res.mesh, obj["name"], meta))
        (od_out / "hull.glb").write_bytes(to_glb(hull, obj["name"], {"label": "VERIFIED", "method": "visual hull"}))
        tiles, labels = [], []
        for v in ("az45_el20", "top"):
            for tag, m in (("gt", gt), ("hull", hull), ("imagined", res.mesh)):
                tiles.append(shade(m, render(m, cams[v])))
                labels.append(f"{tag} {v}")
        Image.fromarray(contact_sheet(tiles, labels)).save(od_out / "compare.png")
        rows.append({"object": obj["name"], "category": obj["category"],
                     "hull": {**evaluate(hull, gt, cams, ref, held), "seconds": t_hull},
                     "imagined": {**evaluate(res.mesh, gt, cams, ref, held), "seconds": t_im},
                     "parts": res.parts, "history": res.history})
    summary = {"benchmark": "imagination loop v0 on " + manifest.get("version", "dataset"),
               "inputs": "four reference views (front/back/left/right): RGBA shading + silhouettes; light unknown",
               "params": asdict(params or ImagineParams()), "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary
