"""Side-by-side comparison of image -> 3D methods on inputs without ground truth (e.g. turnaround sheets).

Measured: silhouette IoU of a front-only result against held-out reference views (right side, back). Both
silhouettes are cropped to their bounding box, scaled to the same height and aligned at the bottom centre,
so only shape is compared (the reference renders are perspective, ours orthographic: approximate).
Pictures: one sheet per input, rows = methods, columns = front, right, back, 3/4 and two close-ups
(hand and face). Close-ups are located by a silhouette heuristic (lateral extremes of the arm band; top of
the head), not by a detector, and are labelled as such. Same Lambert shading for every method.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..reconstruct.surface import Mesh
from .render import contact_sheet, frame, orbit_camera, render, shade

VIEWS = {"front": 0, "right": -90, "back": 180, "three_quarter": -35}   # right = person's right (camera at -X)


def normalise_mesh(mesh: Mesh) -> Mesh:
    """Height 1, feet on y=0, centred in x/z (canonical generator frame assumed: +Y up, faces +Z)."""
    v = mesh.vertices
    lo, hi = v.min(0), v.max(0)
    s = 1.0 / max(hi[1] - lo[1], 1e-9)
    c = (lo + hi) / 2
    return Mesh((v - [c[0], lo[1], c[2]]) * s, mesh.faces)


def silhouette(mesh: Mesh, view: str, size: int = 512) -> np.ndarray:
    cam = orbit_camera(VIEWS[view], 0, (0.0, 0.5, 0.0), 0.6, size)
    return render(mesh, cam).face_id >= 0


def crop_scale(mask: np.ndarray, height: int = 400, width: int = 400) -> np.ndarray:
    """Bounding box of the mask scaled to `height`, centred horizontally on a width x height canvas."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((height, width), bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w = max(1, round(crop.shape[1] * height / crop.shape[0]))
    im = Image.fromarray(crop.astype(np.uint8) * 255).resize((min(w, width), height), Image.NEAREST)
    out = np.zeros((height, width), bool)
    a = np.asarray(im) > 127
    x0 = (width - a.shape[1]) // 2
    out[:, x0:x0 + a.shape[1]] = a
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def heldout_iou(mesh: Mesh, ref_masks: dict[str, np.ndarray]) -> dict:
    m = normalise_mesh(mesh)
    out = {v: iou(crop_scale(silhouette(m, v)), crop_scale(ref)) for v, ref in ref_masks.items()}
    out["mean"] = float(np.mean(list(out.values()))) if out else None
    return out


def closeups(mesh: Mesh, size: int = 1024) -> dict[str, np.ndarray]:
    """Front close-ups of the person's right hand, left hand and the face (silhouette heuristic)."""
    m = normalise_mesh(mesh)
    cam = orbit_camera(0, 0, (0.0, 0.5, 0.0), 0.6, size)
    r = render(m, cam)
    img = shade(m, r)
    sil = r.face_id >= 0
    ys, xs = np.nonzero(sil)
    top, bottom = ys.min(), ys.max()
    h = bottom - top
    band = sil[int(top + 0.40 * h):int(top + 0.60 * h)]
    cols = np.nonzero(band.any(0))[0]
    rows_of = lambda c: np.nonzero(band[:, c])[0].mean() + int(top + 0.40 * h)  # noqa: E731
    half = int(0.07 * h)
    out = {}
    for name, c in (("hand_right", cols.min()), ("hand_left", cols.max())):  # front view: person's right on the image left
        cy, cx = int(rows_of(c)), int(c + (half * 0.6 if name == "hand_right" else -half * 0.6))
        out[name] = _crop(img, cx, cy, half)
    head = sil[top:int(top + 0.13 * h)]
    hc = np.nonzero(head.any(0))[0]
    out["face"] = _crop(img, int(hc.mean()), int(top + 0.065 * h), int(0.075 * h))
    return out


def _crop(img: np.ndarray, cx: int, cy: int, half: int, out: int = 256) -> np.ndarray:
    pad = np.pad(img, ((half, half), (half, half), (0, 0)), constant_values=245)
    c = pad[cy:cy + 2 * half, cx:cx + 2 * half]
    return np.asarray(Image.fromarray(c).resize((out, out), Image.LANCZOS))


def sheet(rows: dict[str, Mesh], size: int = 256) -> np.ndarray:
    """Rows = methods; columns = front, right, back, 3/4, right-hand, left-hand and face close-ups."""
    tiles, labels = [], []
    for name, mesh in rows.items():
        m = normalise_mesh(mesh)
        for v in VIEWS:
            r = render(m, orbit_camera(VIEWS[v], 5, (0.0, 0.5, 0.0), 0.6, size))
            tiles.append(shade(m, r))
            labels.append(f"{name} {v}")
        for k, im in closeups(mesh).items():
            tiles.append(np.asarray(Image.fromarray(im).resize((size, size))))
            labels.append(f"{name} {k} (heuristic)")
    return grid(tiles, labels, cols=len(VIEWS) + 3)


def grid(tiles, labels, cols: int) -> np.ndarray:
    rows = [contact_sheet(tiles[i:i + cols], labels[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    return np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=245) for r in rows])
