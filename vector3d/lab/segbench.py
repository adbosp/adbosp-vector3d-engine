"""Background-removal benchmark with exact ground truth.

The synthetic dataset's reference renders carry a perfect alpha channel.
Each render is tinted and composited onto four seeded backgrounds (plain,
gradient, noise texture, clutter), and every segmenter is scored by mask
IoU and mean absolute alpha error against that alpha, plus time per image.
Optional real photos (no ground truth) only get cut-outs for visual review.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..ai.segment import MODELS, matte
from .metrics import iou

BACKGROUNDS = ("plain", "gradient", "noise", "clutter")


def background(kind: str, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = size
    if kind == "plain":
        return np.full((h, w, 3), rng.integers(200, 256, 3), np.float32)
    if kind == "gradient":
        a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        return np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.float32)
    if kind == "noise":
        from scipy import ndimage

        n = ndimage.gaussian_filter(rng.random((h, w, 3)), (3, 3, 0))
        n = (n - n.min()) / (np.ptp(n) or 1)
        return (n * 255).astype(np.float32)
    img = np.full((h, w, 3), rng.integers(80, 200, 3), np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(25):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(size // 20, size // 5)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(0, 256, 3)
    return img


def composite(fg_rgba: np.ndarray, bg: np.ndarray, tint) -> np.ndarray:
    a = fg_rgba[..., 3:4] / 255.0
    fg = fg_rgba[..., :3] * np.asarray(tint, np.float32)
    rgb = fg * a + bg * (1 - a)
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.full(rgb.shape[:2], 255, np.uint8)])
    return out


def run_segbench(data_dir, out_dir, methods=("classic", "u2net", "isnet", "birefnet-lite", "birefnet"),
                 views=("front", "left"), images=None) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rows, sheets = [], {}
    rng = np.random.default_rng(0)
    cases = []
    for obj in manifest["objects"]:
        for v in views:
            fg = np.asarray(Image.open(data_dir / obj["name"] / "views" / f"{v}.png").convert("RGBA")).astype(np.float32)
            for k, kind in enumerate(BACKGROUNDS):
                tint = rng.uniform(0.5, 1.3, 3)
                cases.append((f"{obj['name']}/{v}/{kind}", composite(fg, background(kind, fg.shape[0], k), tint),
                              fg[..., 3] > 127, fg[..., 3]))
    for m in methods:
        scores, alpha_err, times, failed = [], [], [], None
        tiles = []
        for name, img, gt, gt_alpha in cases:
            t0 = time.perf_counter()
            try:
                alpha, used = matte(img, m)
            except Exception as exc:  # noqa: BLE001 - a missing model is reported, not scored
                failed = f"{type(exc).__name__}: {exc}"
                break
            times.append(time.perf_counter() - t0)
            scores.append(iou(alpha > 127, gt))
            alpha_err.append(float(np.abs(alpha.astype(np.float32) - gt_alpha).mean() / 255))
            if len(tiles) < 8:
                tiles.append(np.dstack([img[..., :3], alpha]))
        row = {"method": m, "status": "failed" if failed else "ok", "reason": failed,
               "licence": MODELS[m][1] if m in MODELS else "own code"}
        if not failed:
            per_bg = {kind: float(np.mean([s for (n, *_), s in zip(cases, scores) if n.endswith(kind)]))
                      for kind in BACKGROUNDS}
            row.update(mean_iou=float(np.mean(scores)), min_iou=float(np.min(scores)), iou_by_background=per_bg,
                       mean_alpha_error=float(np.mean(alpha_err)),
                       seconds_per_image=float(np.mean(times[1:] or times)), first_call_s=times[0], cases=len(scores))
            sheets[m] = tiles
        rows.append(row)
        (out_dir / "summary.json").write_text(json.dumps(   # partial, so a crash keeps finished methods
            {"benchmark": "segbench synthetic_v1", "cases": len(cases), "rows": rows, "partial": True}, indent=1),
            encoding="utf-8")
    for m, tiles in sheets.items():
        s = tiles[0].shape[0]
        sheet = Image.new("RGB", (s * len(tiles), s), (255, 0, 255))   # magenta shows through removed areas
        for i, t in enumerate(tiles):
            tile = Image.fromarray(t, "RGBA")
            sheet.paste(tile, (i * s, 0), tile)
        sheet.save(out_dir / f"cutouts_{m}.png")
    real = []
    for p in sorted(Path(images).glob("*")) if images else []:
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        rgba = np.asarray(Image.open(p).convert("RGBA"))
        for m in methods:
            try:
                alpha, used = matte(rgba, m)
            except Exception:  # noqa: BLE001
                continue
            Image.fromarray(np.dstack([rgba[..., :3], alpha]), "RGBA").save(out_dir / f"real_{p.stem}_{m}.png")
            real.append({"image": p.name, "method": m, "used": used})
    summary = {"benchmark": "segbench synthetic_v1", "cases": len(cases), "backgrounds": BACKGROUNDS,
               "rows": rows, "real_images": real}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
