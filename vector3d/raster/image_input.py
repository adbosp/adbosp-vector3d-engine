"""Raster views (PNG/JPG): background removal and silhouette vectorization.

SVG stays the preferred geometry source. A raster view is reduced to its
silhouette, traced into closed polylines and wrapped in an SvgDocument so
the rest of the pipeline treats both input kinds the same way. Pixel
coordinates never carry a physical scale.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from skimage import filters, measure

from ..core.geometry import Contour, line
from ..svg.parser import ScaleInfo, SvgDocument

MAX_SIDE = 1024  # larger images are downsampled; silhouettes don't need more


def load_image(data: bytes) -> np.ndarray:
    """Decode to an (H, W, 4) uint8 RGBA array."""
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGBA")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(img)


def silhouette_mask(rgba: np.ndarray) -> tuple[np.ndarray, str]:
    """Foreground mask and the method used to find it.

    Transparent images use alpha. Otherwise the background colour is the
    median of the border pixels and the foreground is whatever differs from
    it by more than an Otsu threshold.
    """
    alpha = rgba[..., 3]
    if alpha.min() < 250:
        mask, method = alpha > 127, "alpha"
    else:
        rgb = rgba[..., :3].astype(float)
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0)
        dist = np.abs(rgb - bg).max(axis=2)
        thr = max(filters.threshold_otsu(dist), 20.0) if dist.max() > 0 else 255.0
        mask, method = dist > thr, "background_color"

    area = mask.sum()
    if area == 0:
        return mask, method
    # Drop specks and pinholes; real holes and parts are far larger than 0.05% of the object.
    min_size = max(16, int(area * 0.0005))
    mask = _drop_small(mask, min_size)
    mask = ~_drop_small(~mask, min_size, keep_border=True)
    return mask, method


def _drop_small(mask: np.ndarray, min_size: int, keep_border: bool = False) -> np.ndarray:
    """Remove connected components smaller than min_size pixels.

    keep_border keeps components touching the image edge (the background when
    called on the inverted mask, so only enclosed pinholes get filled).
    """
    labels = measure.label(mask, connectivity=1)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    if keep_border:
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        keep[edge] = True
    keep[0] = False
    return keep[labels]


def mask_contours(mask: np.ndarray, tolerance: float = 0.75) -> list[np.ndarray]:
    """Closed boundary polylines in (x right, y down) pixel coordinates."""
    padded = np.pad(mask.astype(float), 1)
    out = []
    for c in measure.find_contours(padded, 0.5):
        c = measure.approximate_polygon(c, tolerance)
        if len(c) >= 4:
            out.append(c[:, ::-1] - 1.0)  # (row, col) -> (x, y), undo padding
    return out


def raster_document(data: bytes, source: str = "<image>", view_hint: str | None = None) -> SvgDocument:
    mask, method = silhouette_mask(load_image(data))
    return mask_document(mask, method, source, view_hint)


def mask_document(mask: np.ndarray, method: str, source: str = "<image>", view_hint: str | None = None) -> SvgDocument:
    """Wrap a foreground mask as an SvgDocument (pixel units, y down, unknown scale)."""
    warnings = []
    contours = []
    for poly in mask_contours(mask):
        segs = [line(a, b) for a, b in zip(poly[:-1], poly[1:])]
        contours.append(Contour(segs, closed=True, tag="raster",
                                semantic={"fill_rule": "evenodd", "source": "raster", "mask_method": method}))
    if not contours:
        warnings.append("no foreground found in image")
    h, w = mask.shape
    return SvgDocument(
        source=source, view_hint=view_hint, view_box=(0.0, 0.0, float(w), float(h)),
        contours=contours, construction=[], annotations=[], origin=None,
        scale=ScaleInfo(None, "unknown", None, ["raster image: pixels are not physical units"]),
        warnings=warnings,
    )


def is_svg(data: bytes) -> bool:
    head = data[:2048].lstrip().lower()
    return head.startswith(b"<?xml") or head.startswith(b"<svg") or b"<svg" in head
