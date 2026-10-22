"""Foreground segmentation for photos.

Uses rembg (U^2-Net, downloaded on first use) when it is installed, else the
deterministic background-colour method from vector3d.raster.
"""
from __future__ import annotations

import numpy as np

from ..raster.image_input import _drop_small, silhouette_mask


def segment(rgba: np.ndarray, backend: str = "auto") -> tuple[np.ndarray, str]:
    """Foreground mask and the method used. backend: auto | rembg | classic."""
    if rgba[..., 3].min() < 250:                     # already cut out
        return silhouette_mask(rgba)
    if backend in ("auto", "rembg"):
        try:
            from rembg import remove

            alpha = np.asarray(remove(rgba[..., :3], only_mask=True, post_process_mask=True))
            mask = alpha > 127
            if mask.any():
                area = int(mask.sum())
                mask = _drop_small(mask, max(16, area // 2000))
                return mask, "rembg"
        except Exception:
            if backend == "rembg":
                raise
    return silhouette_mask(rgba)
