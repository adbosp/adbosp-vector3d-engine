"""Line drawings: turn strokes into the solid region they outline.

A line drawing's foreground is only its strokes, so its raw silhouette is a
thin ring and the visual hull collapses to fragments. When the strokes
cover little of the region they enclose, we bridge small gaps with a
morphological closing and fill everything the outline surrounds. This is
an assumption about the drawing (ESTIMATED), and it is reported as one.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

LINE_ART_RATIO = 0.5  # strokes cover less than this share of the enclosed region


def fill_line_art(mask: np.ndarray, gap_px: int | None = None) -> tuple[np.ndarray, bool]:
    """(filled mask, True) for line art, else (mask, False)."""
    if not mask.any():
        return mask, False
    gap = gap_px if gap_px is not None else max(2, round(0.012 * max(mask.shape)))
    pad = gap + 2
    m = np.pad(mask, pad)
    st = ndimage.generate_binary_structure(2, 1)
    closed = ndimage.binary_closing(m, st, iterations=gap)
    filled = ndimage.binary_fill_holes(closed)[pad:-pad, pad:-pad]
    if mask.sum() >= LINE_ART_RATIO * filled.sum():
        return mask, False
    return filled | mask, True
