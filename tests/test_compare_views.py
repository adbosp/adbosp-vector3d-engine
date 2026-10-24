"""compare_views: shape-only held-out silhouette IoU and comparison sheets (CPU)."""
import numpy as np

from vector3d.lab import compare_views as cv
from vector3d.lab import humanoid
from vector3d.reconstruct.surface import Mesh


def _figure():
    hand = humanoid.hand_mesh(voxel=0.004)
    return Mesh(hand.vertices * [1.0, 4.0, 1.0], hand.faces)       # a tall shape with arms (thumb) to one side


def test_crop_scale_ignores_scale_and_shift():
    a = np.zeros((300, 300), bool)
    a[50:250, 100:140] = True
    b = np.zeros((600, 500), bool)
    b[10:410, 300:380] = True                                       # same shape, twice as big, moved
    assert cv.iou(cv.crop_scale(a), cv.crop_scale(b)) > 0.98


def test_heldout_iou_of_own_silhouettes_is_near_one():
    m = _figure()
    n = cv.normalise_mesh(m)
    refs = {v: cv.silhouette(n, v) for v in ("right", "back")}
    moved = Mesh(m.vertices * 3.0 + [5, 1, -2], m.faces)          # scale and position must not matter
    r = cv.heldout_iou(moved, refs)
    assert r["right"] > 0.97 and r["back"] > 0.97


def test_heldout_iou_drops_for_a_different_shape():
    m = _figure()
    refs = {"right": cv.silhouette(cv.normalise_mesh(m), "right")}
    flat = Mesh(m.vertices * [1.0, 1.0, 0.2], m.faces)              # much thinner in depth
    assert cv.heldout_iou(flat, refs)["right"] < 0.8


def test_sheet_has_one_row_per_method():
    m = _figure()
    s = cv.sheet({"a": m, "b": m}, size=64)
    assert s.ndim == 3 and s.shape[0] > 2 * 64
