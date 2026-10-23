"""Imagination loop v0: silhouettes stay exact, unseen cross-sections are inferred."""
import numpy as np
import pytest

from vector3d.lab import dataset
from vector3d.lab.imagine import ImagineParams, build_mesh, row_parts, run_benchmark
from vector3d.lab.metrics import iou
from vector3d.lab.render import silhouette, view_camera


def _masks(size=64):
    """A 'mug-like' scene: one body run plus a side run in the middle rows."""
    front = np.zeros((size, size), bool)
    side = np.zeros((size, size), bool)
    front[10:54, 16:40] = True
    front[24:40, 44:52] = True                 # a handle: second x-run, same z-run
    side[10:54, 20:44] = True
    return {"front": front, "back": front[:, ::-1], "left": side, "right": side[:, ::-1]}


@pytest.mark.parametrize("n,thin", [(2.0, False), (4.0, True), (60.0, False)])
def test_every_hypothesis_keeps_the_silhouettes(n, thin):
    m = _masks()
    cams = {v: view_camera(v, (0, 0, 0), 1.0, 64) for v in dataset.REFERENCE_VIEWS}
    mesh = build_mesh(m, cams, np.full(64, n), 64, np.full(64, thin))
    for v in ("front", "left"):
        assert iou(silhouette(mesh, cams[v]), m[v]) > 0.97, (v, n, thin)


def test_row_parts_split_on_structure_change():
    parts = row_parts(_masks())
    assert parts[0] == -1 and parts[10] >= 0
    assert parts[23] != parts[24] and parts[39] != parts[40]          # handle rows form their own part
    assert len(set(parts[24:40])) == 1


def test_benchmark_beats_hull_on_round_object(tmp_path):
    dataset.build(tmp_path / "ds", resolution=64, size=128, names=["lamp"])
    s = run_benchmark(tmp_path / "ds", tmp_path / "out", ImagineParams(exponents=(2.0, 6.0, 60.0)))
    row = s["rows"][0]
    assert row["imagined"]["ref_iou_mean"] > 0.95                     # silhouettes are kept
    assert row["imagined"]["heldout_iou_mean"] > row["hull"]["heldout_iou_mean"] + 0.05
    assert all(p["label"] == "ESTIMATED" for p in row["parts"])
    assert (tmp_path / "out" / "lamp" / "imagined.glb").exists()
