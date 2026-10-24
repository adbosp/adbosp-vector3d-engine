"""human_v1 region metrics and layout, without WSL or a GPU."""
import json

import numpy as np

from vector3d.lab import human_v1, humanoid
from vector3d.reconstruct.surface import Mesh


def _joints(offset=(0.0, 0.0, 0.0)):
    o = np.array(offset)
    base = {"wrist": (0, 0, 0), "thumb_null": (-0.08, 0.08, 0.02), "index_null": (-0.02, 0.17, 0),
            "middle_null": (0.0, 0.18, 0), "ring_null": (0.02, 0.17, 0), "pinky_null": (0.04, 0.15, 0),
            "index1": (-0.02, 0.1, 0), "pinky1": (0.04, 0.09, 0)}
    return {f"r_{k}": np.array(v) + o for k, v in base.items()}


def test_region_box_covers_hand_joints():
    lo, hi = human_v1.hand_region(_joints(), "r", margin=0.01)
    assert np.allclose(lo, [-0.09, -0.01, -0.01]) and np.allclose(hi, [0.05, 0.19, 0.03])


def test_region_metrics_zero_for_identical_and_grow_with_shift():
    hand = humanoid.hand_mesh(voxel=0.002)
    box = (hand.vertices.min(0), hand.vertices.max(0))
    same = human_v1.region_metrics(hand, hand, box, n=20000)
    shifted = human_v1.region_metrics(Mesh(hand.vertices + [0.01, 0, 0], hand.faces), hand, box, n=20000)
    assert same["chamfer_mm"] < 1.0 and same["fscore@5mm"] > 0.99
    # nearest-surface distance of a 10 mm sideways shift is well under 10 mm (surfaces overlap laterally)
    assert shifted["chamfer_mm"] > same["chamfer_mm"] + 1.5 and shifted["fscore@5mm"] < same["fscore@5mm"]


def test_local_variant_removes_placement_offset():
    hand = humanoid.hand_mesh(voxel=0.002)
    box = (hand.vertices.min(0), hand.vertices.max(0))
    moved = Mesh(hand.vertices + [0.02, -0.01, 0.015], hand.faces)
    placed = human_v1.region_metrics(moved, hand, box, n=20000)
    local = human_v1.region_metrics_local(moved, hand, box)
    assert local["chamfer_mm"] < 1.0 < placed["chamfer_mm"]
    assert 20 < local["icp_shift_mm"] < 32                      # |(20, -10, 15)| mm = 26.9 mm


def test_region_metrics_reports_empty_region():
    hand = humanoid.hand_mesh(voxel=0.002)
    far = (np.array([5.0, 5, 5]), np.array([6.0, 6, 6]))
    assert human_v1.region_metrics(hand, hand, far, n=5000)["status"] == "empty"


def test_build_layout(tmp_path):
    hand = humanoid.hand_mesh(voxel=0.003)
    names = np.array(["r_" + j for j in human_v1.HAND_JOINTS])
    np.savez(tmp_path / "person_00.npz", vertices=hand.vertices.astype(np.float32), faces=hand.faces,
             joints=np.zeros((len(names), 3), np.float32), joint_names=names)
    m = human_v1.build(tmp_path, tmp_path / "out", size=64)
    d = tmp_path / "out" / "person_00"
    assert m["version"] == "human_v1" and m["objects"][0]["category"] == "human"
    assert (d / "gt.glb").is_file() and (d / "views" / "front.png").is_file() and (d / "views" / "top.png").is_file()
    assert set(json.loads((d / "joints.json").read_text())) == set(names)
