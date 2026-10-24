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
    same = human_v1.region_metrics(hand, hand, box)
    shifted = human_v1.region_metrics(Mesh(hand.vertices + [0.01, 0, 0], hand.faces), hand, box)
    assert same["chamfer_mm"] < 1.0 and same["fscore@5mm"] > 0.99
    # nearest-surface distance of a 10 mm sideways shift is well under 10 mm (surfaces overlap laterally)
    assert shifted["chamfer_mm"] > same["chamfer_mm"] + 1.5 and shifted["fscore@5mm"] < same["fscore@5mm"]


def test_local_variant_removes_placement_offset():
    hand = humanoid.hand_mesh(voxel=0.002)
    box = (hand.vertices.min(0), hand.vertices.max(0))
    moved = Mesh(hand.vertices + [0.02, -0.01, 0.015], hand.faces)
    placed = human_v1.region_metrics(moved, hand, box)
    local = human_v1.region_metrics_local(moved, hand, box)
    assert local["chamfer_mm"] < 1.0 < placed["chamfer_mm"]
    assert 20 < local["icp_shift_mm"] < 32                      # |(20, -10, 15)| mm = 26.9 mm


def test_noise_floor_same_for_placed_and_local():
    hand = humanoid.hand_mesh(voxel=0.002)
    body = Mesh(np.concatenate([hand.vertices, hand.vertices + [0.5, 0, 0]]),
                np.concatenate([hand.faces, hand.faces + len(hand.vertices)]))   # a second hand far away
    box = (hand.vertices.min(0), hand.vertices.max(0))
    placed = human_v1.region_metrics(body, body, box)
    local = human_v1.region_metrics_local(body, body, box)
    assert placed["chamfer_mm"] < 0.6 and abs(placed["chamfer_mm"] - local["chamfer_mm"]) < 0.15


def test_front_only_placement_uses_front_view_only():
    from vector3d.lab import dataset
    from vector3d.lab.align import place_front_only
    from vector3d.lab.render import silhouette

    hand = humanoid.hand_mesh(voxel=0.003)
    cams = dataset.cameras_for(hand, 256)
    mask = silhouette(hand, cams["front"])
    raw = Mesh((hand.vertices - hand.vertices.mean(0)) * 4.0 + [1.0, 2.0, 3.0], hand.faces)
    placed, info = place_front_only(raw, mask, cams["front"])
    lo, hi = placed.vertices.min(0), placed.vertices.max(0)
    glo, ghi = hand.vertices.min(0), hand.vertices.max(0)
    pix = 2 * cams["front"].half_size / cams["front"].size
    assert np.allclose(lo[:2], glo[:2], atol=2 * pix) and np.allclose(hi[:2], ghi[:2], atol=2 * pix)
    assert np.isclose((lo[2] + hi[2]) / 2, cams["front"].center[2], atol=1e-9)
    assert info["inputs"] == ["front silhouette", "front camera"]


def test_summarize_counts_empty_hands_as_zero():
    s = {"p0": {"l": {"placed": {"chamfer_mm": 10.0, "fscore@5mm": 0.4}},
                "r": {"placed": {"status": "empty"}}},
         "protocol": "front_only"}
    out = human_v1.summarize(s, "placed")
    assert out["hands"] == 2 and out["measured"] == 1 and out["coverage"] == 0.5
    assert out["fscore@5mm_mean_all"] == 0.2 and out["fscore@5mm_mean_measured"] == 0.4


def test_region_metrics_reports_empty_region():
    hand = humanoid.hand_mesh(voxel=0.002)
    far = (np.array([5.0, 5, 5]), np.array([6.0, 6, 6]))
    assert human_v1.region_metrics(hand, hand, far)["status"] == "empty"


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
