"""OmniForge pieces that run without a GPU: SAM 3D Body blocked path, hand probe, VAE-gate helpers."""
import json
import sys

import numpy as np
import pytest
from PIL import Image

from vector3d.lab import humanoid, vae_ceiling
from vector3d.lab.backends import get_backend
from vector3d.lab.backends.sam3d_body import AWAITING_REVIEW, NO_RUNNER, SAM3DBodyBackend


def _image(tmp_path):
    p = tmp_path / "person.png"
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(p)
    return p


def test_sam3d_blocked_without_weights(tmp_path):
    be = SAM3DBodyBackend(weights=str(tmp_path / "missing"))
    assert be.check() == (False, AWAITING_REVIEW)
    rec = be.generate({"front": _image(tmp_path)}, out_dir=tmp_path / "run")
    assert rec.status == "blocked" and rec.reason == AWAITING_REVIEW
    assert rec.output_path is None
    saved = json.loads((tmp_path / "run" / "record.json").read_text())
    assert saved["status"] == "blocked" and "model.ckpt" in saved["config"]["missing"]
    assert not list((tmp_path / "run").glob("*.glb"))


def test_sam3d_unverified_download_still_blocked(tmp_path):
    w = tmp_path / "w"
    (w / "assets").mkdir(parents=True)
    (w / "model.ckpt").write_bytes(b"x")
    (w / "assets" / "mhr_model.pt").write_bytes(b"x")
    assert SAM3DBodyBackend(weights=str(w)).check() == (False, AWAITING_REVIEW)   # no verification manifest
    (w / "download-verification.json").write_text("{}")
    be = SAM3DBodyBackend(weights=str(w))
    assert be.check() == (False, NO_RUNNER)
    assert be.generate([_image(tmp_path)], out_dir=tmp_path / "run").status == "blocked"


def test_sam3d_registered():
    assert isinstance(get_backend("sam3d_body", weights="nowhere"), SAM3DBodyBackend)


def test_hand_probe_has_four_separate_fingers():
    m = humanoid.hand_mesh(voxel=0.001)
    assert humanoid.count_fingers(m, voxel=0.001) == 4
    lo, hi = m.vertices.min(0), m.vertices.max(0)
    assert 0.15 < hi[1] - lo[1] < 0.20          # wrist to middle fingertip, metres


def test_fused_fingers_count_as_one_region():
    p = humanoid.HandParams(gap=-0.004)            # fingers overlap: one solid slab
    assert humanoid.count_fingers(humanoid.hand_mesh(p, voxel=0.001), p, voxel=0.001) < 4


def test_reference_params_are_not_reported_as_measurements():
    gt = humanoid.hand_mesh(voxel=0.001)
    flat = humanoid.hand_mesh(humanoid.HandParams(ridge=0.0, crease=0.0), voxel=0.001)
    measured = vae_ceiling.compare_hand(flat, gt, n=5000)
    assert not {"crease_depth_mm", "ridge_amplitude_mm", "finger_gap_mm"} & set(measured)
    ref = vae_ceiling.reference_params(humanoid.HandParams())
    assert ref["ridge_amplitude_mm"] == 0.5         # describes the GT generator, whatever the prediction
    assert measured["chamfer_palm_ridges_mm"] > vae_ceiling.compare_hand(gt, gt, n=5000)["chamfer_palm_ridges_mm"]


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_invoke_rejects_child_that_dies_before_writing(tmp_path):
    rr = vae_ceiling.invoke(_child("import sys; sys.exit(1)"), tmp_path / "r.json", tmp_path / "m.glb")
    assert rr["status"] == "failed" and rr["returncode"] == 1 and "no result file" in rr["reason"]


def test_invoke_refuses_stale_success(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"status": "success"}))
    with pytest.raises(FileExistsError):
        vae_ceiling.invoke(_child("pass"), tmp_path / "r.json", tmp_path / "m.glb")


def test_invoke_rejects_nonzero_exit_even_with_success_json(tmp_path):
    res, mesh = tmp_path / "r.json", tmp_path / "m.glb"
    code = (f"import json,sys; open(r'{mesh}','w').write('x'); "
            f"json.dump({{'status':'success'}}, open(r'{res}','w')); sys.exit(3)")
    rr = vae_ceiling.invoke(_child(code), res, mesh)
    assert rr["status"] == "failed" and rr["returncode"] == 3


def test_invoke_accepts_fresh_success(tmp_path):
    res, mesh = tmp_path / "r.json", tmp_path / "m.glb"
    code = f"import json; open(r'{mesh}','w').write('x'); json.dump({{'status':'success','duration_s':1}}, open(r'{res}','w'))"
    rr = vae_ceiling.invoke(_child(code), res, mesh)
    assert rr["status"] == "success" and rr["duration_s"] == 1


def test_bbox_fit_maps_scaled_copy_back():
    m = humanoid.hand_mesh(voxel=0.002)
    moved = vae_ceiling.Mesh(m.vertices * 7.0 + 1.5, m.faces)
    fitted, info = vae_ceiling.bbox_fit(moved, m)
    assert np.allclose(fitted.vertices, m.vertices, atol=1e-9) and np.isclose(info["scale"], 1 / 7)


def test_model_space_and_self_compare():
    m = humanoid.hand_mesh(voxel=0.001)
    pts, nrm = vae_ceiling.surface_points(m, 5000)
    mp, c, s = vae_ceiling.to_model_space(pts, 0.9)
    assert np.isclose(np.abs(mp).max(), 0.9, atol=1e-3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-4)
    r = vae_ceiling.compare_hand(m, m, n=5000)
    assert r["fingers"] == 4 and r["normal_agreement"] > 0.95 and r["chamfer_mm"] < 2
