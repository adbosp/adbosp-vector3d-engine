"""OmniForge pieces that run without a GPU: SAM 3D Body blocked path, hand probe, VAE-gate helpers."""
import json

import numpy as np
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


def test_model_space_and_self_compare():
    m = humanoid.hand_mesh(voxel=0.001)
    pts, nrm = vae_ceiling.surface_points(m, 5000)
    mp, c, s = vae_ceiling.to_model_space(pts, 0.9)
    assert np.isclose(np.abs(mp).max(), 0.9, atol=1e-3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-4)
    r = vae_ceiling.compare_hand(m, m, n=5000)
    assert r["fingers"] == 4 and r["normal_agreement"] > 0.95 and r["chamfer_mm"] < 2
