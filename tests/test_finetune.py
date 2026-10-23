"""TripoSG fine-tuning: training data and the trainer's pure helpers (no GPU, no weights)."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vector3d.lab import dataset as ds
from vector3d.lab import finetune as ft

spec = importlib.util.spec_from_file_location("tsg_ft", ft.TRAINER)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_surface_normals_point_outwards():
    mesh = ds.mesh_sdf(lambda p: ds.sd_sphere(p, 0.4), 40)
    pts, nrm = ft.surface_samples(mesh, 2000, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(pts, axis=1), 0.4, atol=0.03)
    assert (np.einsum("ij,ij->i", pts, nrm) > 0).mean() > 0.99


def test_build_dataset(tmp_path):
    m = ft.build_dataset(tmp_path, 4, seed=3, image_size=96, points=1024)
    ok = [r for r in m["samples"] if r["status"] == "ok"]
    assert ok and m["license"].startswith("procedurally generated")
    d = tmp_path / ok[0]["name"]
    z = np.load(d / "surface.npz")
    assert z["points"].shape == (1024, 3) and z["normals"].shape == (1024, 3)
    rgba = np.asarray(Image.open(d / "cond.png"))
    assert rgba.shape == (96, 96, 4) and (rgba[..., 3] > 0).mean() > 0.05


def test_prepare_cond_matches_triposg_layout(tmp_path):
    img = np.zeros((100, 200, 4), np.uint8)
    img[20:60, 50:150] = (200, 10, 10, 255)                  # 100 wide, 40 tall
    Image.fromarray(img, "RGBA").save(tmp_path / "c.png")
    out = np.asarray(trainer.prepare_cond(tmp_path / "c.png"))
    assert out.shape[0] == out.shape[1] == 120                # square, 10 % padding
    assert (out[0, 0] == 255).all() and tuple(out[60, 60]) == (200, 10, 10)


def test_normalise_and_fps():
    torch = pytest.importorskip("torch")

    p = np.random.default_rng(0).uniform(-3, 5, (500, 3))
    q = trainer.normalise(p, 0.9)
    assert np.isclose((q.max(0) - q.min(0)).max() / 2, 0.9)
    pts = torch.rand(400, 3)
    idx = trainer.farthest_point_sampling(pts, torch.zeros(400, dtype=torch.long), 0.25)
    assert len(idx) == 100 and len(set(idx.tolist())) == 100


@pytest.mark.skipif(not os.environ.get("V3D_TRIPOSG_SRC"), reason="set V3D_TRIPOSG_SRC to a TripoSG checkout")
def test_tiny_training_run(tmp_path):
    ft.build_dataset(tmp_path / "data", 6, seed=1, image_size=128, points=4096)
    r = ft.train(tmp_path / "data", tmp_path / "run", steps=4, rank=4, accum=1,
                 repo=os.environ["V3D_TRIPOSG_SRC"], tiny=True, timeout=600)
    assert r["status"] == "success", r.get("traceback")
    assert (Path(r["lora_dir"]) / "adapter_config.json").exists()
