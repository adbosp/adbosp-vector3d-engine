"""Omni distillation data builder: teacher outputs become TripoSG training pairs only when they beat it."""
import json

import numpy as np
import pytest
from PIL import Image

from vector3d.export.writers import to_glb
from vector3d.lab import dataset as ds
from vector3d.lab import omni_distill as od
from vector3d.lab.render import render, shade
from vector3d.reconstruct.surface import Mesh


def body_sdf(p):
    torso = ds.sd_box(p, (0.16, 0.3, 0.12), 0.03)
    head = ds.sd_sphere(ds._t(p, 0, 0.42, 0), 0.1)
    arm = ds.sd_capsule(p, np.array([0.17, 0.2, 0.0]), np.array([0.3, -0.1, 0.1]), 0.035)
    return np.minimum.reduce([torso, head, arm])


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("people")
    mesh = ds.mesh_sdf(body_sdf, 96, 0.7)
    for name in ("person_a", "person_b"):
        d = root / name
        (d / "views").mkdir(parents=True)
        cams = ds.cameras_for(mesh, 128)
        for v, cam in cams.items():
            Image.fromarray(shade(mesh, render(mesh, cam)), "RGBA").save(d / "views" / f"{v}.png")
        (d / "cameras.json").write_text(json.dumps({k: c.to_dict() for k, c in cams.items()}))
        (d / "gt.glb").write_bytes(to_glb(mesh, "gt"))
    return root, mesh


def gen_of(make):
    def gen(image, out):
        out.mkdir(parents=True, exist_ok=True)
        m = make(image)
        (out / "raw.glb").write_bytes(to_glb(m, "raw"))
        return out / "raw.glb"
    return gen


def flat(mesh):
    """TripoSG-like failure: depth squashed to a plate."""
    c = mesh.vertices.mean(0)
    return Mesh(c + (mesh.vertices - c) * [1, 1, 0.25], mesh.faces)


def test_teacher_accepted_only_when_it_beats_the_base(data, tmp_path):
    root, mesh = data
    gens = {"triposg": gen_of(lambda img: flat(mesh) if "person_a" in str(img) else mesh),
            "trellis": gen_of(lambda img: mesh if "person_a" in str(img) else flat(mesh)),
            "broken": gen_of(lambda img: (_ for _ in ()).throw(RuntimeError("blocked: no CUDA")))}
    man = od.build(root, tmp_path / "out", gens, points=2048)
    rows = {r["name"]: r for r in man["samples"]}
    assert rows["person_a"]["status"] == "accepted" and rows["person_a"]["accepted"] == "trellis"
    assert rows["person_b"]["status"] == "rejected"
    assert "error" in rows["person_a"]["scores"]["broken"]
    assert "front" not in rows["person_a"]["heldout_views"] and "back" in rows["person_a"]["heldout_views"]
    s = tmp_path / "out" / "person_a"
    z = np.load(s / "surface.npz")
    assert z["points"].shape == (2048, 3) and np.allclose(np.linalg.norm(z["normals"], axis=1), 1, atol=1e-3)
    prov = json.loads((s / "provenance.json").read_text())
    assert prov["label"] == "AI_GENERATED" and prov["uses_ground_truth"] is False
    assert not (tmp_path / "out" / "person_b").exists()
    assert man["accepted"] == 1 and man["version"] == "v3d_finetune_v1"


def test_camera_frame_and_arbiter_rules():
    from vector3d.lab.render import view_camera

    cam = view_camera("right", (0.0, 0.0, 0.0), 1.0, 64)
    m = Mesh(np.array([[0.0, 0, 0], [np.array(cam.toward)[0], 0, np.array(cam.toward)[2]], [0, 1, 0]]),
             np.array([[0, 1, 2]]))
    assert np.allclose(od.to_camera_frame(m, cam).vertices[1], [0, 0, 1])
    base = {"mean_iou": 0.6, "normal_consistency": 0.99}
    assert od.arbitrate({"b": base, "t": {"mean_iou": 0.61, "normal_consistency": 0.99}}, "b")["accepted"] is None
    assert od.arbitrate({"b": base, "t": {"mean_iou": 0.7, "normal_consistency": 0.95}}, "b")["accepted"] is None
    assert od.arbitrate({"b": base, "t": {"mean_iou": 0.7, "normal_consistency": 0.985}}, "b")["accepted"] == "t"
