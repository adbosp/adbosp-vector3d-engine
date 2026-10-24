"""Omni mode in Generate (UI): stand-ins for TripoSG and the anatomy step, CPU only."""
import io

import numpy as np
import pytest
from PIL import Image

from vector3d import generate as G
from vector3d.export.writers import to_glb
from vector3d.lab import dataset as ds
from vector3d.lab import teacher
from vector3d.lab.render import frame, render, shade, view_camera


def person_sdf(p):
    torso = ds.sd_box(p, (0.18, 0.3, 0.1), 0.02)
    arm = ds.sd_capsule(p, np.array([0.17, 0.2, 0.0]), np.array([0.32, 0.0, 0.05]), 0.03)
    hand = ds.sd_box(ds._t(p, 0.32, -0.03, 0.05), (0.03, 0.05, 0.012), 0.005)
    return np.minimum.reduce([torso, arm, hand])


@pytest.fixture(scope="module")
def person():
    m = ds.mesh_sdf(person_sdf, 96, 0.6)
    c, h = frame(m)
    rgba = shade(m, render(m, view_camera("front", c, h, 512)))
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return m, buf.getvalue()


@pytest.fixture
def stand_ins(monkeypatch, person):
    mesh, _ = person
    calls = []

    class Ok:
        def check(self):
            return True, "stand-in"

    def fake_gen(profile, seed, **kw):
        def gen(image, out):
            calls.append(image.name)
            out.mkdir(parents=True, exist_ok=True)
            f = out / "raw_triposg.glb"
            f.write_bytes(to_glb(mesh, "stand-in"))
            return f
        return gen

    monkeypatch.setattr("vector3d.lab.backends.get_backend", lambda name, **kw: Ok())
    monkeypatch.setattr(teacher, "triposg_generate", fake_gen)
    return calls


def right_blob_keypoints(image):
    """21 left-hand keypoints spread over the rightmost tenth of the subject (where the hand is)."""
    a = np.asarray(Image.open(image))[..., 3] > 127
    ys, xs = np.nonzero(a)
    keep = xs > np.percentile(xs, 90)
    pick = np.linspace(0, keep.sum() - 1, 21).astype(int)
    xy, score = np.zeros((133, 2)), np.zeros(133)
    xy[91:112] = np.column_stack([xs[keep][pick], ys[keep][pick]])
    score[91:112] = 0.9
    return {"xy": xy, "score": score}


def test_omni_without_anatomy_reports_skipped_hands(stand_ins, person, monkeypatch):
    def blocked(out_dir, **kw):
        def kp(image):
            raise RuntimeError("anatomy step blocked: WSL not available")
        return kp

    monkeypatch.setattr(teacher, "rtmw_keypoints", blocked)
    res = G.generate({"front": ("person.png", person[1])}, G.GenerateOptions(mode="omni", segmenter="classic"))
    md = res.metadata
    assert md["mode"] == "omni" and md["status"] == "experimental"
    assert md["omni"]["anatomy"]["status"] == "skipped" and "WSL" in md["omni"]["anatomy"]["reason"]
    assert md["geometry_confidence"] == "ai_generated"
    assert stand_ins == ["front.png"]                       # body only, no hand generation
    assert len(res.glb) > 0


def test_omni_with_keypoints_adds_a_hand(stand_ins, person, monkeypatch):
    monkeypatch.setattr(teacher, "rtmw_keypoints", lambda out_dir, **kw: right_blob_keypoints)
    steps = {}
    res = G.generate({"front": ("person.png", person[1])}, G.GenerateOptions(mode="omni", segmenter="classic"),
                     lambda k, s, d: steps.__setitem__(k, s))
    md = res.metadata
    assert stand_ins == ["front.png", "hand_l.png"]
    assert md["omni"]["teacher_steps"]["hand_l"]["status"] == "success"
    assert md["omni"]["teacher_steps"]["hand_r"]["status"] == "skipped"
    assert md["omni"]["uses_ground_truth"] is False and md["validation"]["watertight"] is False
    assert steps["hands"] == "done" and steps["merge"] == "done"
    assert md["geometry_confidence"] == "estimated"


def test_omni_rejects_unknown_options():
    with pytest.raises(ValueError):
        G.generate({"front": ("p.png", b"x")}, G.GenerateOptions(mode="omni", omni_depth="nope"))
