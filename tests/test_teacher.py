"""Omni teacher assembly (hand close-ups merged into the body), CPU only with a stand-in generator."""
import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial import cKDTree

from vector3d.lab import dataset as ds
from vector3d.lab import teacher
from vector3d.lab.render import frame, render, shade, view_camera
from vector3d.reconstruct.surface import Mesh

HAND_AT = np.array([0.32, 0.05, 0.12])


def hand_sdf(p):
    q = ds._t(p, *HAND_AT)
    palm = ds.sd_box(q, (0.035, 0.045, 0.012), 0.005)
    fingers = [ds.sd_capsule(q, np.array([x, 0.04, 0]), np.array([x, 0.085, 0]), 0.008)
               for x in (-0.027, -0.009, 0.009, 0.027)]
    return np.minimum.reduce([palm, *fingers])


def body_sdf(p):
    torso = ds.sd_box(p, (0.18, 0.3, 0.1), 0.02)
    arm = ds.sd_capsule(p, np.array([0.17, 0.2, 0.0]), HAND_AT - np.array([0, 0.055, 0.0]), 0.03)
    return np.minimum.reduce([torso, arm, hand_sdf(p)])


@pytest.fixture(scope="module")
def scene():
    body = ds.mesh_sdf(body_sdf, 160, 0.6)
    hand = ds.mesh_sdf(hand_sdf, 96, 0.6)
    c, h = frame(body)
    cam = view_camera("front", c, h, 1024)
    return body, hand, cam


def canonical(mesh):
    """What a generator returns: the object centred in a unit box, its own scale."""
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    return Mesh((mesh.vertices - (lo + hi) / 2) / (hi - lo).max() * 1.9, mesh.faces)


def visible_in(mesh, cam, region):
    """A perfect generator for a crop: the part of the mesh whose faces project inside the region."""
    p = cam.project(mesh.vertices)
    x, y = p[:, 0].astype(int).clip(0, cam.size - 1), p[:, 1].astype(int).clip(0, cam.size - 1)
    keep = region[y, x][mesh.faces].all(1)
    used = np.unique(mesh.faces[keep])
    remap = -np.ones(len(mesh.vertices), int)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.vertices[used], remap[mesh.faces[keep]])


def keypoints_for(hand, cam):
    """Stand-in for RTMW: the left-hand slots get 21 points spread over the projected hand (FPS), like
    wrist, knuckles and fingertips; the right hand is not detected."""
    xy, score = np.zeros((133, 2)), np.zeros(133)
    proj = cam.project(hand.vertices)[:, :2]
    idx, dist = [0], np.linalg.norm(proj - proj[0], axis=1)
    for _ in range(20):
        idx.append(int(dist.argmax()))
        dist = np.minimum(dist, np.linalg.norm(proj - proj[idx[-1]], axis=1))
    pts = proj[idx]
    xy[91:112], score[91:112] = pts, 0.9
    return {"xy": xy, "score": score}


def test_crop_camera_maps_box_to_crop(scene):
    _, hand, cam = scene
    box = teacher.hand_box(**keypoints_for(hand, cam), side="l")
    cc = teacher.crop_camera(cam, box, 512)
    p_full = cam.project(hand.vertices)[:, :2]
    p_crop = cc.project(hand.vertices)[:, :2]
    expect = (p_full - [box.cx - box.side_px / 2, box.cy - box.side_px / 2]) * 512 / box.side_px
    assert np.abs(p_crop - expect).max() < 1e-6


def test_hand_box_needs_confident_points(scene):
    _, hand, cam = scene
    kp = keypoints_for(hand, cam)
    assert teacher.hand_box(kp["xy"], kp["score"], "r") is None          # right hand never detected
    kp["score"][91:108] = 0.1
    assert teacher.hand_box(kp["xy"], kp["score"], "l") is None          # only 4 confident points


def test_run_places_hand_without_gt(scene, tmp_path):
    body, hand, cam = scene
    png = tmp_path / "front.png"
    Image.fromarray(shade(body, render(body, cam)), "RGBA").save(png)
    calls = []
    kp = keypoints_for(hand, cam)
    seen = visible_in(body, cam, teacher.region_mask(teacher.hand_box(kp["xy"], kp["score"], "l"),
                                                     (cam.size, cam.size)))

    def generate(image, out):
        calls.append(image.name)
        out.mkdir(parents=True, exist_ok=True)
        f = out / "raw.npz"
        m = canonical(seen if image.name.startswith("hand_") else body)
        np.savez(f, v=m.vertices, f=m.faces)
        return f

    load = lambda f: Mesh(np.load(f)["v"], np.load(f)["f"])  # noqa: E731
    rep = teacher.run(png, cam, kp, generate, tmp_path / "runs", crop_size=512, load=load)
    assert calls == ["front.png", "hand_l.png"]
    assert rep["uses_ground_truth"] is False
    assert rep["steps"]["hand_r"]["status"] == "skipped"
    assert rep["steps"]["hand_l"]["label"] == "ESTIMATED"
    assert rep["steps"]["merge"]["watertight"] is False
    run_dir = tmp_path / "runs" / next(p.name for p in (tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "teacher.json").read_text())["steps"]["hand_l"]["status"] == "success"
    from vector3d.lab.mesh_io import load_mesh

    merged = load_mesh(run_dir / "teacher_placed.glb")
    assert len(merged.vertices) == len(body.vertices) + len(seen.vertices)
    part = merged.vertices[len(body.vertices):]           # the placed close-up shell only
    d = cKDTree(part).query(seen.vertices)[0]            # GT of what the crop shows -> placed part
    assert np.median(d) < 0.002 and d.max() < 0.006
    assert rep["steps"]["merge"]["body_faces_removed"] > 0


def test_merge_cuts_only_inside_region_and_depth_band(scene):
    body, hand, cam = scene
    box = teacher.hand_box(**keypoints_for(hand, cam), side="l")
    region = teacher.region_mask(box, (cam.size, cam.size))
    far = Mesh(hand.vertices + np.asarray(cam.toward) * -0.5, hand.faces)   # same pixels, far behind
    _, info_far = teacher.merge(body, [(far, region)], cam)
    assert info_far["body_faces_removed"] == 0
    merged, info = teacher.merge(body, [(hand, region)], cam)
    assert info["body_faces_removed"] > 0
    assert len(merged.faces) == len(body.faces) - info["body_faces_removed"] + len(hand.faces)


def test_run_human_v1_and_scores(scene, tmp_path):
    """Batch driver on a one-person human_v1-layout folder; scoring reads the teacher outputs by protocol."""
    from vector3d.export.writers import to_glb
    from vector3d.lab import human_v1

    body, hand, _ = scene
    d = tmp_path / "data" / "person_00"
    (d / "views").mkdir(parents=True)
    (d / "gt.glb").write_bytes(to_glb(body, "gt"))
    cams = ds.cameras_for(body, 256)
    (d / "cameras.json").write_text(json.dumps({k: c.to_dict() for k, c in cams.items()}))
    png, cam = human_v1.render_front(d, 1024)
    assert png.name == "front_1024.png" and cam.size == 1024 and cam.half_size == cams["front"].half_size
    kp = keypoints_for(hand, cam)
    seen = visible_in(body, cam, teacher.region_mask(teacher.hand_box(kp["xy"], kp["score"], "l"), (1024, 1024)))

    def generate(image, out):
        out.mkdir(parents=True, exist_ok=True)
        m = canonical(seen if image.name.startswith("hand_") else body)
        np.savez(out / "raw.npz", v=m.vertices, f=m.faces)
        return out / "raw.npz"

    load = lambda f: Mesh(np.load(f)["v"], np.load(f)["f"])  # noqa: E731
    res = teacher.run_human_v1(tmp_path / "data", tmp_path / "bench", generate, lambda p: kp, size=1024, load=load)
    assert res["people"]["person_00"]["status"] == "success"
    for proto in ("teacher_front_only", "body_front_only_hires"):
        assert len(human_v1.placed_mesh(tmp_path / "bench" / "person_00", d, proto).faces) > 0

    def broken(image, out):
        raise RuntimeError("TripoSG blocked: no GPU")

    res = teacher.run_human_v1(tmp_path / "data", tmp_path / "bench2", broken, lambda p: kp, size=1024)
    assert res["people"]["person_00"]["status"] == "failed"
    assert not (tmp_path / "bench2" / "person_00" / "teacher_placed.glb").exists()
