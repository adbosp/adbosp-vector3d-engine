import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vector3d.agent import describe
from vector3d.export.writers import to_glb, to_obj
from vector3d.generate import GenerateOptions, generate, load_view
from vector3d.normalize.normalizer import normalize_views
from vector3d.raster.image_input import raster_document, silhouette_mask
from vector3d.reconstruct.hints import validate_hints
from vector3d.ui.server import JOBS, start_job

EX = Path(__file__).resolve().parent.parent / "examples"


def box_inputs():
    return {v: (f"box_{v}.svg", (EX / f"box_{v}.svg").read_bytes()) for v in ("front", "right", "top")}


def png(draw, size=(200, 120), bg="white", mode="RGB"):
    img = Image.new(mode, size, bg)
    draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_box_with_hole_is_watertight_and_exact():
    r = generate(box_inputs(), GenerateOptions(resolution=64, smooth_iterations=0))
    v = r.metadata["validation"]
    assert v["watertight"] and v["non_manifold_edges"] == 0
    assert np.allclose(v["bbox"], [[-1, -0.5, -1.5], [1, 0.5, 1.5]], atol=1e-6)
    expected = 2 * 1 * 3 - np.pi * 0.25 ** 2 * 3
    assert abs(v["volume"] - expected) / expected < 0.03
    assert all(iou > 0.99 for iou in v["silhouette_iou"].values())
    assert r.metadata["geometry_confidence"] == "estimated"


def test_progress_reports_every_step_in_order():
    seen = []
    generate(box_inputs(), GenerateOptions(resolution=32), lambda k, s, d: s != "running" and seen.append((k, s)))
    keys = [k for k, _ in seen]
    assert keys == ["load", "normalize", "semantics", "carve", "ai_fill", "surface", "smooth", "validate", "export"]
    assert dict(seen)["semantics"] == "skipped"


def test_raster_background_and_alpha_masks():
    data = png(lambda d: d.rectangle([50, 30, 149, 89], fill=(200, 30, 30)))
    mask, method = silhouette_mask(np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")))
    assert method == "background_color" and mask.sum() == 100 * 60
    data = png(lambda d: d.ellipse([20, 20, 80, 80], fill=(0, 0, 0, 255)), size=(100, 100), bg=(0, 0, 0, 0), mode="RGBA")
    mask, method = silhouette_mask(np.asarray(Image.open(io.BytesIO(data))))
    assert method == "alpha" and abs(mask.sum() - np.pi * 30.5 ** 2) / mask.sum() < 0.05


def test_raster_hole_is_kept():
    data = png(lambda d: (d.rectangle([20, 20, 179, 99], fill="black"), d.ellipse([80, 40, 120, 80], fill="white")))
    doc = raster_document(data)
    assert len(doc.contours) == 2 and not doc.scale.known


def test_single_image_extrudes_with_assumption():
    data = png(lambda d: d.rectangle([50, 30, 149, 89], fill="black"))
    r = generate({"front": ("f.png", data)}, GenerateOptions(resolution=32, mode="drawing"))
    assert r.metadata["units"] == "drawing_units"
    assert any("Z depth not visible" in a for a in r.metadata["assumptions"])
    assert r.metadata["validation"]["watertight"]


def test_revolve_hint_builds_round_solid():
    data = png(lambda d: d.rectangle([60, 10, 139, 109], fill="black"))  # cylinder seen from the front
    hints = {"author": "test", "revolve": {"view": "front", "axis": "y"}}
    r = generate({"front": ("f.png", data)}, GenerateOptions(resolution=64, smooth_iterations=0, hints=hints, mode="drawing"))
    v = r.metadata["validation"]
    (x0, _, z0), (x1, _, z1) = v["bbox"]
    assert abs((x1 - x0) - (z1 - z0)) < 0.05 * (x1 - x0)          # round in plan
    radius, height = 40, 100
    assert abs(v["volume"] - np.pi * radius ** 2 * height) / v["volume"] < 0.06
    assert r.metadata["geometry_confidence"] == "ai_generated"
    assert v["silhouette_iou"]["front"] > 0.99


def test_primitive_hint_subtracts():
    hints = {"primitives": [{"op": "subtract", "type": "box", "center": [0, 0.5, 0], "size": [1, 0.5, 1]}]}
    base = generate(box_inputs(), GenerateOptions(resolution=48, smooth_iterations=0)).metadata["validation"]["volume"]
    cut = generate(box_inputs(), GenerateOptions(resolution=48, smooth_iterations=0, hints=hints)).metadata["validation"]["volume"]
    assert base - cut == pytest.approx(1 * 0.25 * 1, rel=0.1)


@pytest.mark.parametrize("bad", [
    {"extent": {"w": [0, 1]}},
    {"extent": {"z": [1, 0]}},
    {"revolve": {"axis": "q"}},
    {"primitives": [{"op": "melt", "type": "box", "center": [0, 0, 0]}]},
    {"primitives": [{"op": "add", "type": "cone", "center": [0, 0, 0]}]},
])
def test_invalid_hints_rejected(bad):
    with pytest.raises(ValueError):
        validate_hints(bad)


def test_glb_and_obj_are_well_formed():
    r = generate(box_inputs(), GenerateOptions(resolution=24))
    magic, version, length = np.frombuffer(r.glb[:12], dtype="<u4")
    assert magic == 0x46546C67 and version == 2 and length == len(r.glb)
    jlen = int(np.frombuffer(r.glb[12:16], dtype="<u4")[0])
    gltf = json.loads(r.glb[20:20 + jlen])
    assert gltf["accessors"][2]["count"] == r.metadata["validation"]["triangles"] * 3
    assert r.obj.count("\nf ") == r.metadata["validation"]["triangles"]


def test_describe_reports_symmetry_and_unobserved_axes():
    data = png(lambda d: d.polygon([(100, 10), (160, 110), (40, 110)], fill="black"))
    facts = describe(normalize_views({"front": load_view("front", "t.png", data)}), resolution=48)
    v = facts["views"]["front"]
    assert facts["unobserved_axes"] == ["Z"]
    assert v["mirror_symmetry"]["about_vertical_centerline"] > 0.95
    assert v["mirror_symmetry"]["about_horizontal_centerline"] < 0.8
    prof = [p for p in v["width_profile_bottom_to_top"] if p > 0]
    assert prof[0] > prof[-1]                                      # wide base, narrow top


def test_job_runs_in_background():
    views = {v: {"name": n, "data": base64.b64encode(d).decode()} for v, (n, d) in box_inputs().items()}
    job = start_job({"views": views, "options": {"resolution": 24}})
    deadline = time.time() + 30
    while job.status not in ("done", "error") and time.time() < deadline:
        time.sleep(0.05)
    snap = JOBS[job.id].snapshot()
    assert snap["status"] == "done", snap["error"]
    assert all(s["status"] in ("done", "skipped") for s in snap["steps"])
    assert snap["files"]["glb"].endswith("model.glb")


def test_job_rejects_bad_input_before_starting():
    with pytest.raises(ValueError):
        start_job({"views": {"front": {"name": "x.png", "data": base64.b64encode(b"not an image").decode()}}})


# ---------- photo mode ----------

def snowman_jpg(noise: float = 3.0) -> bytes:
    H, W = 240, 180
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.dstack([200 + 20 * yy / H, 205 + 15 * yy / H, 215 + 10 * yy / H])
    for cx, cy, r, col in [(90, 165, 55, (230, 80, 60)), (90, 80, 38, (240, 200, 70))]:
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        z = np.sqrt(np.clip(r * r - d2, 0, None)) / r
        for c in range(3):
            img[..., c] = np.where(d2 < r * r, col[c] * (0.4 + 0.6 * z), img[..., c])
    img = np.clip(img + np.random.default_rng(0).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def fake_depth_model(tmp_path, monkeypatch):
    """A stand-in ONNX model with Depth Anything's I/O signature: depth = mean of the RGB channels."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("predicted_depth", TensorProto.FLOAT, [1, "h", "w"])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("ReduceMean", ["pixel_values", "axes"], ["predicted_depth"], keepdims=0)
    model = helper.make_model(helper.make_graph([node], "fake", [inp], [out], initializer=[axes]),
                              opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    path = tmp_path / "fake_depth.onnx"
    onnx.save(model, path)
    monkeypatch.setenv("V3D_DEPTH_MODEL", str(path))
    return path


def test_auto_mode_picks_photo_for_raster_only():
    from vector3d.generate import resolve_mode

    assert resolve_mode({"front": ("a.jpg", snowman_jpg())}) == "photo"
    assert resolve_mode(box_inputs()) == "drawing"
    with pytest.raises(ValueError):
        resolve_mode(box_inputs(), "sculpt")


def test_photo_mode_with_inflation_backend():
    seen = []
    r = generate({"front": ("s.jpg", snowman_jpg())},
                 GenerateOptions(resolution=64, depth_backend="inflation", segmenter="classic"),
                 lambda k, s, d: s != "running" and seen.append(k))
    v = r.metadata["validation"]
    assert seen == [k for k, _ in __import__("vector3d.generate", fromlist=["PHOTO_STEPS"]).PHOTO_STEPS]
    assert v["watertight"] and r.metadata["mode"] == "photo"
    assert max(np.subtract(v["bbox"][1], v["bbox"][0])) == pytest.approx(1.0, rel=1e-6)
    assert v["silhouette_iou"]["front"] > 0.97
    assert set(r.images) == {"mask.png", "depth.png", "normal.png"}
    assert r.metadata["geometry_confidence"] == "ai_generated"


def test_photo_mode_runs_onnx_depth_and_colours(fake_depth_model):
    r = generate({"front": ("s.jpg", snowman_jpg())}, GenerateOptions(resolution=64, segmenter="classic"))
    assert r.metadata["depth_backend"] == "depth-anything-v2-small"
    assert r.metadata["validation"]["watertight"]
    jlen = int(np.frombuffer(r.glb[12:16], dtype="<u4")[0])
    gltf = json.loads(r.glb[20:20 + jlen])
    assert "COLOR_0" in gltf["meshes"][0]["primitives"][0]["attributes"]


def test_photo_depth_falls_back_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("V3D_DEPTH_MODEL", str(tmp_path / "missing.onnx"))
    monkeypatch.setenv("V3D_DEPTH_URL", "http://127.0.0.1:9/none.onnx")
    from vector3d.ai.depth import estimate_depth

    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    res = estimate_depth(np.zeros((40, 40, 3), np.uint8), mask)
    assert res.backend == "inflation" and "unavailable" in res.note
    assert res.depth[20, 20] == pytest.approx(1.0) and res.depth[0, 0] == 0


def test_flat_back_is_thinner_than_symmetric():
    opts = dict(resolution=48, depth_backend="inflation", segmenter="classic", size_m=1.0)
    sym = generate({"front": ("s.jpg", snowman_jpg())}, GenerateOptions(**opts)).metadata["validation"]
    flat = generate({"front": ("s.jpg", snowman_jpg())}, GenerateOptions(back="flat", **opts)).metadata["validation"]
    depth = lambda v: v["bbox"][1][2] - v["bbox"][0][2]  # noqa: E731
    assert depth(flat) < 0.8 * depth(sym)


def test_side_photo_sets_depth():
    side = png(lambda d: d.rectangle([60, 10, 99, 109], fill="black"), size=(160, 120))   # 40 px deep, 100 px tall
    front = png(lambda d: d.rectangle([40, 10, 119, 109], fill="black"), size=(160, 120))  # 80 px wide, 100 px tall
    r = generate({"front": ("f.png", front), "right": ("r.png", side)},
                 GenerateOptions(resolution=64, depth_backend="inflation", segmenter="classic", mode="photo"))
    (x0, y0, z0), (x1, y1, z1) = r.metadata["validation"]["bbox"]
    assert (z1 - z0) / (y1 - y0) == pytest.approx(0.4, abs=0.06)
