import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

from vector3d.export.writers import to_glb
from vector3d.lab import dataset
from vector3d.lab.align import align_to, align_to_silhouettes
from vector3d.lab.backends import get_backend
from vector3d.lab.env_audit import audit
from vector3d.lab.experiments import degrade, run_silhouette_experiment
from vector3d.lab.gates import run_gates
from vector3d.lab.mesh_io import load_mesh
from vector3d.lab.metrics import chamfer_fscore, iou, mesh_quality
from vector3d.lab.refine import RefineParams, refine_silhouettes
from vector3d.lab.render import frame, orbit_camera, render, silhouette, view_camera
from vector3d.lab.report import write_report
from vector3d.reconstruct.surface import Mesh


def box(half=(0.5, 0.3, 0.2), res=48):
    return dataset.mesh_sdf(lambda p: dataset.sd_box(p, half), res, 0.75)


def test_box_silhouette_matches_its_extent():
    m = box()
    cam = view_camera("front", (0, 0, 0), 1.0, 200)
    area = silhouette(m, cam).sum()
    expected = (1.0 * 0.6) / (2.0 / 200) ** 2
    assert abs(area - expected) / expected < 0.03


def test_orbit_matches_named_views_and_object_sides():
    for az, v in ((0, "front"), (90, "left"), (180, "back"), (270, "right")):
        a, b = orbit_camera(az), view_camera(v)
        assert np.allclose(a.right, b.right) and np.allclose(a.toward, b.toward)
    # Mug handle sits at +X, the object's left: right half of the Front image.
    m = dataset.mesh_sdf(dataset.mug, 48)
    c, h = frame(m)
    mask = silhouette(m, view_camera("front", c, h, 128))
    cols = np.nonzero(mask.any(axis=0))[0]
    centre_px = 64 - c[0] / (2 * h / 128)
    assert cols.max() - centre_px > centre_px - cols.min()


def test_metrics_identity_and_shift():
    m = box()
    r = chamfer_fscore(m, m, n=4000)
    assert r["chamfer_l1"] < 0.015 and r["fscore@0.05"] > 0.99
    shifted = Mesh(m.vertices + [0.1, 0, 0], m.faces)
    assert chamfer_fscore(shifted, m, n=4000)["chamfer_l1"] > r["chamfer_l1"]
    a = np.zeros((4, 4), bool)
    a[:2] = True
    assert iou(a, a) == 1.0 and iou(a, ~a) == 0.0
    q = mesh_quality(m)
    assert q["watertight"] and q["finite"] and q["normal_consistency"] > 0.9


def test_align_recovers_rotation_and_scale():
    gt = box()
    rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    pred = Mesh(gt.vertices @ rot.T * 3 + 5, gt.faces)
    aligned, info = align_to(pred, gt, n=2000)
    assert info["label"] == "ESTIMATED"
    assert chamfer_fscore(aligned, gt, n=4000)["fscore@0.02"] > 0.95
    cams = {v: view_camera(v, (0, 0, 0), 0.8, 128) for v in ("front", "left")}
    targets = {v: silhouette(gt, c) for v, c in cams.items()}
    placed, pinfo = align_to_silhouettes(pred, targets, cams)
    assert pinfo["mean_ref_iou"] > 0.95


def test_refinement_improves_silhouettes_without_ground_truth():
    gt = dataset.mesh_sdf(lambda p: dataset.sd_sphere(p, 0.45), 40)
    base, info = degrade(gt, seed=3)
    assert info["label"] == "SIMULATED_BASELINE"
    c, h = frame(gt)
    cams = {v: view_camera(v, c, h, 128) for v in dataset.REFERENCE_VIEWS}
    targets = {v: silhouette(gt, cams[v]) for v in cams}
    res = refine_silhouettes(base, targets, cams, RefineParams(iterations=15))
    assert res.history[res.best_iteration]["mean_iou"] > res.history[0]["mean_iou"] + 0.02
    assert chamfer_fscore(res.mesh, gt, n=4000)["chamfer_l1"] < chamfer_fscore(base, gt, n=4000)["chamfer_l1"]
    assert np.isfinite(res.mesh.vertices).all()


def test_dataset_and_experiment_roundtrip(tmp_path):
    m = dataset.build(tmp_path / "ds", resolution=40, size=96, names=["crate"])
    assert m["objects"][0]["name"] == "crate"
    d = tmp_path / "ds" / "crate"
    assert (d / "gt.glb").exists() and (d / "cameras.json").exists()
    assert set(dataset.load_cameras(d)) >= set(dataset.REFERENCE_VIEWS)
    s = run_silhouette_experiment(tmp_path / "ds", tmp_path / "exp", seeds=(0,), params=RefineParams(iterations=4))
    run = s["runs"][0]
    assert run["degradation"]["label"] == "SIMULATED_BASELINE"
    assert (tmp_path / "exp" / "crate_seed0" / "compare.png").exists()
    text = write_report(tmp_path / "r.md", s)
    assert "Not run" in text and "SIMULATED_BASELINE" in text


def test_gates_pass_real_mesh_and_fail_nan(tmp_path):
    m = box(res=24)
    p = tmp_path / "m.glb"
    p.write_bytes(to_glb(m))
    g = run_gates(p, tmp_path / "g", metadata={"model": "x", "version": "1"})
    assert g["accepted"]
    assert g["gates"]["previews_rendered"]["status"] == "pass"
    assert (tmp_path / "g" / "preview_back.png").exists()
    assert g["gates"]["blender_roundtrip"]["status"] in ("skipped", "pass")
    bad = Mesh(m.vertices.copy(), m.faces)
    bad.vertices[0] = np.nan
    pb = tmp_path / "bad.obj"
    pb.write_text("\n".join([f"v {a} {b} {c}" for a, b, c in bad.vertices] +
                            [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in bad.faces]))
    gb = run_gates(pb, tmp_path / "gb", metadata={"model": "x", "version": "1"})
    assert not gb["accepted"] and gb["gates"]["finite_coordinates"]["status"] == "fail"
    assert not run_gates(p, tmp_path / "g2")["accepted"]  # no model/version recorded


def test_load_mesh_roundtrip(tmp_path):
    m = box(res=24)
    (tmp_path / "m.glb").write_bytes(to_glb(m))
    back = load_mesh(tmp_path / "m.glb")
    assert len(back.faces) == len(m.faces)


def _png(path):
    a = np.zeros((64, 64, 4), np.uint8)
    a[16:48, 20:44] = (200, 80, 40, 255)
    Image.fromarray(a, "RGBA").save(path)
    return path


def test_triposg_blocked_without_install(tmp_path, monkeypatch):
    monkeypatch.delenv("V3D_TRIPOSG_DIR", raising=False)
    img = _png(tmp_path / "in.png")
    before = img.read_bytes()
    rec = get_backend("triposg").generate([img], out_dir=tmp_path / "run")
    assert rec.status == "blocked" and "V3D_TRIPOSG_DIR" in rec.reason
    assert rec.output_path is None
    assert img.read_bytes() == before  # original untouched
    assert (tmp_path / "run" / "inputs" / "image0.png").exists()
    assert json.loads((tmp_path / "run" / "record.json").read_text())["status"] == "blocked"


def _fake_triposg(root, with_torch: bool):
    """A stand-in TripoSG checkout for testing the subprocess plumbing only."""
    (root / "triposg" / "pipelines").mkdir(parents=True)  # no __init__.py: a namespace package, as upstream
    (root / "triposg" / "pipelines" / "pipeline_triposg.py").write_text(textwrap.dedent("""
        import numpy as np
        class _Out:
            def __init__(self):
                v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], float)
                f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
                self.samples = [(v, f)]
        class TripoSGPipeline:
            device = 'cuda'
            transformer = 'base'
            @classmethod
            def from_pretrained(cls, path): return cls()
            def to(self, *a): return self
            def __call__(self, **kw):
                import sys
                print('\\u258d', file=sys.stderr)  # tqdm bar glyph: byte 0x8d is not in cp1252
                return _Out()
        """))
    (root / "scripts").mkdir()
    (root / "scripts" / "image_process.py").write_text("def prepare_image(p, bg_color, rmbg_net=None):\n    return p\n")
    w = root / "pretrained_weights" / "TripoSG"
    w.mkdir(parents=True)
    (w / "model_index.json").write_text('{"_class_name": "TripoSGPipeline"}')
    if with_torch:
        t = root / "fake_site" / "torch"
        t.mkdir(parents=True)
        (t / "__init__.py").write_text(textwrap.dedent("""
            __version__ = 'fake'; float16 = 'float16'
            class version: cuda = 'fake'
            class _P: total_memory = 12 * 2**30
            class cuda:
                is_available = staticmethod(lambda: True)
                get_device_name = staticmethod(lambda i: 'FAKE GPU')
                get_device_properties = staticmethod(lambda i: _P())
                set_per_process_memory_fraction = staticmethod(lambda f, i: None)
                reset_peak_memory_stats = staticmethod(lambda: None)
                synchronize = staticmethod(lambda: None)
                memory_allocated = staticmethod(lambda: 0)
                max_memory_allocated = staticmethod(lambda: 5 * 2**30)
                max_memory_reserved = staticmethod(lambda: 6 * 2**30)
            def device(x): return x
            class no_grad:
                def __enter__(self): pass
                def __exit__(self, *a): pass
            class Generator:
                def __init__(self, device=None): pass
                def manual_seed(self, s): return self
            """))
    return root


def test_triposg_failure_is_reported_not_faked(tmp_path, monkeypatch):
    repo = _fake_triposg(tmp_path / "TripoSG", with_torch=False)
    monkeypatch.setenv("V3D_TRIPOSG_DIR", str(repo))
    monkeypatch.setenv("PYTHONPATH", "")
    rec = get_backend("triposg", python=sys.executable).generate({"front": _png(tmp_path / "f.png")},
                                                                out_dir=tmp_path / "run")
    has_torch = subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode == 0
    if has_torch:
        pytest.skip("real torch installed; covered by the GPU run instead")
    assert rec.status == "failed" and "torch" in rec.reason
    assert rec.output_path is None and not (tmp_path / "run" / "raw_triposg.glb").exists()


def test_triposg_plumbing_with_stub_model(tmp_path, monkeypatch):
    repo = _fake_triposg(tmp_path / "TripoSG", with_torch=True)
    monkeypatch.setenv("V3D_TRIPOSG_DIR", str(repo))
    monkeypatch.setenv("PYTHONPATH", str(repo / "fake_site"))
    rec = get_backend("triposg", python=sys.executable).generate(
        {"front": _png(tmp_path / "f.png"), "left": _png(tmp_path / "l.png")}, out_dir=tmp_path / "run", seed=7)
    assert rec.status == "success", rec.reason
    assert rec.seed == 7 and rec.peak_vram_mb == 5 * 1024 and rec.device == "FAKE GPU"
    assert rec.model_version.startswith("weights:")
    assert any("single-image" in w for w in rec.warnings)
    assert load_mesh(rec.output_path).faces.shape == (4, 3)
    assert (tmp_path / "run" / "input_rgba.png").exists()


def test_audit_reports_models():
    rep = audit()
    assert "triposg" in rep["models"] and "disk_free_gb" in rep
    if not rep["gpus"]:
        assert rep["models"]["triposg"]["verdict"] == "no NVIDIA GPU detected"


def test_generate_ai_mode_with_stub_model(tmp_path, monkeypatch):
    from vector3d.generate import GenerateOptions, generate, resolve_mode

    repo = _fake_triposg(tmp_path / "TripoSG", with_torch=True)
    monkeypatch.setenv("V3D_TRIPOSG_DIR", str(repo))
    monkeypatch.setenv("V3D_TRIPOSG_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(repo / "fake_site"))
    monkeypatch.chdir(tmp_path)
    data = _png(tmp_path / "f.png").read_bytes()
    assert resolve_mode({"front": ("f.png", data)}) == "ai"
    steps = []
    r = generate({"front": ("f.png", data)}, GenerateOptions(mode="auto", size_m=2.0),
                 progress=lambda k, s, d: steps.append((k, s)))
    assert r.metadata["mode"] == "ai" and r.metadata["geometry_confidence"] == "ai_generated"
    assert r.metadata["model"]["peak_vram_mb"] == 5 * 1024
    assert ("ai_generate", "done") in steps
    bbox = np.array(r.metadata["validation"]["bbox"])
    assert abs((bbox[1] - bbox[0]).max() - 2.0) < 1e-6
    assert r.glb[:4] == b"glTF"


def test_ai_mode_unavailable_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    from vector3d.generate import GenerateOptions, generate, resolve_mode

    monkeypatch.delenv("V3D_TRIPOSG_DIR", raising=False)
    data = _png(tmp_path / "f.png").read_bytes()
    assert resolve_mode({"front": ("f.png", data)}) == "photo"
    with pytest.raises(RuntimeError, match="TripoSG is not available"):
        generate({"front": ("f.png", data)}, GenerateOptions(mode="ai"))


def test_ai_mode_fits_the_model_to_every_view(tmp_path, monkeypatch):
    from vector3d.generate import GenerateOptions, generate

    repo = _fake_triposg(tmp_path / "TripoSG", with_torch=True)
    monkeypatch.setenv("V3D_TRIPOSG_DIR", str(repo))
    monkeypatch.setenv("V3D_TRIPOSG_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(repo / "fake_site"))
    monkeypatch.chdir(tmp_path)
    data = _png(tmp_path / "f.png").read_bytes()   # a 24 x 32 px block: X 24, Y 32, and Z 24 from the side
    seen = []
    r = generate({"front": ("f.png", data), "left": ("l.png", data)}, GenerateOptions(mode="ai"),
                 lambda k, s, d: s != "running" and seen.append(k))
    assert seen == ["load", "segment", "ai_generate", "cleanup", "normalize", "fit_views", "refine", "place",
                    "color", "validate", "export"]
    md = r.metadata
    assert md["fit"]["views"] == ["front", "left"] and md["fit"]["observed_axes"] == "XYZ"
    assert set(md["validation"]["silhouette_iou"]) == {"front", "left"}
    lo, hi = np.array(md["validation"]["bbox"])
    assert np.allclose((hi - lo) / (hi - lo).max(), [0.75, 1.0, 0.75], atol=0.05)
    assert md["geometry_confidence"] == ("estimated" if md["fit"]["refined"] else "ai_generated")


def test_ai_mode_with_lora_adapter(tmp_path, monkeypatch):
    from vector3d.generate import GenerateOptions, generate

    repo = _fake_triposg(tmp_path / "TripoSG", with_torch=True)
    peft = repo / "fake_site" / "peft"
    peft.mkdir()
    (peft / "__init__.py").write_text(
        "class PeftModel:\n"
        "    @staticmethod\n"
        "    def from_pretrained(model, path):\n"
        "        class _M:\n"
        "            def merge_and_unload(self): return model + '+lora'\n"
        "        return _M()\n")
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}")
    for k, v in (("V3D_TRIPOSG_DIR", repo), ("V3D_TRIPOSG_PYTHON", sys.executable),
                 ("PYTHONPATH", repo / "fake_site"), ("V3D_TRIPOSG_LORA", lora)):
        monkeypatch.setenv(k, str(v))
    monkeypatch.chdir(tmp_path)
    data = _png(tmp_path / "f.png").read_bytes()
    r = generate({"front": ("f.png", data)}, GenerateOptions(mode="ai", ai_lora=True))
    assert r.metadata["model"]["lora"] == str(lora)
    assert "+lora:" in r.metadata["model"]["version"]
    (lora / "adapter_config.json").unlink()
    with pytest.raises(RuntimeError, match="LoRA adapter not found"):
        generate({"front": ("f.png", data)}, GenerateOptions(mode="ai", ai_lora=True))


def test_compare_summaries(tmp_path):
    from vector3d.lab.benchmark import compare_summaries

    def run(d, f):
        (tmp_path / d).mkdir()
        rows = [{"object": "crate", "record": {"duration_s": 80},
                 "A": {"chamfer_l1": 0.1 * f, "fscore@0.02": 0.5 / f, "heldout_iou_mean": 0.6 / f,
                       "normal_consistency": 0.9}}]
        (tmp_path / d / "summary.json").write_text(json.dumps({"rows": rows}))
    run("base", 1.0)
    run("lora", 0.8)
    t = compare_summaries(tmp_path / "base", tmp_path / "lora")
    assert "| crate | 0.1000 | 0.0800 |" in t
    assert "Chamfer-L1 1/1" in t and "normal cons. 0/1" in t
