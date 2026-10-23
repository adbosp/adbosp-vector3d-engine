"""Background removal: model routing, licences and fallbacks (AI models stubbed)."""
import numpy as np
import pytest

from vector3d.ai import segment as seg


def _photo(size=96):
    """Opaque RGBA: grey disc on a white background."""
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size / 3) ** 2
    img = np.full((size, size, 4), 255, np.uint8)
    img[disc, :3] = 90
    return img, disc


def _stub(monkeypatch, ok=("birefnet", "isnet", "birefnet-lite", "u2net")):
    calls = []

    def fake(rgb, model):
        calls.append(model)
        if model not in ok:
            raise RuntimeError(f"{model} unavailable")
        _, disc = _photo(rgb.shape[0])
        return (disc * 255).astype(np.uint8)

    monkeypatch.setattr(seg, "ai_alpha", fake)
    monkeypatch.delenv("V3D_SEGMENTER", raising=False)
    monkeypatch.setattr(seg, "has_cuda", lambda: True)
    return calls


def test_models_are_commercially_licensed():
    for key, (_, licence) in seg.MODELS.items():
        assert licence.split(" ")[0] in ("MIT", "Apache-2.0"), key
    assert all("bria" not in name and "rmbg" not in name for name, _ in seg.MODELS.values())


def test_auto_prefers_birefnet(monkeypatch):
    calls = _stub(monkeypatch)
    alpha, method = seg.matte(_photo()[0])
    assert calls == ["birefnet"] and method.startswith("birefnet (MIT")
    assert alpha.dtype == np.uint8


def test_cpu_skips_birefnet_in_auto(monkeypatch):
    calls = _stub(monkeypatch)
    monkeypatch.setattr(seg, "has_cuda", lambda: False)
    _, method = seg.matte(_photo()[0])
    assert calls == ["isnet"] and method.endswith("cpu)")
    calls.clear()
    seg.matte(_photo()[0], "birefnet")                     # still available when asked for by name
    assert calls == ["birefnet"]


def test_auto_falls_back_when_model_missing(monkeypatch):
    calls = _stub(monkeypatch, ok=("isnet",))
    _, method = seg.matte(_photo()[0])
    assert calls == ["birefnet", "isnet"] and method.startswith("isnet")
    calls = _stub(monkeypatch, ok=())
    mask, method = seg.segment(*[_photo()[0]])
    assert "birefnet" not in method and "isnet" not in method   # classic, reported as such
    assert mask.any()


def test_env_override_and_alias(monkeypatch):
    calls = _stub(monkeypatch)
    monkeypatch.setenv("V3D_SEGMENTER", "birefnet-lite")
    _, method = seg.matte(_photo()[0])
    assert calls == ["birefnet-lite"] and method.startswith("birefnet-lite")
    calls.clear()
    _, method = seg.matte(_photo()[0], "rembg")
    assert calls == ["birefnet"]


def test_existing_alpha_is_kept(monkeypatch):
    calls = _stub(monkeypatch)
    img, disc = _photo()
    img[..., 3] = np.where(disc, 200, 0)
    alpha, _ = seg.matte(img)
    assert calls == [] and alpha[disc].max() == 200 and alpha[~disc].max() == 0


def test_line_art_goes_to_classic(monkeypatch):
    calls = _stub(monkeypatch)
    img = np.full((96, 96, 4), 255, np.uint8)
    img[20:76, 20:23, :3] = img[20:76, 73:76, :3] = 0
    img[20:23, 20:76, :3] = img[73:76, 20:76, :3] = 0
    mask, method = seg.segment(img)
    assert calls == [] and "line_art" in method
    assert mask[48, 48]                                   # enclosed region filled


def test_explicit_model_errors_are_not_hidden(monkeypatch):
    _stub(monkeypatch, ok=())
    with pytest.raises(RuntimeError):
        seg.matte(_photo()[0], "birefnet")
    with pytest.raises(ValueError):
        seg.matte(_photo()[0], "rmbg-2.0")


def test_segbench_classic(tmp_path, monkeypatch):
    from vector3d.lab import dataset
    from vector3d.lab.segbench import run_segbench

    _stub(monkeypatch, ok=())
    dataset.build(tmp_path / "ds", resolution=32, size=64, names=["crate"])
    s = run_segbench(tmp_path / "ds", tmp_path / "out", methods=("classic", "birefnet"))
    rows = {r["method"]: r for r in s["rows"]}
    assert s["cases"] == 8
    assert rows["classic"]["status"] == "ok" and rows["classic"]["iou_by_background"]["plain"] > 0.9
    assert rows["birefnet"]["status"] == "failed" and "mean_iou" not in rows["birefnet"]
    assert (tmp_path / "out" / "cutouts_classic.png").exists()
