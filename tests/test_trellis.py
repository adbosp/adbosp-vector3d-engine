"""TRELLIS backend pieces that run without WSL or a GPU."""
import json
import shlex

import numpy as np
import pytest
from PIL import Image

from vector3d.lab import human_v1
from vector3d.lab.backends import get_backend
from vector3d.lab.backends.trellis import PROFILES, TrellisBackend


def _png(path):
    a = np.zeros((16, 16, 4), np.uint8)
    a[4:12, 4:12] = 255
    Image.fromarray(a, "RGBA").save(path)
    return path


def test_registered():
    assert isinstance(get_backend("trellis"), TrellisBackend)


def test_blocked_without_environment(tmp_path):
    be = TrellisBackend(distro="no-such-distro-v3d", python="/nonexistent/python")
    rec = be.generate({"front": _png(tmp_path / "f.png")}, out_dir=tmp_path / "run")
    assert rec.status == "blocked" and rec.reason and rec.output_path is None
    assert json.loads((tmp_path / "run" / "record.json").read_text())["status"] == "blocked"
    assert not list((tmp_path / "run").glob("*.glb"))


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(ValueError):
        TrellisBackend().generate({"front": _png(tmp_path / "f.png")}, quality_profile="ultra", out_dir=tmp_path)


def test_script_quotes_windows_paths_and_passes_every_view(tmp_path):
    d = tmp_path / "with space"
    d.mkdir()
    imgs = [d / "front view.png", d / "right.png"]
    s = TrellisBackend(python="~/env/bin/python").script(imgs, d / "o.glb", d / "r.json", 42, PROFILES["quality"])
    args = shlex.split(s)
    from vector3d.lab.anatomy import wsl_path        # C:\\.. -> /mnt/c/..; POSIX paths unchanged (Linux CI)
    assert [args[i + 1] for i, a in enumerate(args) if a == "--image"] == [wsl_path(p) for p in imgs]
    assert args[args.index("--ss-steps") + 1] == "25"
    assert s.startswith("LD_LIBRARY_PATH=$(dirname ~/env/bin/python)/../lib:$LD_LIBRARY_PATH ~/env/bin/python ")


def test_trellis_frame_becomes_y_up_facing_plus_z():
    from vector3d.lab.backends.trellis_runner import to_y_up
    head_up_z = [0.0, 0.0, 0.5]           # TRELLIS: up is +Z
    front_minus_y = [0.0, -0.3, 0.0]      # TRELLIS: the front looks toward -Y
    up, front = to_y_up([head_up_z, front_minus_y])
    assert np.allclose(up, [0, 0.5, 0]) and np.allclose(front, [0, 0, 0.3])


def test_placeholder_module_raises_only_when_called():
    import sys

    from vector3d.lab.backends import trellis_runner as tr
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] == "nvdiffrast"}
    try:
        placed = tr.install_placeholders()
        if "nvdiffrast.torch" not in placed:          # a real nvdiffrast is installed here: nothing to test
            pytest.skip("nvdiffrast present")
        import nvdiffrast.torch as dr               # import succeeds, as TRELLIS and kaolin need
        ctx = dr.RasterizeCudaContext                # taking a reference succeeds (kaolin does this)
        with pytest.raises(RuntimeError, match="deliberately not installed"):
            ctx()
    finally:
        for k in [k for k in sys.modules if k.split(".")[0] == "nvdiffrast"]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_front_only_protocol_finds_either_backend_output(tmp_path):
    (tmp_path / "raw_trellis.glb").write_bytes(b"x")
    assert human_v1.protocol_file(tmp_path, "front_only").name == "raw_trellis.glb"
    (tmp_path / "raw_triposg.glb").write_bytes(b"x")
    assert human_v1.protocol_file(tmp_path, "front_only").name == "raw_triposg.glb"
    assert human_v1.protocol_file(tmp_path, "multiview_oracle").name == "A_placed.glb"
