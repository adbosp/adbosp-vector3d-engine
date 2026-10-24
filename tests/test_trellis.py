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
    assert [args[i + 1] for i, a in enumerate(args) if a == "--image"] == [
        "/mnt/" + str(p.resolve())[0].lower() + "/" + "/".join(p.resolve().parts[1:]) for p in imgs]
    assert args[args.index("--ss-steps") + 1] == "25" and args[0] == "~/env/bin/python"


def test_front_only_protocol_finds_either_backend_output(tmp_path):
    (tmp_path / "raw_trellis.glb").write_bytes(b"x")
    assert human_v1.protocol_file(tmp_path, "front_only").name == "raw_trellis.glb"
    (tmp_path / "raw_triposg.glb").write_bytes(b"x")
    assert human_v1.protocol_file(tmp_path, "front_only").name == "raw_triposg.glb"
    assert human_v1.protocol_file(tmp_path, "multiview_oracle").name == "A_placed.glb"
