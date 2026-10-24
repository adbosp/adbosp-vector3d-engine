"""`python -m vector3d generate --mode omni` reaches generate() with the Omni options (no GPU)."""
import numpy as np
from PIL import Image

import vector3d.__main__ as cli
import vector3d.generate as gen


def test_generate_cli_passes_omni_options(tmp_path, monkeypatch):
    img = tmp_path / "person.png"
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(img)
    seen = {}

    class Result:
        glb, obj, metadata, images = b"glb", "obj", {}, {}

    def fake_generate(inputs, opts, progress=None):
        seen["opts"], seen["inputs"] = opts, inputs
        raise SystemExit(0)                      # stop before export: only the parsing is under test

    monkeypatch.setattr(gen, "generate", fake_generate)
    try:
        cli.main(["generate", "--front", str(img), "--mode", "omni", "--omni-isolation", "band",
                  "--omni-depth", "skeleton", "-o", str(tmp_path / "out")])
    except SystemExit:
        pass
    o = seen["opts"]
    assert o.mode == "omni" and o.omni_isolation == "band" and o.omni_depth == "skeleton"
    assert list(seen["inputs"]) == ["front"]
