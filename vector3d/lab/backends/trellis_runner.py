"""Runs inside the TRELLIS environment (scripts/setup_trellis_env.sh, WSL2/Linux), never inside the engine's.

Usage:
  python trellis_runner.py --repo DIR --weights DIR --image RGBA.png [--image more.png ...] --out mesh.glb
      --result result.json [--seed 42 --ss-steps 25 --slat-steps 25 --vram-fraction 0.95]

Geometry only: formats=['mesh'] (FlexiCubes), so no renderer, nvdiffrast or Gaussian rasteriser is used.
TRELLIS imports nvdiffrast (renderer) at package load; it is not installed (non-commercial licence) and is
replaced by a placeholder that raises on any use (NOT_INSTALLED). open3d (MIT) is installed: TRELLIS's
text pipeline evaluates it at import time.
The image must carry a real alpha channel: TRELLIS then uses it and never calls rembg. Several images use
TRELLIS's own multi-image conditioning (run_multi_image, 'stochastic' mode). The mesh is exported with
TRELLIS's own Z-up -> Y-up rotation (as in its to_glb), in its normalised [-0.5, 0.5] frame. Peak GPU
memory comes from torch.cuda.max_memory_allocated. The result JSON is written whether the run succeeds or fails.
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
import types


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--weights", "--out", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ss-steps", type=int, default=25)
    ap.add_argument("--slat-steps", type=int, default=25)
    ap.add_argument("--vram-fraction", type=float, default=0.95)
    a = ap.parse_args()
    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform(),
           "images": a.image, "seed": a.seed, "ss_steps": a.ss_steps, "slat_steps": a.slat_steps}
    try:
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        os.environ.setdefault("SPCONV_ALGO", "native")
        import numpy as np
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            res["reason"] = "CUDA not available; refusing to run TRELLIS on CPU"
            return _finish(a, res, 2)
        res.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   gpu_total_mb=torch.cuda.get_device_properties(0).total_memory / 2**20)
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
        res["not_installed_placeholders"] = install_placeholders()
        sys.path.insert(0, a.repo)
        from trellis.pipelines import TrellisImageTo3DPipeline

        if not os.path.isfile(os.path.join(a.weights, "pipeline.json")):
            res["reason"] = f"weights not found in {a.weights}"
            return _finish(a, res, 3)
        t0 = time.perf_counter()
        pipe = TrellisImageTo3DPipeline.from_pretrained(a.weights)
        pipe.cuda()
        torch.cuda.synchronize()
        res["load_s"] = time.perf_counter() - t0
        res["vram_after_load_mb"] = torch.cuda.memory_allocated() / 2**20
        images = [Image.open(p).convert("RGBA") for p in a.image]
        for p, im in zip(a.image, images):
            if np.all(np.asarray(im)[..., 3] == 255):
                res["reason"] = f"{p} has no transparent background: refusing to let TRELLIS run rembg"
                return _finish(a, res, 4)
        ss = {"steps": a.ss_steps}
        sl = {"steps": a.slat_steps}
        t1 = time.perf_counter()
        with torch.no_grad():
            if len(images) == 1:
                out = pipe.run(images[0], seed=a.seed, formats=["mesh"],
                               sparse_structure_sampler_params=ss, slat_sampler_params=sl)
            else:
                out = pipe.run_multi_image(images, seed=a.seed, formats=["mesh"], mode="stochastic",
                                           sparse_structure_sampler_params=ss, slat_sampler_params=sl)
        torch.cuda.synchronize()
        res["duration_s"] = time.perf_counter() - t1
        res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        m = out["mesh"][0]
        if not m.success:
            res["reason"] = "TRELLIS produced an empty mesh"
            return _finish(a, res, 1)
        v = to_y_up(m.vertices.detach().float().cpu().numpy())
        f = m.faces.detach().cpu().numpy()
        import trimesh

        trimesh.Trimesh(v, f, process=False).export(a.out)
        res.update(vertices=int(len(v)), faces=int(len(f)), output_path=a.out,
                   frame="TRELLIS normalised, Y up (its to_glb rotation)", status="success")
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        try:
            import torch
            if torch.cuda.is_available():
                res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        except Exception:  # noqa: BLE001
            pass
        return _finish(a, res, 1)


def to_y_up(v):
    """TRELLIS mesh frame (Z up, front toward -Y) -> engine frame (+Y up, front toward +Z): the same rotation
    as TRELLIS's own to_glb. Checked on a real run: the face looks toward +Z, the head is at +Y."""
    import numpy as np

    return np.asarray(v, np.float32) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)


NOT_INSTALLED = {
    "nvdiffrast": "NVIDIA Source Code License (non-commercial); TRELLIS imports it for its mesh renderer only",
    "nvdiffrast.torch": "see nvdiffrast",
}


class _NotInstalled(types.ModuleType):
    """Stands in for a module TRELLIS imports at load but the mesh-only image path never calls.
    Any real use raises, so nothing can silently run without it (dunder lookups by the import system pass)."""

    def __init__(self, name: str, why: str):
        super().__init__(name)
        self._why = why

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        name, why = f"{self.__name__}.{attr}", self._why

        def unavailable(*_a, **_k):       # libraries may take a reference at import (kaolin does); calling raises
            raise RuntimeError(f"{name} called, but it is deliberately not installed: {why}")

        return unavailable


def install_placeholders() -> list[str]:
    """Register placeholders for NOT_INSTALLED modules that are really absent; returns their names."""
    import importlib.machinery
    import importlib.util

    placed = []
    for name, why in NOT_INSTALLED.items():
        root = name.split(".")[0]
        if name in sys.modules or (root not in placed and importlib.util.find_spec(root) is not None):
            continue
        mod = _NotInstalled(name, why)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = mod
        if "." in name:
            setattr(sys.modules[root], name.split(".", 1)[1], mod)
        placed.append(name)
    return placed


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
