"""Runs inside the TripoSG Python environment, never inside the engine's.

Usage (called by TripoSGBackend):
  python triposg_runner.py --repo DIR --weights DIR --image RGBA.png --out mesh.glb
      --result result.json [--seed 42 --steps 50 --guidance 7.0 --faces -1
      --vram-fraction 0.9 --download]

The image must already carry a valid alpha channel: TripoSG's prepare_image
then uses that alpha and never calls RMBG-1.4, so its licence is not
involved. Peak GPU memory comes from torch.cuda.max_memory_allocated /
max_memory_reserved. The result JSON is written whether the run succeeds or
fails. This file only imports TripoSG's own modules.
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--weights", "--image", "--out", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--faces", type=int, default=-1)
    ap.add_argument("--vram-fraction", type=float, default=0.9)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()

    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy as np
        import torch

        res.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            res["reason"] = "CUDA not available in this environment; refusing to run TripoSG on CPU"
            return _finish(a, res, 2)
        dev = torch.device("cuda")
        res["gpu"] = torch.cuda.get_device_name(0)
        res["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction, 0)
        torch.cuda.reset_peak_memory_stats()

        sys.path[:] = import_path(a.repo, os.path.dirname(os.path.abspath(__file__)), sys.path)
        from image_process import prepare_image
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        if not os.path.isfile(os.path.join(a.weights, "model_index.json")):
            if not a.download:
                res["reason"] = f"weights not found in {a.weights} (pass --download to fetch VAST-AI/TripoSG)"
                return _finish(a, res, 3)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=a.weights)
        res["weights_dir"] = a.weights
        res["model_index"] = json.load(open(os.path.join(a.weights, "model_index.json")))
        res["weights_fingerprint"] = _fingerprint(a.weights)

        t0 = time.perf_counter()
        dtype = getattr(torch, a.dtype)
        pipe = TripoSGPipeline.from_pretrained(a.weights).to(dev, dtype)
        torch.cuda.synchronize()
        res["load_s"] = time.perf_counter() - t0
        res["vram_after_load_mb"] = torch.cuda.memory_allocated() / 2**20

        img = prepare_image(a.image, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = pipe(image=img, generator=torch.Generator(device=pipe.device).manual_seed(a.seed),
                       num_inference_steps=a.steps, guidance_scale=a.guidance).samples[0]
        torch.cuda.synchronize()
        res["duration_s"] = time.perf_counter() - t1
        res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20

        import trimesh
        mesh = trimesh.Trimesh(out[0].astype(np.float32), np.ascontiguousarray(out[1]))
        res["raw_faces"] = int(len(mesh.faces))
        if a.faces > 0 and len(mesh.faces) > a.faces:
            from inference_triposg import simplify_mesh
            mesh = simplify_mesh(mesh, a.faces)
        res["faces"] = int(len(mesh.faces))
        res["vertices"] = int(len(mesh.vertices))
        mesh.export(a.out)
        res["output_path"] = a.out
        res["status"] = "success"
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


def import_path(repo: str, here: str, path: list) -> list:
    """TripoSG's repo first, and this file's folder removed: it holds the engine's triposg.py, which would shadow TripoSG's package."""
    here = os.path.normcase(os.path.abspath(here))
    kept = [p for p in path if os.path.normcase(os.path.abspath(p or os.curdir)) != here]
    return [repo, os.path.join(repo, "scripts")] + kept


def _fingerprint(root: str) -> str:
    """Short hash of every weight file's path and size: identifies the checkpoint without hashing gigabytes."""
    import hashlib

    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.endswith((".safetensors", ".bin", ".json", ".pt", ".ckpt")):
                full = os.path.join(d, f)
                h.update(f"{os.path.relpath(full, root)}:{os.path.getsize(full)}".encode())
    return h.hexdigest()[:16]


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
