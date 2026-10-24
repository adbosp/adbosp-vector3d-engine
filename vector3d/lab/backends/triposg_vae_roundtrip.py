"""Runs inside the TripoSG Python environment: encode a surface with TripoSG's VAE and decode it again.

Usage:
  python triposg_vae_roundtrip.py --repo DIR --weights DIR --surface surface.npz --out mesh.glb
      --result result.json [--tokens 2048 --octree-depth 9 --vram-fraction 0.9 --seed 0]

surface.npz holds points (N,3) already in TripoSG model space and unit normals (N,3).
The latent mean is decoded with the same flash decoder the pipeline uses, so the
output shows how much of the input survives the VAE alone (no DiT involved).
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback

from triposg_finetune import farthest_point_sampling
from triposg_runner import _finish, _fingerprint, import_path


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--weights", "--surface", "--out", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--octree-depth", type=int, default=9)
    ap.add_argument("--vram-fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform(),
           "tokens": a.tokens, "octree_depth": a.octree_depth}
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            res["reason"] = "CUDA not available in this environment; refusing to run the VAE on CPU"
            return _finish(a, res, 2)
        dev = torch.device("cuda")
        res.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction, 0)
        torch.cuda.reset_peak_memory_stats()

        sys.path[:] = import_path(a.repo, os.path.dirname(os.path.abspath(__file__)), sys.path)
        import triposg.models.autoencoders.autoencoder_kl_triposg as ak
        if not hasattr(ak, "fps"):                 # torch_cluster import is commented out upstream
            ak.fps = farthest_point_sampling
        from triposg.inference_utils import flash_extract_geometry
        from triposg.models.autoencoders import TripoSGVAEModel

        t0 = time.perf_counter()
        vae = TripoSGVAEModel.from_pretrained(a.weights, subfolder="vae").to(dev, torch.float16).eval()
        res["load_s"] = time.perf_counter() - t0
        res["weights_fingerprint"] = _fingerprint(os.path.join(a.weights, "vae"))

        z = np.load(a.surface)
        surf = torch.from_numpy(np.concatenate([z["points"], z["normals"]], 1)).float()[None].to(dev, torch.float16)
        res["input_points"] = int(surf.shape[1])
        t1 = time.perf_counter()
        with torch.no_grad():
            lat = vae.encode(surf, num_tokens=a.tokens, seed=a.seed).latent_dist.mean
            vae.set_flash_decoder()
            out = flash_extract_geometry(lat, vae, bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                                         octree_depth=a.octree_depth)[0]
        torch.cuda.synchronize()
        res["duration_s"] = time.perf_counter() - t1
        res["latent_shape"] = list(lat.shape)
        res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20

        import trimesh
        mesh = trimesh.Trimesh(out[0].astype(np.float32), np.ascontiguousarray(out[1]))
        res["faces"] = int(len(mesh.faces))
        mesh.export(a.out)
        res["output_path"] = a.out
        res["status"] = "success"
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        return _finish(a, res, 1)


if __name__ == "__main__":
    sys.exit(main())
