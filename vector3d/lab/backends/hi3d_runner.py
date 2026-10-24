"""Runs inside the Hi3D v1.0 environment (Linux / WSL2): Hi3D's own refine_mesh.py under a VRAM cap.

Usage:
  python hi3d_runner.py --repo DIR --result result.json [--vram-fraction 0.9] -- <refine_mesh.py arguments>

Wraps the official demo script unchanged: sets torch's per-process memory
fraction first, then records wall time, peak VRAM and the exact error. The
result JSON is written whether the run succeeds, fails or runs out of memory.
"""
import argparse
import json
import os
import platform
import runpy
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--vram-fraction", type=float, default=0.9)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.rest[1:] if a.rest[:1] == ["--"] else a.rest

    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform(),
           "vram_fraction": a.vram_fraction, "args": args}
    try:
        import torch

        if not torch.cuda.is_available():
            res["reason"] = "CUDA not available; Hi3D has no CPU path"
            return _finish(a, res, 2)
        res.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   gpu_total_mb=torch.cuda.get_device_properties(0).total_memory / 2**20)
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction, 0)
        torch.cuda.reset_peak_memory_stats()
        sys.argv = [os.path.join(a.repo, "demo", "refine_mesh.py")] + args
        sys.path.insert(0, a.repo)
        t0 = time.perf_counter()
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                raise RuntimeError(f"refine_mesh.py exited with {e.code}") from e
        torch.cuda.synchronize()
        res["duration_s"] = time.perf_counter() - t0
        res["status"] = "success"
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        return _finish(a, res, 1)
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
                res["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
                _finish(a, res, 0)
        except Exception:  # noqa: BLE001
            pass


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
