"""TRELLIS v1 (microsoft/TRELLIS-image-large, MIT): image(s) -> shape, run in its own WSL2 environment.

TRELLIS needs Linux CUDA extensions (spconv, kaolin, xformers), so it lives in the micromamba env of
scripts/setup_trellis_env.sh and is driven through trellis_runner.py via `wsl`. Geometry only: the
non-commercial nvdiffrast / Inria rasterisers are neither installed nor called (research/trellis/LICENCES.md).
Several reference views use TRELLIS's own multi-image conditioning (no camera input). Paths (inside WSL):
  V3D_TRELLIS_DISTRO   WSL distribution (default Ubuntu-24.04)
  V3D_TRELLIS_PYTHON   python of the env (default ~/micromamba/envs/trellis/bin/python)
  V3D_TRELLIS_REPO     clone of microsoft/TRELLIS (default ~/ai3d/TRELLIS)
  V3D_TRELLIS_WEIGHTS  TRELLIS-image-large (default ~/ai3d/weights/TRELLIS-image-large)
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Backend, GenerationRecord, invoke_runner, now, preserve_inputs

RUNNER = Path(__file__).with_name("trellis_runner.py")
# TRELLIS-image-large pipeline.json uses 25 steps for both stages; fast halves them
PROFILES = {"fast": 12, "balanced": 25, "quality": 25}


class TrellisBackend(Backend):
    name = "trellis"
    model = "microsoft/TRELLIS-image-large"
    license = "MIT"
    multiview = True

    def __init__(self, distro: str | None = None, python: str | None = None, repo: str | None = None,
                 weights: str | None = None, timeout: int = 1800):
        self.distro = distro or os.environ.get("V3D_TRELLIS_DISTRO", "Ubuntu-24.04")
        self.python = python or os.environ.get("V3D_TRELLIS_PYTHON", "~/micromamba/envs/trellis/bin/python")
        self.repo = repo or os.environ.get("V3D_TRELLIS_REPO", "~/ai3d/TRELLIS")
        self.weights = weights or os.environ.get("V3D_TRELLIS_WEIGHTS", "~/ai3d/weights/TRELLIS-image-large")
        self.timeout = timeout

    def _wsl(self, script: str) -> list[str]:
        return ["wsl", "-d", self.distro, "--", "bash", "-lc", script]

    def check(self) -> tuple[bool, str]:
        test = (f"test -x {self.python} && test -f {self.repo}/trellis/pipelines/trellis_image_to_3d.py "
                f"&& test -f {self.weights}/pipeline.json")
        try:
            p = subprocess.run(self._wsl(test), capture_output=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, f"WSL not available: {e}"
        if p.returncode != 0:
            return False, (f"TRELLIS environment, repo or weights missing in WSL {self.distro} "
                           f"({self.python}, {self.repo}, {self.weights}); run scripts/setup_trellis_env.sh")
        return True, "ok"

    def script(self, images: list[Path], out: Path, result: Path, seed: int, steps: int) -> str:
        from ..anatomy import wsl_path

        q = lambda p: shlex.quote(wsl_path(p))  # noqa: E731 - Windows paths may contain spaces
        imgs = " ".join(f"--image {q(i)}" for i in images)
        return (f"{self.python} {q(RUNNER)} --repo {self.repo} --weights {self.weights} {imgs} --out {q(out)} "
                f"--result {q(result)} --seed {int(seed)} --ss-steps {int(steps)} --slat-steps {int(steps)}")

    def _prepare(self, src: Path, dest: Path) -> str:
        """RGBA cut-out with a real alpha channel (engine segmenters only, MIT/Apache), so TRELLIS skips rembg."""
        from ...ai.segment import matte

        rgba = np.asarray(Image.open(src).convert("RGBA"))
        alpha, method = matte(rgba)
        if not (alpha > 127).any():
            raise ValueError(f"segmentation found no foreground in {src}")
        out = rgba.copy()
        out[..., 3] = alpha
        Image.fromarray(out, "RGBA").save(dest)
        return method

    def generate(self, reference_images, camera_metadata=None, prompt=None,
                 quality_profile: str = "balanced", *, out_dir=None, seed: int = 42) -> GenerationRecord:
        if quality_profile not in PROFILES:
            raise ValueError(f"unknown quality profile {quality_profile!r}; choose from {sorted(PROFILES)}")
        steps = PROFILES[quality_profile]
        run_dir = Path(out_dir or f"outputs/trellis_{time.strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        rec = GenerationRecord(self.name, self.model, "unknown", self.license, "blocked", quality_profile,
                               config={"steps": steps, "formats": ["mesh"]}, seed=seed, started_at=now())
        rec.inputs = preserve_inputs(reference_images, run_dir)
        if camera_metadata:
            rec.warnings.append("camera_metadata ignored: TRELLIS multi-image conditioning takes no cameras")
        if prompt:
            rec.warnings.append("prompt ignored: image-conditioned TRELLIS takes no text")
        ok, why = self.check()
        if not ok:
            rec.reason = why
            rec.save(run_dir / "record.json")
            return rec
        order = sorted(rec.inputs, key=lambda i: i["view"] != "front")      # front first
        images, segs = [], []
        for i in order:
            dest = run_dir / f"input_{i['view']}_rgba.png"
            segs.append(self._prepare(Path(i["copy"]), dest))
            images.append(dest)
        rec.config.update(segmentation=segs, views=[i["view"] for i in order],
                          conditioning="multi-image (stochastic)" if len(images) > 1 else "single image")
        out, result = run_dir / "raw_trellis.glb", run_dir / "runner_result.json"
        cmd = self._wsl(self.script(images, out, result, seed, steps))
        rec.config["command"] = cmd
        r = invoke_runner(cmd, result, out, timeout=self.timeout)
        rec.duration_s, rec.load_s = r.get("duration_s"), r.get("load_s")
        rec.peak_vram_mb, rec.peak_vram_reserved_mb = r.get("peak_vram_mb"), r.get("peak_vram_reserved_mb")
        rec.device = r.get("gpu")
        rec.model_version = "TRELLIS-image-large (pinned by setup_trellis_env.sh)"
        rec.config["runtime"] = {k: r.get(k) for k in ("python", "torch", "platform", "gpu_total_mb", "vertices",
                                                      "faces", "vram_after_load_mb", "frame")}
        if r["status"] == "success":
            rec.status, rec.output_path = "success", str(out)
        else:
            rec.status = "blocked" if r.get("returncode") in (2, 3) else "failed"
            rec.reason = r.get("reason", "unknown failure")
        rec.save(run_dir / "record.json")
        return rec
