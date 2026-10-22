"""TripoSG (VAST-AI, MIT): single image -> shape, run in its own environment.

TripoSG pins numpy==1.22.3 and needs a CUDA build of diso, so it lives in a
separate Python environment and is driven through triposg_runner.py. Paths:
  V3D_TRIPOSG_DIR      clone of https://github.com/VAST-AI-Research/TripoSG
  V3D_TRIPOSG_PYTHON   python.exe of that environment
  V3D_TRIPOSG_WEIGHTS  weights dir (default <dir>/pretrained_weights/TripoSG)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..profiles import TRIPOSG_PROFILES, get_profile
from .base import Backend, GenerationRecord, now, preserve_inputs

RUNNER = Path(__file__).with_name("triposg_runner.py")
# docs/LAB.md install location, used when the environment variables are not set
DEFAULT_REPO = Path(r"C:\ai3d\TripoSG")
DEFAULT_PYTHON = Path(r"C:\ai3d\triposg-env\Scripts\python.exe")


class TripoSGBackend(Backend):
    name = "triposg"
    model = "VAST-AI/TripoSG"
    license = "MIT"
    multiview = False

    def __init__(self, repo: str | None = None, python: str | None = None, weights: str | None = None,
                 download: bool = False, timeout: int = 1800):
        r = repo or os.environ.get("V3D_TRIPOSG_DIR") or (str(DEFAULT_REPO) if DEFAULT_REPO.is_dir() else None)
        self.repo = Path(r) if r else None
        self.python = python or os.environ.get("V3D_TRIPOSG_PYTHON") or \
            (str(DEFAULT_PYTHON) if DEFAULT_PYTHON.is_file() else sys.executable)
        w = weights or os.environ.get("V3D_TRIPOSG_WEIGHTS")
        self.weights = Path(w) if w else (self.repo / "pretrained_weights" / "TripoSG" if self.repo else None)
        self.download = download
        self.timeout = timeout

    def check(self) -> tuple[bool, str]:
        if self.repo is None:
            return False, "V3D_TRIPOSG_DIR is not set (clone VAST-AI-Research/TripoSG and point to it)"
        if not (self.repo / "triposg" / "pipelines" / "pipeline_triposg.py").is_file():
            return False, f"{self.repo} is not a TripoSG checkout"
        if not Path(self.python).exists():
            return False, f"python not found: {self.python}"
        if not (self.weights and (self.weights / "model_index.json").is_file()) and not self.download:
            return False, f"weights missing at {self.weights}; rerun with download enabled"
        return True, "ok"

    def _prepare(self, src: Path, run_dir: Path) -> tuple[Path, str]:
        """RGBA cut-out with a real alpha channel, so TripoSG skips RMBG-1.4."""
        from ...ai.segment import segment

        rgba = np.asarray(Image.open(src).convert("RGBA"))
        mask, method = segment(rgba)
        if not mask.any():
            raise ValueError("segmentation found no foreground")
        out = rgba.copy()
        out[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        path = run_dir / "input_rgba.png"
        Image.fromarray(out, "RGBA").save(path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(run_dir / "mask.png")
        return path, method

    def generate(self, reference_images, camera_metadata=None, prompt=None,
                 quality_profile: str = "balanced", *, out_dir=None, seed: int = 42) -> GenerationRecord:
        prof = get_profile(TRIPOSG_PROFILES, quality_profile)
        run_dir = Path(out_dir or f"outputs/triposg_{time.strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        rec = GenerationRecord(self.name, self.model, "unknown", self.license, "blocked", prof.name,
                               config=prof.to_dict(), seed=seed, started_at=now())
        rec.inputs = preserve_inputs(reference_images, run_dir)
        if len(rec.inputs) > 1:
            rec.warnings.append(f"TripoSG is single-image; used '{rec.inputs[0]['view']}', ignored the other "
                                f"{len(rec.inputs) - 1} view(s)")
        if camera_metadata:
            rec.warnings.append("camera_metadata ignored: TripoSG has no camera conditioning")
        if prompt:
            rec.warnings.append("prompt ignored: image-conditioned TripoSG takes no text")
        ok, why = self.check()
        if not ok:
            rec.reason = why
            rec.save(run_dir / "record.json")
            return rec

        primary = next((i for i in rec.inputs if i["view"] == "front"), rec.inputs[0])
        image, seg = self._prepare(Path(primary["copy"]), run_dir)
        rec.config["segmentation"] = seg
        out = run_dir / "raw_triposg.glb"
        result = run_dir / "runner_result.json"
        cmd = [self.python, str(RUNNER), "--repo", str(self.repo), "--weights", str(self.weights),
               "--image", str(image), "--out", str(out), "--result", str(result), "--seed", str(seed),
               "--steps", str(prof.steps), "--guidance", str(prof.guidance), "--faces", str(prof.faces),
               "--vram-fraction", str(prof.vram_fraction), "--dtype", prof.dtype]
        if self.download:
            cmd.append("--download")
        rec.config["command"] = cmd
        try:
            # progress bars and model logs are not in the console code page on Windows
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=self.timeout)
            (run_dir / "runner.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired:
            rec.status, rec.reason = "failed", f"timed out after {self.timeout}s"
            rec.save(run_dir / "record.json")
            return rec
        r = json.loads(result.read_text()) if result.exists() else {"status": "failed",
                                                                    "reason": f"runner exited {p.returncode} without a result"}
        rec.duration_s, rec.load_s = r.get("duration_s"), r.get("load_s")
        rec.peak_vram_mb, rec.peak_vram_reserved_mb = r.get("peak_vram_mb"), r.get("peak_vram_reserved_mb")
        rec.device = r.get("gpu")
        rec.model_version = f"weights:{r['weights_fingerprint']}" if r.get("weights_fingerprint") else "unknown"
        rec.config["runtime"] = {k: r.get(k) for k in ("python", "torch", "torch_cuda", "platform", "gpu_total_mb",
                                                      "raw_faces", "faces", "vertices", "vram_after_load_mb")}
        if r.get("status") == "success" and out.is_file():
            rec.status, rec.output_path = "success", str(out)
        else:
            rec.status = "blocked" if p.returncode in (2, 3) else "failed"
            rec.reason = r.get("reason", "unknown failure")
        rec.save(run_dir / "record.json")
        return rec
