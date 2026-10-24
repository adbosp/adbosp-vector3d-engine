"""Anatomy step: open replacement candidate for SAM 3D Body while that model is gated.

image -> RTMW whole-body keypoints (open-mmlab RTMPose project, Apache-2.0, run through rtmlib on
onnxruntime) -> MHR body model (Meta, Apache-2.0 code and assets) fitted by optimisation to the 2D
keypoints with the model's own joint limits, a finger interpenetration penalty and pose priors.
Runs in its own WSL2 environment (scripts/setup_anatomy_env.sh) through backends/anatomy_runner.py.

Labels: keypoints AI_GENERATED, fitted pose and mesh ESTIMATED, reprojection residuals VERIFIED (a
measured distance between the fit and the detections, nothing more). This is NOT claimed to be
equivalent to SAM 3D Body: no same-conditions benchmark has been run. Depth (MoGe-2) and face
fitting are reported as skipped. A missing environment returns blocked, never a substitute.
  V3D_ANATOMY_DISTRO  WSL distribution (default Ubuntu-24.04)
  V3D_ANATOMY_PYTHON  python of the anatomy env inside WSL (default ~/micromamba/envs/anatomy/bin/python)
  V3D_MHR_ASSETS      MHR assets dir inside WSL (default ~/ai3d/mhr/assets)
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path, PureWindowsPath

from .backends.base import invoke_runner

RUNNER = Path(__file__).parent / "backends" / "anatomy_runner.py"
LABELS = {"VERIFIED", "ESTIMATED", "AI_GENERATED"}
EXPECTED_LABELS = {"keypoints_2d": "AI_GENERATED", "fit": "ESTIMATED", "mesh": "ESTIMATED",
                   "reprojection": "VERIFIED"}


def wsl_path(p) -> str:
    """C:\\a\\b -> /mnt/c/a/b (absolute Windows paths only)."""
    w = PureWindowsPath(Path(p).resolve())
    if not w.drive:
        raise ValueError(f"not an absolute Windows path: {p}")
    return "/mnt/" + w.drive[0].lower() + "/" + "/".join(w.parts[1:])


def check_labels(result: dict) -> list[str]:
    """Problems with the result's labels: unknown labels, or any fitted output claiming VERIFIED."""
    out = []
    labels = result.get("labels", {})
    for key, want in EXPECTED_LABELS.items():
        got = labels.get(key)
        if got not in LABELS:
            out.append(f"{key}: missing or unknown label {got!r}")
        elif got != want:
            out.append(f"{key}: label {got} but must be {want}")
    for side, h in result.get("hands", {}).items():
        if h.get("label") != "ESTIMATED":
            out.append(f"hands.{side}: fitted hand must be ESTIMATED")
    return out


def finger_summary(result: dict) -> dict:
    """Per hand: whether the fitted hand has five fingers whose tips land on the detected tips.
    correct = model fingers 5, every detected fingertip matched, no finger pair closer than 1 cm.
    Only detected fingers can be checked; undetected ones are counted, not assumed correct."""
    out = {}
    for side, h in result.get("hands", {}).items():
        checked, matched = h.get("fingertips_checked", 0), h.get("fingertips_matched", 0)
        out[side] = {"fingers_detected": h.get("fingers_detected", 0), "checked": checked, "matched": matched,
                     "correct": bool(h.get("fingers_model") == 5 and checked > 0 and matched == checked
                                     and h.get("finger_pairs_closer_than_1cm", 1) == 0)}
    return out


class AnatomyStep:
    name = "anatomy_rtmw_mhr"

    def __init__(self, distro: str | None = None, python: str | None = None, assets: str | None = None):
        self.distro = distro or os.environ.get("V3D_ANATOMY_DISTRO", "Ubuntu-24.04")
        self.python = python or os.environ.get("V3D_ANATOMY_PYTHON", "~/micromamba/envs/anatomy/bin/python")
        self.assets = assets or os.environ.get("V3D_MHR_ASSETS", "~/ai3d/mhr/assets")

    def _wsl(self, script: str) -> list[str]:
        return ["wsl", "-d", self.distro, "--", "bash", "-lc", script]

    def check(self) -> tuple[bool, str]:
        test = f"test -x {self.python} && test -f {self.assets}/compact_v6_1.model && test -f {self.assets}/lod1.fbx"
        try:
            p = subprocess.run(self._wsl(test), capture_output=True, timeout=120)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, f"WSL not available: {e}"
        if p.returncode != 0:
            return False, (f"anatomy environment or MHR assets missing in WSL {self.distro} "
                           f"({self.python}, {self.assets}); run scripts/setup_anatomy_env.sh")
        return True, "ok"

    def run(self, image, out_dir, steps: int = 600, seed: int = 0, timeout: int = 1800) -> dict:
        """One invocation = one new folder; returns the runner result plus checks, or blocked/failed."""
        run_dir = Path(out_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True)
        ok, why = self.check()
        if not ok:
            return {"status": "blocked", "reason": why, "run_dir": str(run_dir)}
        result, mesh = run_dir / "result.json", run_dir / "mhr_fit.glb"
        script = (f"{self.python} {wsl_path(RUNNER)} --image {wsl_path(image)} --assets {self.assets} "
                  f"--out-dir {wsl_path(run_dir)} --result {wsl_path(result)} --steps {steps} --seed {seed}")
        rr = invoke_runner(self._wsl(script), result, mesh, timeout=timeout)
        rr["run_dir"] = str(run_dir)
        if rr["status"] == "success":
            rr["label_problems"] = check_labels(rr)
            rr["finger_summary"] = finger_summary(rr)
            if rr["label_problems"]:
                rr["status"], rr["reason"] = "failed", "label check failed: " + "; ".join(rr["label_problems"])
        return rr
