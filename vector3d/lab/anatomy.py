"""Anatomy step: open replacement candidate for SAM 3D Body while that model is gated.

image -> RTMW whole-body keypoints (open-mmlab RTMPose project, Apache-2.0, run through rtmlib on
onnxruntime) -> MHR body model (Meta, Apache-2.0 code and assets) fitted by optimisation to the 2D
keypoints with the model's own joint limits, a finger interpenetration penalty and pose priors.
Runs in its own WSL2 environment (scripts/setup_anatomy_env.sh) through backends/anatomy_runner.py.

Labels: keypoints AI_GENERATED, fitted pose and mesh ESTIMATED, reprojection residuals VERIFIED (a
measured distance between the fit and the detections, nothing more). Joint limits are a soft penalty;
the remaining violation is reported, not assumed zero. The finger proximity check uses skeleton axis
points, not the mesh surface: it is not a collision test. This is NOT claimed to be
equivalent to SAM 3D Body: no same-conditions benchmark has been run. Depth (MoGe-2) and face
fitting are reported as skipped. A missing environment returns blocked, never a substitute.
  V3D_ANATOMY_DISTRO  WSL distribution (default Ubuntu-24.04)
  V3D_ANATOMY_PYTHON  python of the anatomy env inside WSL (default ~/micromamba/envs/anatomy/bin/python)
  V3D_MHR_ASSETS      MHR assets dir inside WSL (default ~/ai3d/mhr/assets)
"""
from __future__ import annotations

import os
import shlex
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
    """C:\\a\\b -> /mnt/c/a/b. A POSIX absolute path (engine already on Linux/WSL) is returned as is."""
    w = PureWindowsPath(str(p))
    if not w.drive:
        r = Path(p).resolve()
        if r.is_absolute() and str(r).startswith("/"):
            return str(r)
        w = PureWindowsPath(r)
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
    """Per hand, agreement between the fitted MHR fingertips and the 2D detector, with coverage.
    This is detector consistency only: it says nothing about 3D anatomical correctness, which has no
    ground truth here. `all_five_tips_consistent` needs all 5 tips detected AND matched; partial evidence
    (e.g. 1/5 checked) is reported as coverage, never as a consistent hand."""
    out = {}
    for side, h in result.get("hands", {}).items():
        checked, matched = h.get("fingertips_checked", 0), h.get("fingertips_matched", 0)
        close = h.get("finger_axis_pairs_closer_than_1cm", h.get("finger_pairs_closer_than_1cm"))
        out[side] = {"fingers_detected": h.get("fingers_detected", 0), "checked": checked, "matched": matched,
                     "coverage": checked / 5,
                     "detected_tips_consistent": bool(checked > 0 and matched == checked),
                     "all_five_tips_consistent": bool(checked == 5 and matched == 5),
                     "finger_axis_pairs_closer_than_1cm": close,
                     "anatomical_correctness": "not measured (no 3D ground truth; skeleton distance is not a "
                                               "mesh collision test)"}
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

    def script(self, image, run_dir, result, steps: int, seed: int) -> str:
        """bash command line; every path that comes from Windows is shell-quoted (spaces, quotes).
        The configured python/assets paths are left as given so that ~ and $HOME still expand."""
        q = lambda p: shlex.quote(wsl_path(p))  # noqa: E731
        return (f"{self.python} {q(RUNNER)} --image {q(image)} --assets {self.assets} "
                f"--out-dir {q(run_dir)} --result {q(result)} --steps {int(steps)} --seed {int(seed)}")

    def run(self, image, out_dir, steps: int = 600, seed: int = 0, timeout: int = 1800) -> dict:
        """One invocation = one new folder; returns the runner result plus checks, or blocked/failed."""
        run_dir = Path(out_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True)
        ok, why = self.check()
        if not ok:
            return {"status": "blocked", "reason": why, "run_dir": str(run_dir)}
        result, mesh = run_dir / "result.json", run_dir / "mhr_fit.glb"
        rr = invoke_runner(self._wsl(self.script(image, run_dir, result, steps, seed)), result, mesh, timeout=timeout)
        rr["run_dir"] = str(run_dir)
        if rr["status"] == "success":
            rr["label_problems"] = check_labels(rr)
            rr["finger_summary"] = finger_summary(rr)
            if rr["label_problems"]:
                rr["status"], rr["reason"] = "failed", "label check failed: " + "; ".join(rr["label_problems"])
        return rr
