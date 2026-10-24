"""SAM 3D Body (Meta, SAM License, gated): one image of a person -> body, feet and hand pose + MHR mesh.

Status 2026-10-24: blocked. Ropy's access request on Hugging Face
(facebook/sam-3d-body-dinov3) awaits the authors' review, so no weights exist
on this machine and nothing has been run. This backend only reports that;
it never returns a stand-in mesh. Community re-uploads of the weights bypass
the authors' gate and must not be used.

Once access is granted, the weights come from Codex's downloader
(docs/SAM3D_BODY_DOWNLOAD_HANDOFF.md), which pins the revision and writes
download-verification.json. Inference will run in its own environment through
a subprocess runner, like TripoSG; that runner does not exist yet, so even with
weights present the status stays blocked until a real run is verified. Paths:
  V3D_SAM3DB_WEIGHTS  downloaded repo (default C:\\AI\\models\\sam-3d-body-dinov3)
  V3D_SAM3DB_DIR      clone of the official code
  V3D_SAM3DB_PYTHON   python of its separate environment
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Backend, GenerationRecord, now, preserve_inputs

DEFAULT_WEIGHTS = Path(r"C:\AI\models\sam-3d-body-dinov3")
WEIGHT_FILES = ("model.ckpt", "assets/mhr_model.pt")
VERIFICATION = "download-verification.json"
AWAITING_REVIEW = "weights not downloaded: Hugging Face access awaiting author review"
NO_RUNNER = ("weights present but inference is not integrated yet: no SAM 3D Body runner or environment "
             "has been set up and verified on this machine")


class SAM3DBodyBackend(Backend):
    name = "sam3d_body"
    model = "facebook/sam-3d-body-dinov3"
    license = "SAM License (gated)"
    multiview = False

    def __init__(self, weights: str | None = None, repo: str | None = None, python: str | None = None):
        self.weights = Path(weights or os.environ.get("V3D_SAM3DB_WEIGHTS") or DEFAULT_WEIGHTS)
        r = repo or os.environ.get("V3D_SAM3DB_DIR")
        self.repo = Path(r) if r else None
        self.python = python or os.environ.get("V3D_SAM3DB_PYTHON")

    def missing_weights(self) -> list[str]:
        need = list(WEIGHT_FILES) + [VERIFICATION]
        return [f for f in need if not (self.weights / f).is_file()]

    def check(self) -> tuple[bool, str]:
        if self.missing_weights():
            return False, AWAITING_REVIEW
        return False, NO_RUNNER

    def generate(self, reference_images, camera_metadata=None, prompt=None,
                 quality_profile: str = "default", *, out_dir=None, seed: int = 42) -> GenerationRecord:
        run_dir = Path(out_dir or f"outputs/sam3d_body_{time.strftime('%Y%m%d_%H%M%S')}")
        run_dir.mkdir(parents=True, exist_ok=True)
        rec = GenerationRecord(self.name, self.model, "unknown", self.license, "blocked", quality_profile,
                               config={"weights": str(self.weights), "missing": self.missing_weights()},
                               seed=seed, started_at=now(), label="ESTIMATED")
        rec.inputs = preserve_inputs(reference_images, run_dir)
        _, rec.reason = self.check()
        rec.save(run_dir / "record.json")
        return rec
