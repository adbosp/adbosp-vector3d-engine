"""Backend interface and the record every generation call returns."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GenerationRecord:
    backend: str
    model: str
    model_version: str
    license: str
    status: str                      # success | failed | blocked
    quality_profile: str
    config: dict = field(default_factory=dict)
    seed: int | None = None
    duration_s: float | None = None  # inference only
    load_s: float | None = None      # model loading
    peak_vram_mb: float | None = None
    peak_vram_reserved_mb: float | None = None
    device: str | None = None
    output_path: str | None = None
    inputs: list[dict] = field(default_factory=list)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    label: str = "AI_GENERATED"
    started_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve_inputs(reference_images, run_dir: Path) -> list[dict]:
    """Copy every reference image into the run folder; originals are never modified."""
    items = reference_images.items() if isinstance(reference_images, dict) else \
        ((f"image{i}", p) for i, p in enumerate(reference_images))
    dest = run_dir / "inputs"
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for view, p in items:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        copy = dest / f"{view}{p.suffix.lower()}"
        shutil.copy2(p, copy)
        out.append({"view": view, "source": str(p), "copy": str(copy), "sha256": sha256(p)})
    return out


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Backend(ABC):
    name: str = ""
    model: str = ""
    license: str = ""
    multiview: bool = False

    @abstractmethod
    def check(self) -> tuple[bool, str]:
        """Whether the model can run here, and why not if it cannot."""

    @abstractmethod
    def generate(self, reference_images, camera_metadata=None, prompt=None,
                 quality_profile: str = "balanced", *, out_dir=None, seed: int = 42) -> GenerationRecord:
        """Run real inference and return the record (status blocked/failed instead of a fake mesh)."""
