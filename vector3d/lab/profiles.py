"""Quality profiles. Numbers are generation settings, not promises: a
profile is only marked as fitting a GPU after a measured run (see
research/benchmark_report.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int             # diffusion / flow steps
    guidance: float
    faces: int             # target face count after decimation, -1 = keep
    vram_fraction: float   # hard cap passed to torch.cuda.set_per_process_memory_fraction
    dtype: str = "float16"

    def to_dict(self) -> dict:
        return asdict(self)


# TripoSG defaults from scripts/inference_triposg.py: 50 steps, guidance 7.0, faces -1.
TRIPOSG_PROFILES = {
    "fast": Profile("fast", 25, 7.0, 20000, 0.90),
    "balanced": Profile("balanced", 50, 7.0, 50000, 0.90),
    "quality": Profile("quality", 50, 7.0, -1, 0.95),
    "experimental": Profile("experimental", 75, 7.0, -1, 0.95),
}


def get_profile(table: dict, name: str) -> Profile:
    try:
        return table[name.lower()]
    except KeyError:
        raise ValueError(f"unknown quality profile {name!r}; choose from {sorted(table)}") from None
