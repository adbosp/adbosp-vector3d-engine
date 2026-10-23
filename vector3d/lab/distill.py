"""Random organic shapes for fine-tuning data: smooth unions of a few primitives.

Complements finetune.hard_surface_sdf (boxy objects). Every shape fits inside the
dataset bound (|p| < 0.75) so mesh_sdf can extract it; ground truth is exact.
"""
from __future__ import annotations

import numpy as np

from . import dataset as ds


def random_sdf(rng: np.random.Generator):
    """2-5 random primitives (sphere, capsule, cylinder, torus, rounded box), smoothly blended,
    sometimes with a carved cavity: a varied, closed, exactly known shape."""
    n = int(rng.integers(2, 6))
    parts = []
    for _ in range(n):
        kind = rng.choice(["sphere", "capsule", "cylinder", "torus", "box"])
        c = rng.uniform(-0.25, 0.25, 3)
        if kind == "sphere":
            r = rng.uniform(0.08, 0.25)
            parts.append(lambda p, c=c, r=r: ds.sd_sphere(p - c, r))
        elif kind == "capsule":
            b = c + rng.uniform(-0.3, 0.3, 3)
            r = rng.uniform(0.04, 0.12)
            parts.append(lambda p, a=c, b=b, r=r: ds.sd_capsule(p, a, b, r))
        elif kind == "cylinder":
            r, h = rng.uniform(0.05, 0.18), rng.uniform(0.08, 0.3)
            parts.append(lambda p, c=c, r=r, h=h: ds.sd_cyl_y(p - c, r, h))
        elif kind == "torus":
            R = rng.uniform(0.1, 0.22)
            r = rng.uniform(0.03, 0.06)
            parts.append(lambda p, c=c, R=R, r=r: ds.sd_torus_z(p - c, R, r))
        else:
            half = rng.uniform(0.06, 0.22, 3)
            parts.append(lambda p, c=c, half=half: ds.sd_box(p - c, half, 0.3 * half.min()))
    k = rng.uniform(0.02, 0.08)
    cavity = None
    if rng.random() < 0.25:
        cc, cr = rng.uniform(-0.15, 0.15, 3), rng.uniform(0.05, 0.12)
        cavity = lambda p: ds.sd_sphere(p - cc, cr)  # noqa: E731

    def sdf(p):
        d = parts[0](p)
        for f in parts[1:]:
            d = ds.smin(d, f(p), k)
        if cavity is not None:
            d = np.maximum(d, -cavity(p))
        return d

    return sdf
