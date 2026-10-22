"""Parser for the SVG `transform` attribute."""
from __future__ import annotations

import re

from ..core import affine

_FUNC_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_transform(text: str | None) -> affine.Affine:
    """Compose the transform list left to right, as the SVG spec requires."""
    m = affine.identity()
    if not text:
        return m
    for name, args_text in _FUNC_RE.findall(text):
        args = [float(a) for a in _NUM_RE.findall(args_text)]
        if name == "matrix" and len(args) == 6:
            t = affine.matrix(*args)
        elif name == "translate" and args:
            t = affine.translate(*args[:2])
        elif name == "scale" and args:
            t = affine.scale(*args[:2])
        elif name == "rotate" and args:
            t = affine.rotate(args[0], *(args[1:3] if len(args) >= 3 else ()))
        elif name == "skewX" and args:
            t = affine.skew_x(args[0])
        elif name == "skewY" and args:
            t = affine.skew_y(args[0])
        else:
            raise ValueError(f"malformed transform: {name}({args_text})")
        m = m @ t
    return m
