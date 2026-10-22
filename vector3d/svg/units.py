"""SVG length parsing and unit conversion.

Only absolute physical units map to meters. `px` and unitless lengths are
screen units: they never imply a physical object size on their own.
"""
from __future__ import annotations

import re

METERS_PER_UNIT: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "q": 0.00025,
    "in": 0.0254,
    "pt": 0.0254 / 72,
    "pc": 0.0254 / 6,
}
SCREEN_UNITS = {"", "px"}
PX_PER_IN = 96.0  # CSS reference pixel, used only to relate px to other units inside a document

_LENGTH_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]*)\s*$")


def parse_length(text: str | None) -> tuple[float, str] | None:
    if text is None:
        return None
    m = _LENGTH_RE.match(text)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def to_meters(value: float, unit: str) -> float | None:
    """Physical length in meters, or None for screen/relative units."""
    factor = METERS_PER_UNIT.get(unit.lower())
    return None if factor is None else value * factor


def to_px(value: float, unit: str) -> float | None:
    """CSS px equivalent, used for viewport sizing only."""
    unit = unit.lower()
    if unit in SCREEN_UNITS:
        return value
    m = to_meters(value, unit)
    return None if m is None else m / 0.0254 * PX_PER_IN


def parse_drawing_scale(text: str) -> float:
    """'1:10' -> 10.0 (one drawing unit represents ten real units); '2:1' -> 0.5."""
    if ":" in text:
        a, b = (float(x) for x in text.split(":", 1))
        return b / a
    return float(text)
