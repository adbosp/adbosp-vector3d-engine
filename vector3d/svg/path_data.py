"""SVG path `d` attribute parser producing exact Bézier segments.

Supports M L H V C S Q T A Z in absolute and relative forms, implicit command
repetition, and arc flags packed without separators (e.g. `a1 1 0 01 2 2`).
Elliptical arcs are converted to cubic Béziers (max 90° per piece, error < 3e-4 of radius).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core.geometry import Contour, Segment, cubic, line, quad

_TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_PARAMS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


class _Scanner:
    def __init__(self, d: str):
        self.d = d
        self.pos = 0

    def _skip(self) -> None:
        while self.pos < len(self.d) and (self.d[self.pos].isspace() or self.d[self.pos] == ","):
            self.pos += 1

    def peek_command(self) -> str | None:
        self._skip()
        if self.pos < len(self.d) and self.d[self.pos] in "MmLlHhVvCcSsQqTtAaZz":
            return self.d[self.pos]
        return None

    def at_end(self) -> bool:
        self._skip()
        return self.pos >= len(self.d)

    def command(self) -> str:
        c = self.peek_command()
        if c is None:
            raise ValueError(f"expected path command at {self.pos} in {self.d!r}")
        self.pos += 1
        return c

    def number(self) -> float:
        self._skip()
        m = _TOKEN_RE.match(self.d, self.pos)
        if not m or m.group(0).isalpha():
            raise ValueError(f"expected number at {self.pos} in {self.d!r}")
        self.pos = m.end()
        return float(m.group(0))

    def flag(self) -> float:
        self._skip()
        ch = self.d[self.pos : self.pos + 1]
        if ch not in ("0", "1"):
            raise ValueError(f"expected arc flag at {self.pos} in {self.d!r}")
        self.pos += 1
        return float(ch)

    def has_number(self) -> bool:
        self._skip()
        if self.pos >= len(self.d):
            return False
        m = _TOKEN_RE.match(self.d, self.pos)
        return bool(m) and not m.group(0).isalpha()


def arc_to_cubics(p0, rx, ry, phi_deg, large_arc, sweep, p1) -> list[Segment]:
    """SVG endpoint arc -> cubic Béziers (SVG 1.1 implementation notes F.6)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if np.allclose(p0, p1):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [line(p0, p1)]
    phi = math.radians(phi_deg)
    cphi, sphi = math.cos(phi), math.sin(phi)
    dx, dy = (p0 - p1) / 2
    x1p = cphi * dx + sphi * dy
    y1p = -sphi * dx + cphi * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large_arc == sweep:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cphi * cxp - sphi * cyp + (p0[0] + p1[0]) / 2
    cy = sphi * cxp + cphi * cyp + (p0[1] + p1[1]) / 2

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    theta1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    n = max(1, math.ceil(abs(dtheta) / (math.pi / 2) - 1e-9))
    delta = dtheta / n
    k = 4 / 3 * math.tan(delta / 4)

    def pt(t):
        x, y = rx * math.cos(t), ry * math.sin(t)
        return np.array([cphi * x - sphi * y + cx, sphi * x + cphi * y + cy])

    def deriv(t):
        x, y = -rx * math.sin(t), ry * math.cos(t)
        return np.array([cphi * x - sphi * y, sphi * x + cphi * y])

    segs = []
    t = theta1
    start = p0
    for i in range(n):
        t2 = t + delta
        end = p1 if i == n - 1 else pt(t2)
        segs.append(cubic(start, start + k * deriv(t), end - k * deriv(t2), end))
        start, t = end, t2
    return segs


def parse_path(d: str) -> list[Contour]:
    """Parse path data into one Contour per subpath, in the element's own user space."""
    sc = _Scanner(d)
    contours: list[Contour] = []
    segs: list[Segment] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    last_ctrl: np.ndarray | None = None
    last_cmd = ""

    def flush(closed: bool) -> None:
        nonlocal segs
        if segs:
            contours.append(Contour(segments=segs, closed=closed))
        segs = []

    cmd = None
    while not sc.at_end():
        if sc.peek_command() is not None:
            cmd = sc.command()
            if last_cmd == "" and cmd not in "Mm":
                raise ValueError(f"path data must start with moveto: {d!r}")
        elif cmd is None:
            raise ValueError(f"path data must start with a command: {d!r}")
        elif cmd in "Zz":
            raise ValueError(f"numbers after Z in {d!r}")
        rel = cmd.islower()
        C = cmd.upper()
        base = cur if rel else np.zeros(2)

        if C == "Z":
            if segs and not np.allclose(cur, start):
                segs.append(line(cur, start))
            flush(closed=True)
            cur = start.copy()
            last_ctrl, last_cmd = None, "Z"
            continue

        if C == "M":
            flush(closed=False)
            cur = base + [sc.number(), sc.number()]
            start = cur.copy()
            cmd = "l" if rel else "L"  # implicit lineto for subsequent pairs
            last_ctrl, last_cmd = None, "M"
            continue

        if C == "A":
            rx, ry, phi = sc.number(), sc.number(), sc.number()
            fa, fs = sc.flag(), sc.flag()
            end = base + [sc.number(), sc.number()]
            segs.extend(arc_to_cubics(cur, rx, ry, phi, fa, fs, end))
            cur, last_ctrl = end, None
        elif C == "L":
            end = base + [sc.number(), sc.number()]
            segs.append(line(cur, end))
            cur, last_ctrl = end, None
        elif C == "H":
            x = sc.number() + (cur[0] if rel else 0)
            end = np.array([x, cur[1]])
            segs.append(line(cur, end))
            cur, last_ctrl = end, None
        elif C == "V":
            y = sc.number() + (cur[1] if rel else 0)
            end = np.array([cur[0], y])
            segs.append(line(cur, end))
            cur, last_ctrl = end, None
        elif C == "C":
            c1 = base + [sc.number(), sc.number()]
            c2 = base + [sc.number(), sc.number()]
            end = base + [sc.number(), sc.number()]
            segs.append(cubic(cur, c1, c2, end))
            cur, last_ctrl = end, c2
        elif C == "S":
            c1 = 2 * cur - last_ctrl if last_cmd in ("C", "S") and last_ctrl is not None else cur.copy()
            c2 = base + [sc.number(), sc.number()]
            end = base + [sc.number(), sc.number()]
            segs.append(cubic(cur, c1, c2, end))
            cur, last_ctrl = end, c2
        elif C == "Q":
            c1 = base + [sc.number(), sc.number()]
            end = base + [sc.number(), sc.number()]
            segs.append(quad(cur, c1, end))
            cur, last_ctrl = end, c1
        elif C == "T":
            c1 = 2 * cur - last_ctrl if last_cmd in ("Q", "T") and last_ctrl is not None else cur.copy()
            end = base + [sc.number(), sc.number()]
            segs.append(quad(cur, c1, end))
            cur, last_ctrl = end, c1
        last_cmd = C
    flush(closed=False)
    return contours
