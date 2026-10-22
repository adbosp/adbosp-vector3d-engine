"""SVG geometry parser.

Reads one orthographic view drawing and returns exact contours in the root
user coordinate system (viewBox units, y pointing down), with every nested
transform applied. Physical scale is resolved separately and is allowed to be
unknown: SVG/pixel coordinates are never treated as physical sizes by default.

Engine-specific attributes (all optional) use the `data-v3d-` prefix:
  root  data-v3d-view="front|back|left|right|top|bottom"
  root  data-v3d-scale="0.001"            meters per user unit
  root  data-v3d-drawing-scale="1:10"     drawing scale applied on top of document units
  any   data-v3d-role="geometry|hidden|annotation|construction|dimension|origin"
  any   data-v3d-dimension="250mm"        real length of this element (line or path end-to-end)
  any   data-v3d-part="wheel_front_left"  semantic id, inherited by children
"""
from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import affine
from ..core.confidence import Confidence
from ..core.geometry import Contour, bbox_of, line
from .path_data import arc_to_cubics, parse_path
from .transform import parse_transform
from .units import parse_drawing_scale, parse_length, to_meters

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
INKSCAPE_GROUPMODE = "{http://www.inkscape.org/namespaces/inkscape}groupmode"

SKIP_TAGS = {"defs", "clipPath", "mask", "symbol", "marker", "pattern", "metadata", "title", "desc", "style", "script"}
SHAPE_TAGS = {"path", "line", "polyline", "polygon", "rect", "circle", "ellipse"}
NON_GEOMETRY_ROLES = {"annotation", "construction", "dimension", "origin"}
INHERITED_SEMANTICS = ("data-v3d-part", "data-v3d-role")

_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass
class ScaleInfo:
    meters_per_unit: float | None
    source: str  # explicit | dimension | document_units | unknown | propagated
    confidence: Confidence | None
    notes: list[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.meters_per_unit is not None


@dataclass
class Annotation:
    kind: str  # text | dimension
    element_id: str | None
    position: np.ndarray
    text: str | None = None
    value_m: float | None = None          # dimension: declared real length
    measured_units: float | None = None   # dimension: length in user units
    contour: Contour | None = None


@dataclass
class SvgDocument:
    source: str
    view_hint: str | None
    view_box: tuple[float, float, float, float]
    contours: list[Contour]
    construction: list[Contour]
    annotations: list[Annotation]
    origin: np.ndarray | None
    scale: ScaleInfo
    warnings: list[str]

    def bbox(self):
        return bbox_of(self.contours)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _f(el: ET.Element, name: str, default: float = 0.0) -> float:
    v = parse_length(el.get(name))
    return default if v is None else v[0]


def _style(el: ET.Element) -> dict[str, str]:
    out = {}
    for item in (el.get("style") or "").split(";"):
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
    for k in ("display", "visibility", "stroke-dasharray"):
        if el.get(k) is not None and k not in out:
            out[k] = el.get(k)
    return out


def _points(text: str | None) -> np.ndarray:
    nums = [float(n) for n in _NUM_RE.findall(text or "")]
    return np.array(nums[: len(nums) // 2 * 2], dtype=float).reshape(-1, 2)


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> list:
    p = lambda a: (cx + rx * math.cos(a), cy + ry * math.sin(a))  # noqa: E731
    segs = arc_to_cubics(p(0), rx, ry, 0, 0, 1, p(math.pi))
    segs += arc_to_cubics(p(math.pi), rx, ry, 0, 0, 1, p(0))
    return segs


def _shape_contours(el: ET.Element, tag: str) -> list[Contour]:
    if tag == "path":
        return parse_path(el.get("d") or "")
    if tag == "line":
        p0 = (_f(el, "x1"), _f(el, "y1"))
        p1 = (_f(el, "x2"), _f(el, "y2"))
        return [Contour([line(p0, p1)], closed=False)]
    if tag in ("polyline", "polygon"):
        pts = _points(el.get("points"))
        if len(pts) < 2:
            return []
        segs = [line(a, b) for a, b in zip(pts[:-1], pts[1:])]
        closed = tag == "polygon"
        if closed and not np.allclose(pts[0], pts[-1]):
            segs.append(line(pts[-1], pts[0]))
        return [Contour(segs, closed=closed)]
    if tag == "rect":
        x, y, w, h = _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height")
        if w <= 0 or h <= 0:
            return []
        rx_attr, ry_attr = el.get("rx"), el.get("ry")
        rx = _f(el, "rx") if rx_attr else (_f(el, "ry") if ry_attr else 0.0)
        ry = _f(el, "ry") if ry_attr else rx
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx <= 0 or ry <= 0:
            d = f"M{x},{y} H{x + w} V{y + h} H{x} Z"
        else:
            d = (f"M{x + rx},{y} H{x + w - rx} A{rx},{ry} 0 0 1 {x + w},{y + ry} V{y + h - ry} "
                 f"A{rx},{ry} 0 0 1 {x + w - rx},{y + h} H{x + rx} A{rx},{ry} 0 0 1 {x},{y + h - ry} "
                 f"V{y + ry} A{rx},{ry} 0 0 1 {x + rx},{y} Z")
        cs = parse_path(d)
        for c in cs:
            c.primitive = {"type": "rect", "x": x, "y": y, "width": w, "height": h, "rx": rx, "ry": ry}
        return cs
    if tag in ("circle", "ellipse"):
        cx, cy = _f(el, "cx"), _f(el, "cy")
        if tag == "circle":
            rx = ry = _f(el, "r")
        else:
            rx, ry = _f(el, "rx"), _f(el, "ry")
        if rx <= 0 or ry <= 0:
            return []
        prim = {"type": tag, "cx": cx, "cy": cy, "rx": rx, "ry": ry}
        return [Contour(_ellipse(cx, cy, rx, ry), closed=True, primitive=prim)]
    return []


def _viewport_mapping(root: ET.Element, warnings: list[str]):
    """Return (viewBox, user->viewport affine, meters per user unit from document units or None)."""
    vb_nums = [float(n) for n in _NUM_RE.findall(root.get("viewBox") or "")]
    width, height = parse_length(root.get("width")), parse_length(root.get("height"))
    if len(vb_nums) == 4 and vb_nums[2] > 0 and vb_nums[3] > 0:
        vb = tuple(vb_nums)
    else:
        if root.get("viewBox"):
            warnings.append("invalid viewBox ignored")
        vb = (0.0, 0.0, width[0] if width else 0.0, height[0] if height else 0.0)

    if not (width and height) or width[1] == "%" or height[1] == "%":
        return vb, affine.translate(-vb[0], -vb[1]), None
    if width[1] != height[1]:
        warnings.append(f"width/height use different units ({width[1]}, {height[1]})")
    sx, sy = width[0] / vb[2] if vb[2] else 1.0, height[0] / vb[3] if vb[3] else 1.0
    par = (root.get("preserveAspectRatio") or "xMidYMid meet").split()
    if par[0] == "none":
        if not math.isclose(sx, sy, rel_tol=1e-9):
            warnings.append("preserveAspectRatio=none with non-uniform scale; geometry is stretched")
    else:
        sx = sy = (max if len(par) > 1 and par[1] == "slice" else min)(sx, sy)
    m = affine.scale(sx, sy) @ affine.translate(-vb[0], -vb[1])
    mpu = None
    wm, hm = to_meters(1.0, width[1]), to_meters(1.0, height[1])
    if wm is not None and hm is not None:
        mpu = (sx * wm, sy * hm)
    return vb, m, mpu


class _Walker:
    def __init__(self, root: ET.Element, warnings: list[str]):
        self.warnings = warnings
        self.ids = {el.get("id"): el for el in root.iter() if el.get("id")}
        self.contours: list[Contour] = []
        self.construction: list[Contour] = []
        self.annotations: list[Annotation] = []
        self.origin: np.ndarray | None = None

    def walk(self, el: ET.Element, m, layers: tuple[str, ...], sem: dict[str, str], depth: int = 0) -> None:
        if depth > 64:
            self.warnings.append("element nesting too deep; truncated")
            return
        tag = _local(el.tag)
        style = _style(el)
        if style.get("display") == "none" or style.get("visibility") == "hidden":
            return
        m = m @ parse_transform(el.get("transform"))
        sem = dict(sem)
        for k in INHERITED_SEMANTICS:
            if el.get(k):
                sem[k] = el.get(k)

        if tag in ("g", "svg", "a", "switch"):
            if tag == "svg" and depth > 0:
                m = m @ affine.translate(_f(el, "x"), _f(el, "y"))
                if el.get("viewBox"):
                    self.warnings.append("nested <svg> viewBox is not applied")
            label = el.get(INKSCAPE_LABEL) or el.get("id")
            child_layers = layers + (label,) if (label and depth > 0) else layers
            for child in el:
                self.walk(child, m, child_layers, sem, depth + 1)
        elif tag == "use":
            ref = el.get(f"{{{XLINK_NS}}}href") or el.get("href") or ""
            target = self.ids.get(ref.lstrip("#"))
            if target is None:
                self.warnings.append(f"<use> target not found: {ref}")
                return
            m = m @ affine.translate(_f(el, "x"), _f(el, "y"))
            if _local(target.tag) == "symbol":
                for child in target:
                    self.walk(child, m, layers, sem, depth + 1)
            else:
                self.walk(target, m, layers, sem, depth + 1)
        elif tag in SHAPE_TAGS:
            self._shape(el, tag, m, layers, sem, style)
        elif tag == "text":
            pos = affine.apply(m, [[_f(el, "x"), _f(el, "y")]])[0]
            text = "".join(el.itertext()).strip()
            self.annotations.append(Annotation("text", el.get("id"), pos, text=text))
        elif tag in SKIP_TAGS:
            return
        else:
            for child in el:
                self.walk(child, m, layers, sem, depth + 1)

    def _shape(self, el, tag, m, layers, sem, style) -> None:
        try:
            raw = _shape_contours(el, tag)
        except ValueError as exc:
            self.warnings.append(f"skipped <{tag} id={el.get('id')}>: {exc}")
            return
        semantic = dict(sem)
        semantic.update({k: v for k, v in el.attrib.items() if k.startswith("data-v3d-")})
        if el.get("class"):
            semantic["class"] = el.get("class")
        dash = style.get("stroke-dasharray")
        if dash and dash != "none":
            semantic.setdefault("line_style", "dashed")
        role = semantic.get("data-v3d-role", "geometry")
        if el.get("id") == "v3d-origin":
            role = "origin"
        contours = []
        for c in raw:
            c = c.transformed(m)
            c.element_id, c.tag, c.layer_path, c.semantic = el.get("id"), tag, layers, semantic
            contours.append(c)
        if not contours:
            return

        if el.get("data-v3d-dimension"):
            self._dimension(el, contours)
        if role == "origin":
            lo, hi = bbox_of(contours)
            self.origin = (lo + hi) / 2
        elif role in NON_GEOMETRY_ROLES:
            self.construction.extend(contours)
        else:
            self.contours.extend(contours)

    def _dimension(self, el, contours) -> None:
        val = parse_length(el.get("data-v3d-dimension"))
        meters = None if val is None else (val[0] if val[1] == "" else to_meters(*val))
        if meters is None or meters <= 0:
            self.warnings.append(f"unreadable data-v3d-dimension on {el.get('id')}: {el.get('data-v3d-dimension')}")
            return
        pts = contours[0].anchor_points
        measured = float(np.linalg.norm(pts[-1] - pts[0]))
        if measured <= 0:
            self.warnings.append(f"dimension element {el.get('id')} has zero length")
            return
        self.annotations.append(Annotation("dimension", el.get("id"), (pts[0] + pts[-1]) / 2,
                                           value_m=meters, measured_units=measured, contour=contours[0]))


def _resolve_scale(root, annotations, doc_mpu, warnings) -> ScaleInfo:
    candidates: list[ScaleInfo] = []
    explicit = root.get("data-v3d-scale")
    if explicit:
        try:
            candidates.append(ScaleInfo(float(explicit), "explicit", Confidence.VERIFIED))
        except ValueError:
            warnings.append(f"invalid data-v3d-scale: {explicit}")

    dims = [a.value_m / a.measured_units for a in annotations if a.kind == "dimension"]
    if dims:
        med = statistics.median(dims)
        spread = max(abs(d - med) / med for d in dims)
        conf = Confidence.VERIFIED if spread < 0.005 else Confidence.ESTIMATED
        info = ScaleInfo(med, "dimension", conf, [f"{len(dims)} dimension(s), max deviation {spread:.2%}"])
        if conf is Confidence.ESTIMATED:
            warnings.append(f"dimension annotations disagree by up to {spread:.2%}")
        candidates.append(info)

    if doc_mpu is not None:
        mx, my = doc_mpu
        ds_text = root.get("data-v3d-drawing-scale")
        ds = parse_drawing_scale(ds_text) if ds_text else 1.0
        conf = Confidence.VERIFIED if ds_text else Confidence.ESTIMATED
        notes = [] if ds_text else ["physical document units with no drawing scale; assumed 1:1"]
        if not math.isclose(mx, my, rel_tol=1e-9):
            notes.append("anisotropic document scale; using horizontal factor")
        candidates.append(ScaleInfo(mx * ds, "document_units", conf, notes))

    if not candidates:
        return ScaleInfo(None, "unknown", None, ["no physical units, dimensions or explicit scale"])
    best = candidates[0]
    for other in candidates[1:]:
        if not math.isclose(best.meters_per_unit, other.meters_per_unit, rel_tol=0.005):
            warnings.append(f"scale from {best.source} ({best.meters_per_unit:g} m/unit) disagrees with "
                            f"{other.source} ({other.meters_per_unit:g} m/unit); using {best.source}")
    return best


def parse_svg(source: str | Path, *, is_text: bool = False) -> SvgDocument:
    """Parse an SVG file (or SVG text with is_text=True)."""
    text = source if is_text else Path(source).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    if _local(root.tag) != "svg":
        raise ValueError("root element is not <svg>")
    warnings: list[str] = []
    vb, _viewport, doc_mpu = _viewport_mapping(root, warnings)
    walker = _Walker(root, warnings)
    walker.walk(root, affine.identity(), (), {})
    scale = _resolve_scale(root, walker.annotations, doc_mpu, warnings)
    return SvgDocument(
        source="<text>" if is_text else str(source),
        view_hint=root.get("data-v3d-view"),
        view_box=vb,
        contours=walker.contours,
        construction=walker.construction,
        annotations=walker.annotations,
        origin=walker.origin,
        scale=scale,
        warnings=warnings,
    )
