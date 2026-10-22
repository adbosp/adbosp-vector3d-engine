"""CLI: python -m vector3d inspect --front f.svg --right r.svg [--top t.svg] [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from .normalize.normalizer import normalize_views
from .normalize.views import ViewName
from .svg.parser import parse_svg


def _summary(scene) -> dict:
    out = {"units": scene.units, "views": {}, "checks": [], "warnings": scene.warnings}
    for v, nv in scene.views.items():
        box = nv.bbox()
        out["views"][v.value] = {
            "source": nv.source,
            "contours": len(nv.contours),
            "segments": sum(len(c.segments) for c in nv.contours),
            "scale": {"meters_per_unit": nv.scale.meters_per_unit, "source": nv.scale.source,
                      "confidence": nv.scale.confidence.value if nv.scale.confidence else None},
            "origin": {"source": nv.origin_source, "confidence": nv.origin_confidence.value},
            "world_extent": {"XYZ"[w]: [round(a, 6), round(b, 6)] for w, (a, b) in nv.world_extent().items()},
            "bbox_uv": None if box is None else [box[0].round(6).tolist(), box[1].round(6).tolist()],
        }
    for c in scene.checks:
        out["checks"].append({"views": [c.view_a.value, c.view_b.value], "axis": c.world_axis,
                              "extents": [round(c.extent_a, 6), round(c.extent_b, 6)],
                              "relative_error": round(c.relative_error, 6), "ok": c.ok})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vector3d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("inspect", help="parse and normalize orthographic SVG views")
    for v in ViewName:
        insp.add_argument(f"--{v.value}", metavar="SVG")
    insp.add_argument("--tolerance", type=float, default=0.01)
    insp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    docs = {v: parse_svg(getattr(args, v.value)) for v in ViewName if getattr(args, v.value)}
    if not docs:
        ap.error("give at least one view, e.g. --front front.svg")
    scene = normalize_views(docs, tolerance=args.tolerance)
    summary = _summary(scene)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        print()
        return 0
    print(f"units: {summary['units']}")
    for name, info in summary["views"].items():
        print(f"[{name}] {info['contours']} contours, scale={info['scale']['meters_per_unit']} "
              f"({info['scale']['source']}), extent={info['world_extent']}")
    for c in summary["checks"]:
        print(f"check {'/'.join(c['views'])} {c['axis']}: {'OK' if c['ok'] else 'MISMATCH'} "
              f"({c['relative_error']:.2%})")
    for w in summary["warnings"]:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
