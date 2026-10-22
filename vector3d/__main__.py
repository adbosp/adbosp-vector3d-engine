"""CLI.

  python -m vector3d inspect --front f.svg --right r.svg [--top t.svg] [--json]
  python -m vector3d ui [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import sys

from .normalize.normalizer import normalize_views
from .normalize.views import ViewName
from .report import summary as _summary
from .svg.parser import parse_svg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vector3d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("inspect", help="parse and normalize orthographic SVG views")
    for v in ViewName:
        insp.add_argument(f"--{v.value}", metavar="SVG")
    insp.add_argument("--tolerance", type=float, default=0.01)
    insp.add_argument("--json", action="store_true")
    ui = sub.add_parser("ui", help="open the local web interface")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "ui":
        from .ui.server import serve
        serve(args.host, args.port, open_browser=not args.no_browser)
        return 0

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
