"""CLI.

  python -m vector3d inspect --front f.svg --right r.svg [--top t.svg] [--json]
  python -m vector3d generate --front photo.jpg [--right r.png ...] [--mode auto|drawing|photo] [--hints h.json] [-o out/]
  python -m vector3d describe --front f.png [...]        JSON facts for AI agents
  python -m vector3d ui [--port 8765]
  python -m vector3d lab <audit|dataset|generate|render|experiment|benchmark|report>   AI 3D lab
"""
from __future__ import annotations

import argparse
import json
import sys

from .normalize.normalizer import normalize_views
from .normalize.views import ViewName
from .report import summary as _summary


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["lab"]:
        from .lab.cli import main as lab_main
        return lab_main(argv[1:])
    ap = argparse.ArgumentParser(prog="vector3d")
    sub = ap.add_subparsers(dest="cmd", required=True)
    insp = sub.add_parser("inspect", help="parse and normalize orthographic SVG views")
    for v in ViewName:
        insp.add_argument(f"--{v.value}", metavar="SVG")
    insp.add_argument("--tolerance", type=float, default=0.01)
    insp.add_argument("--json", action="store_true")
    gen = sub.add_parser("generate", help="build a mesh from the views and export GLB/OBJ/metadata")
    for v in ViewName:
        gen.add_argument(f"--{v.value}", metavar="FILE", help="SVG, PNG or JPG")
    gen.add_argument("--hints", metavar="JSON", help="geometry hints written by an agent (docs/AI_PLAYBOOK.md)")
    gen.add_argument("--resolution", type=int, default=128)
    gen.add_argument("--smooth", type=int, default=10, help="Taubin smoothing iterations (0 = off)")
    gen.add_argument("-o", "--out", default="out", help="output directory")
    gen.add_argument("--mode", choices=["auto", "ai", "drawing", "photo"], default="auto",
                     help="auto: ai (TripoSG) when installed, else photo; drawing for SVG")
    gen.add_argument("--ai-profile", choices=["fast", "balanced", "quality"], default="balanced")
    gen.add_argument("--depth-backend", choices=["auto", "depth-anything", "inflation"], default="auto")
    gen.add_argument("--thickness", type=float, default=0.6, help="photo: depth / width")
    gen.add_argument("--detail", type=float, default=0.7, help="photo: 0 = silhouette only, 1 = AI depth")
    gen.add_argument("--back-side", choices=["symmetric", "flat"], default="symmetric", help="photo: unseen back side")
    gen.add_argument("--size", type=float, default=1.0, help="photo: largest dimension in metres")
    desc = sub.add_parser("describe", help="print JSON facts about the views for AI agents")
    for v in ViewName:
        desc.add_argument(f"--{v.value}", metavar="FILE", help="SVG, PNG or JPG")
    desc.add_argument("--resolution", type=int, default=96)
    ui = sub.add_parser("ui", help="open the local web interface")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    sub.add_parser("lab", help="AI 3D lab: audit, dataset, generate, render, experiment, benchmark, report")
    args = ap.parse_args(argv)

    if args.cmd == "ui":
        from .ui.server import serve
        serve(args.host, args.port, open_browser=not args.no_browser)
        return 0

    if args.cmd == "generate":
        return _generate(args, ap)

    docs = {v: _load(v.value, getattr(args, v.value)) for v in ViewName if getattr(args, v.value)}
    if not docs:
        ap.error("give at least one view, e.g. --front front.svg")
    if args.cmd == "describe":
        from .agent import describe

        json.dump(describe(normalize_views(docs), resolution=args.resolution), sys.stdout, indent=1)
        print()
        return 0
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


def _load(view: str, path: str):
    from pathlib import Path

    from .generate import load_view

    return load_view(view, Path(path).name, Path(path).read_bytes())


def _generate(args, ap) -> int:
    from pathlib import Path

    from .generate import GenerateOptions, generate
    from .reconstruct.hints import load_hints

    inputs = {}
    for v in ViewName:
        path = getattr(args, v.value)
        if path:
            inputs[v.value] = (Path(path).name, Path(path).read_bytes())
    if not inputs:
        ap.error("give at least one view, e.g. --front front.png")
    opts = GenerateOptions(resolution=args.resolution, smooth_iterations=args.smooth, hints=load_hints(args.hints),
                           mode=args.mode, depth_backend=args.depth_backend, thickness=args.thickness,
                           detail=args.detail, back=args.back_side, size_m=args.size, ai_profile=args.ai_profile)

    def show(step: str, status: str, detail: str) -> None:
        if status != "running":
            print(f"[{status:>7}] {step}: {detail}")

    result = generate(inputs, opts, show)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.glb").write_bytes(result.glb)
    (out / "model.obj").write_text(result.obj, encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    for name, png in result.images.items():
        (out / name).write_bytes(png)
    print(f"wrote model.glb, model.obj, metadata.json{''.join(', ' + n for n in result.images)} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
