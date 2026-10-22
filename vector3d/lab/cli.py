"""python -m vector3d lab <command>

  audit       [--python P] [-o audit.json]          hardware/software check, run first
  dataset     -o DIR                                 build the synthetic benchmark set
  generate    --image F [--backend triposg] [--profile balanced] [--seed 42] [--download] -o DIR
  render      MESH -o DIR                            gates + front/back/left/right + turntable
  experiment  --data DIR -o DIR [--seeds 0,1,2]      silhouette refinement on a simulated baseline
  benchmark   --data DIR -o DIR [--backend triposg]  real backend: A raw vs B refined
  report      --experiment DIR [--benchmark DIR] [--audit F] -o report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vector3d lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--python", help="python of the model environment, to check its torch/CUDA")
    a.add_argument("-o", "--out")
    d = sub.add_parser("dataset")
    d.add_argument("-o", "--out", required=True)
    d.add_argument("--resolution", type=int, default=112)
    g = sub.add_parser("generate")
    g.add_argument("--image", action="append", required=True, help="reference image; repeat as view=path for several")
    g.add_argument("--backend", default="triposg")
    g.add_argument("--profile", default="balanced", choices=["fast", "balanced", "quality", "experimental"])
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--download", action="store_true", help="allow downloading model weights")
    g.add_argument("--prompt")
    g.add_argument("-o", "--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("mesh")
    r.add_argument("-o", "--out", required=True)
    e = sub.add_parser("experiment")
    e.add_argument("--data", required=True)
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--seeds", default="0,1,2")
    b = sub.add_parser("benchmark")
    b.add_argument("--data", required=True)
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--backend", default="triposg")
    b.add_argument("--profile", default="balanced")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--download", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--experiment", required=True)
    rp.add_argument("--benchmark")
    rp.add_argument("--audit")
    rp.add_argument("-o", "--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        from .env_audit import audit

        rep = audit(args.python)
        text = json.dumps(rep, indent=1, default=str)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return 0
    if args.cmd == "dataset":
        from .dataset import build

        m = build(args.out, args.resolution)
        print(f"{len(m['objects'])} objects written to {args.out}")
        return 0
    if args.cmd == "generate":
        from .backends import get_backend
        from .gates import run_gates

        images = dict(x.split("=", 1) if "=" in x else ("front", x) for x in args.image)
        kw = {"download": True} if args.download else {}
        rec = get_backend(args.backend, **kw).generate(images, prompt=args.prompt, quality_profile=args.profile,
                                                      out_dir=args.out, seed=args.seed)
        print(json.dumps({k: rec.to_dict()[k] for k in ("status", "reason", "model", "model_version", "duration_s",
                                                        "peak_vram_mb", "output_path", "warnings")}, indent=1))
        if rec.status != "success":
            return 2
        g = run_gates(rec.output_path, Path(args.out) / "gates", metadata={
            "model": rec.model, "version": rec.model_version, "duration_s": rec.duration_s,
            "peak_vram_mb": rec.peak_vram_mb})
        print(json.dumps(g, indent=1, default=str))
        return 0 if g["accepted"] else 1
    if args.cmd == "render":
        from PIL import Image

        from .gates import run_gates
        from .mesh_io import load_mesh
        from .render import contact_sheet, turntable

        g = run_gates(args.mesh, args.out, metadata={"model": "user mesh", "version": Path(args.mesh).name})
        frames = turntable(load_mesh(args.mesh), 8)
        Image.fromarray(contact_sheet([f for _, f in frames], [n for n, _ in frames])).save(Path(args.out) / "turntable.png")
        print(json.dumps(g, indent=1, default=str))
        return 0 if g["accepted"] else 1
    if args.cmd == "experiment":
        from .experiments import run_silhouette_experiment

        s = run_silhouette_experiment(args.data, args.out, seeds=tuple(int(x) for x in args.seeds.split(",")))
        print(f"{len(s['runs'])} runs, summary in {Path(args.out) / 'summary.json'}")
        return 0
    if args.cmd == "benchmark":
        from .benchmark import run_benchmark

        s = run_benchmark(args.data, args.out, args.backend, args.profile, args.seed,
                          backend_kwargs={"download": True} if args.download else None)
        ok = sum(r["record"]["status"] == "success" for r in s["rows"])
        print(f"{ok}/{len(s['rows'])} objects generated, summary in {Path(args.out) / 'summary.json'}")
        return 0 if ok else 2
    if args.cmd == "report":
        from .report import write_report

        load = lambda p: json.loads(Path(p).read_text()) if p else None  # noqa: E731
        write_report(args.out, load(Path(args.experiment) / "summary.json"),
                     load(Path(args.benchmark) / "summary.json") if args.benchmark else None, load(args.audit))
        print(f"wrote {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
