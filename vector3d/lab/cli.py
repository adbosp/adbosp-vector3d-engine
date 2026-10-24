"""python -m vector3d lab <command>

  audit       [--python P] [-o audit.json]          hardware/software check, run first
  dataset     -o DIR                                 build the synthetic benchmark set
  generate    --image F [--backend triposg] [--profile balanced] [--seed 42] [--download] -o DIR
  render      MESH -o DIR                            gates + front/back/left/right + turntable
  experiment  --data DIR -o DIR [--seeds 0,1,2]      silhouette refinement on a simulated baseline
  benchmark   --data DIR -o DIR [--backend triposg] [--lora DIR]  real backend: A raw vs B refined
  compare     BASE_DIR OTHER_DIR [-o table.md]        per-object table of two benchmark runs
  finetune-data -o DIR [--count 600]                 procedural training set for TripoSG (CPU)
  finetune    --data DIR -o DIR [--steps 1500 --rank 16]  LoRA fine-tune of TripoSG (GPU, TripoSG env)
  segbench    --data DIR -o DIR [--methods classic,isnet,birefnet] [--images DIR]   background removal
  imagine     --data DIR -o DIR [--objects a,b]     imagination loop v0 vs visual hull (CPU)
  teacher     --data HUMAN_V1 -o DIR [--size 2048] [--people a,b] [--isolation contour|discs] [--reuse-body DIR]
              Omni teacher (hand close-ups) + scores
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
    g.add_argument("--lora", help="TripoSG LoRA adapter dir (default: none, the released weights)")
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
    b.add_argument("--lora", help="TripoSG LoRA adapter dir (lab finetune output/lora)")
    cp = sub.add_parser("compare")
    cp.add_argument("base")
    cp.add_argument("other")
    cp.add_argument("-o", "--out", help="also write the table to this markdown file")
    fd = sub.add_parser("finetune-data")
    fd.add_argument("-o", "--out", required=True)
    fd.add_argument("--count", type=int, default=600)
    fd.add_argument("--seed", type=int, default=0)
    ft = sub.add_parser("finetune")
    ft.add_argument("--data", required=True)
    ft.add_argument("-o", "--out", required=True)
    ft.add_argument("--steps", type=int, default=1500)
    ft.add_argument("--rank", type=int, default=16)
    ft.add_argument("--lr", type=float, default=1e-4)
    sb = sub.add_parser("segbench")
    sb.add_argument("--data", required=True)
    sb.add_argument("-o", "--out", required=True)
    sb.add_argument("--methods", default="classic,u2net,isnet,birefnet-lite,birefnet")
    sb.add_argument("--images", help="folder of real photos: cut-outs only, no ground truth")
    im = sub.add_parser("imagine")
    im.add_argument("--data", required=True)
    im.add_argument("-o", "--out", required=True)
    im.add_argument("--objects", help="comma-separated subset")
    te = sub.add_parser("teacher")
    te.add_argument("--data", required=True, help="human_v1 folder")
    te.add_argument("-o", "--out", required=True)
    te.add_argument("--size", type=int, default=2048, help="front render resolution given to the pipeline")
    te.add_argument("--people", help="comma-separated subset")
    te.add_argument("--profile", default="quality")
    te.add_argument("--seed", type=int, default=42)
    te.add_argument("--score-only", action="store_true", help="rescore existing outputs, no generation")
    te.add_argument("--isolation", choices=("contour", "discs"), default="contour",
                    help="hand isolation in the crop (discs = v0)")
    te.add_argument("--reuse-body", help="earlier teacher bench dir whose body_placed.glb is reused")
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
        kw = {k: v for k, v in (("download", args.download or None), ("lora", args.lora)) if v}
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
                          backend_kwargs={k: v for k, v in (("download", args.download or None),
                                                            ("lora", args.lora)) if v})
        ok = sum(r["record"]["status"] == "success" for r in s["rows"])
        print(f"{ok}/{len(s['rows'])} objects generated, summary in {Path(args.out) / 'summary.json'}")
        return 0 if ok else 2
    if args.cmd == "compare":
        from .benchmark import compare_summaries

        table = compare_summaries(args.base, args.other, (Path(args.base).name, Path(args.other).name))
        print(table)
        if args.out:
            Path(args.out).write_text(table + "\n", encoding="utf-8")
        return 0
    if args.cmd == "finetune-data":
        from .finetune import build_dataset

        m = build_dataset(args.out, args.count, args.seed)
        print(f"{sum(r['status'] in ('ok', 'exists') for r in m['samples'])} samples in {args.out} "
              f"({m['seconds']:.0f} s)")
        return 0
    if args.cmd == "finetune":
        from .finetune import train

        r = train(args.data, args.out, args.steps, args.rank, args.lr)
        print(json.dumps({k: r.get(k) for k in ("status", "reason", "samples", "trainable_params", "gt_half",
                                                "train_s", "peak_vram_mb", "lora_dir")}, indent=1))
        return 0 if r.get("status") == "success" else 2
    if args.cmd == "segbench":
        from .segbench import run_segbench

        s = run_segbench(args.data, args.out, tuple(args.methods.split(",")), images=args.images)
        for r in s["rows"]:
            print(f"{r['method']:14s} " + (f"IoU {r['mean_iou']:.4f} (min {r['min_iou']:.4f}) "
                                          f"alpha err {r['mean_alpha_error']:.4f} {r['seconds_per_image']:.2f} s/img"
                                          if r["status"] == "ok" else f"FAILED {r['reason']}") + f"  [{r['licence']}]")
        return 0
    if args.cmd == "imagine":
        from .imagine import run_benchmark

        s = run_benchmark(args.data, args.out, names=args.objects.split(",") if args.objects else None)
        for r in s["rows"]:
            for tag in ("hull", "imagined"):
                e = r[tag]
                print(f"{r['object']:10s} {tag:9s} chamfer {e['chamfer_l1']:.4f} F@0.02 {e['fscore@0.02']:.3f} "
                      f"held-out IoU {e['heldout_iou_mean']:.3f} {e['seconds']:.0f} s")
        return 0
    if args.cmd == "teacher":
        from . import teacher

        out = Path(args.out)
        if not args.score_only:
            b = teacher.run_human_v1(args.data, out, teacher.triposg_generate(args.profile, args.seed),
                                     teacher.rtmw_keypoints(out / "anatomy"), args.size,
                                     args.people.split(",") if args.people else None,
                                     isolation=args.isolation, reuse_body=args.reuse_body)
            print(json.dumps({k: v["status"] for k, v in b["people"].items()}, indent=1))
        s = teacher.score(args.data, out)
        (out / "teacher_scores.json").write_text(json.dumps(s, indent=1, default=str))
        print(json.dumps({p: v["summary"] for p, v in s.items()}, indent=1))
        return 0
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
