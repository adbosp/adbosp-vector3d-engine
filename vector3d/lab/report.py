"""Markdown report from measured summaries. Only numbers present in the
summary files are printed; missing runs are listed as not run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRICS = (("ref_iou_mean", "Ref IoU", True), ("heldout_iou_mean", "Held-out IoU", True),
           ("chamfer_l1", "Chamfer-L1", False), ("fscore@0.01", "F@1%", True), ("fscore@0.02", "F@2%", True),
           ("normal_consistency", "Normal cons.", True))


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def experiment_section(summary: dict) -> str:
    runs = summary["runs"]
    lines = [f"## Experiment 001: silhouette refinement on a simulated baseline",
             "", f"Baseline: **{summary['baseline']}**. Dataset `{summary['dataset']}`, "
             f"{len({r['object'] for r in runs})} objects × {len({r['seed'] for r in runs})} seeds = {len(runs)} runs. "
             f"Parameters: `{json.dumps(summary['params'])}`.", "",
             "| Object | Seed | " + " | ".join(m[1] + " before → after" for m in METRICS) + " | Gates |",
             "|---|---|" + "---|" * len(METRICS) + "---|"]
    for r in runs:
        cells = [f"{_fmt(r['before'][k])} → {_fmt(r['after'][k])}" for k, _, _ in METRICS]
        lines.append(f"| {r['object']} | {r['seed']} | " + " | ".join(cells) +
                     f" | {'accepted' if r['gates']['accepted'] else 'FAILED'} |")
    lines += ["", "**Means over all runs**", "", "| Metric | Before | After | Runs improved |", "|---|---|---|---|"]
    regressions = []
    for k, name, higher in METRICS:
        b = np.array([r["before"][k] for r in runs])
        a = np.array([r["after"][k] for r in runs])
        better = (a > b) if higher else (a < b)
        lines.append(f"| {name} | {b.mean():.4f} | {a.mean():.4f} | {int(better.sum())}/{len(runs)} |")
        regressions += [f"{r['object']} seed {r['seed']}: {name} {r['before'][k]:.4f} → {r['after'][k]:.4f}"
                        for r, ok in zip(runs, better) if not ok]
    lines += ["", "**Runs where a metric got worse**", ""] + [f"- {x}" for x in regressions or ["none"]]
    return "\n".join(lines)


def benchmark_section(summary: dict | None) -> str:
    if summary is None:
        return ("## Real foundation-model benchmark (A raw vs B refined)\n\n"
                "**Not run.** No GPU and no access to model weights in the environment that produced this report. "
                "Run `python -m vector3d lab benchmark` on the RTX 3060 to fill this section.")
    lines = [f"## Real benchmark: {summary['backend']} ({summary['profile']}, seed {summary['seed']})", "",
             "| Object | Status | Time (s) | Peak VRAM (MB) | " + " | ".join(f"{m[1]} A → B" for m in METRICS) + " |",
             "|---|---|---|---|" + "---|" * len(METRICS)]
    for r in summary["rows"]:
        rec = r["record"]
        if rec["status"] != "success":
            lines.append(f"| {r['object']} | {rec['status']}: {rec['reason']} | | | " + " | " * (len(METRICS) - 1) + " |")
            continue
        cells = [f"{_fmt(r['A'][k])} → {_fmt(r['B'][k])}" for k, _, _ in METRICS]
        lines.append(f"| {r['object']} | success | {_fmt(rec['duration_s'])} | {_fmt(rec['peak_vram_mb'])} | " +
                     " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path, experiment_summary: dict, benchmark_summary: dict | None = None,
                 audit: dict | None = None, preamble: str = "") -> str:
    parts = ["# Benchmark report", preamble]
    if audit:
        parts.append("## Environment\n\n```json\n" + json.dumps(audit, indent=1, default=str)[:3000] + "\n```")
    parts += [benchmark_section(benchmark_summary), experiment_section(experiment_summary)]
    text = "\n\n".join(p for p in parts if p) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return text
