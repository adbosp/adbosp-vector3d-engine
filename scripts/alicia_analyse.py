import json, re
from pathlib import Path

import numpy as np
from PIL import Image

from vector3d.ai.segment import matte
from vector3d.lab import compare_views as cv
from vector3d.lab.mesh_io import load_mesh

A = Path("bench/inputs/alicia")
O = Path("bench/compare_alicia2")
refs = {}
for v in ("right", "back"):
    a, method = matte(np.asarray(Image.open(A / f"alicia_{v}.png").convert("RGBA")))
    refs[v] = a > 127

runs = {
    "front": {"TripoSG": ("triposg_front/model.glb", "triposg_front.log", True),
              "TripoSG + 3 views": ("ai_views/model.glb", "ai_views.log", False),
              "TRELLIS": ("trellis_front/raw_trellis.glb", "trellis_front/record.json", True),
              "TRELLIS + 3 views": ("trellis_views/raw_trellis.glb", "trellis_views/record.json", False),
              "Omni": ("omni_front/model.glb", "omni_front.log", True)},
    "hoodie": {"TripoSG": ("triposg_hoodie/model.glb", "triposg_hoodie.log", True),
               "TRELLIS": ("trellis_hoodie/raw_trellis.glb", "trellis_hoodie/record.json", True),
               "Omni": ("omni_hoodie/model.glb", "omni_hoodie.log", True)},
}
walls = {l.split()[0]: int(l.split()[-2]) for l in Path("bench/compare_alicia2_batch.txt").read_text().splitlines()
         if " wall " in l} if Path("bench/compare_alicia2_batch.txt").exists() else {}


def info(path):
    p = O / path
    if p.suffix == ".json":
        r = json.loads(p.read_text())
        return {"gen_s": r.get("duration_s"), "load_s": r.get("load_s"), "peak_vram_mb": r.get("peak_vram_mb"),
                "faces": r.get("config", {}).get("runtime", {}).get("faces")}
    t = p.read_text(errors="replace")
    m = re.search(r"ai_generate: ([\d,]+) triangles in (\d+) s(?:, peak VRAM (\d+) MB)?", t)
    out = {"gen_s": int(m.group(2)) if m else None, "peak_vram_mb": int(m.group(3)) if m and m.group(3) else None,
           "raw_triangles": int(m.group(1).replace(",", "")) if m else None}
    r = re.search(r"refine: (.*)", t)
    if r:
        out["refine"] = r.group(1).strip()
    f = re.search(r"fit_views: (.*)", t)
    if f:
        out["fit_views"] = f.group(1).strip()
    return out


report = {"reference": "right/back masks by IS-Net from the held-out turnaround views (perspective renders)"}
for inp, rows in runs.items():
    meshes, table = {}, {}
    for name, (mesh, log, heldout) in rows.items():
        m = load_mesh(O / mesh)
        meshes[name] = m
        row = {**info(log), "faces_output": int(len(m.faces))}
        if inp == "front":
            row["iou"] = cv.heldout_iou(m, refs)
            row["iou_is_heldout"] = heldout
        table[name] = row
    report[inp] = table
    Image.fromarray(cv.sheet(meshes)).save(O / f"sheet_{inp}.png")
(O / "alicia_report.json").write_text(json.dumps(report, indent=1, default=str))
for inp in runs:
    for n, r in report[inp].items():
        print(inp, n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "iou"},
              {k: round(v, 3) for k, v in r.get("iou", {}).items()})
