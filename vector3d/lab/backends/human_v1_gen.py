"""Runs inside the anatomy environment (WSL2): sample synthetic people from MHR v1.0.1 (Apache-2.0) for human_v1.

Usage:
  python human_v1_gen.py --assets MHR_ASSETS_DIR --out-dir DIR --result result.json [--count 12 --seed 0 --lod 1]

Per person writes person_XX.npz: vertices (metres, engine frame: +Y up, faces +Z, right side at -X),
faces, joints (metres), joint_names, model_parameters, identity, expression. Poses are sampled inside
MHR's own parameter limits: random yaw, arm and spine motion, per-finger curl and spread, feet on the
ground. These are synthetic bodies (smooth skin, no hair or clothes), not scans of real people.
"""
import argparse
import json
import sys
import time
import traceback

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--assets", "--out-dir", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lod", type=int, default=1)
    a = ap.parse_args()
    res = {"status": "failed", "count": a.count, "seed": a.seed, "lod": a.lod, "source": "MHR v1.0.1 (Apache-2.0)"}
    try:
        from pathlib import Path

        import numpy as np
        import torch
        from mhr.mhr import MHR

        t0 = time.perf_counter()
        model = MHR.from_files(folder=Path(a.assets), device=torch.device("cpu"), lod=a.lod)
        names = list(model.character.skeleton.joint_names)
        pnames = list(model.character.parameter_transform.names)[:204]
        lim = {}
        for l in model.character.parameter_limits:
            mm = l.data.minmax
            if str(l.type).endswith("MinMax") and mm.model_parameter_index < 204:
                lim[mm.model_parameter_index] = (mm.min, mm.max)
        rng = np.random.default_rng(a.seed)
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        people = []
        for i in range(a.count):
            th = np.zeros(204, np.float32)

            def draw(k, frac):
                lo, hi = lim.get(k, (-0.3, 0.3))
                th[k] = np.clip(rng.uniform(lo, hi) * frac, lo, hi)

            for k, n in enumerate(pnames):
                if n == "root_ry":
                    th[k] = np.radians(rng.uniform(-25, 25))
                elif any(s in n for s in ("clavicle", "uparm", "elbow", "lowarm", "wrist")):
                    draw(k, 0.6)
                elif n.startswith(("spine", "neck", "head")):
                    draw(k, 0.15)
                elif any(f in n for f in FINGERS):
                    draw(k, 0.8 if n.endswith("_rz") else 0.3)       # curl joints wider than spread
            ident = (rng.normal(0, 0.7, 45)).astype(np.float32)
            expr = np.zeros(72, np.float32)
            expr[:10] = rng.normal(0, 0.4, 10)
            with torch.no_grad():
                v, st = model(torch.from_numpy(ident)[None], torch.from_numpy(th)[None], torch.from_numpy(expr)[None])
            v = v[0].numpy() / 100.0
            J = st[0, :, :3].numpy() / 100.0
            ground = v[:, 1].min()
            v[:, 1] -= ground
            J[:, 1] -= ground
            np.savez(out / f"person_{i:02d}.npz", vertices=v.astype(np.float32),
                     faces=np.asarray(model.character.mesh.faces, np.int64), joints=J.astype(np.float32),
                     joint_names=np.array(names), model_parameters=th, identity=ident, expression=expr)
            people.append({"name": f"person_{i:02d}", "height_m": float(v[:, 1].max()), "vertices": int(len(v))})
        res.update(status="success", people=people, duration_s=time.perf_counter() - t0, output=str(out / "people.json"))
        (out / "people.json").write_text(json.dumps(people, indent=1))
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        return _finish(a, res, 1)


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
