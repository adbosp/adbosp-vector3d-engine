"""Runs inside the anatomy environment: joint-limit violation of saved MHR fits (fit.npz from anatomy_runner).

Usage: python anatomy_limits.py --assets MHR_ASSETS_DIR FIT.npz [FIT.npz ...]   (prints JSON)
The fit treats MHR's MinMax limits as a soft penalty; this reports what remains violated afterwards.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("fits", nargs="+")
    a = ap.parse_args()
    import torch
    from mhr.mhr import MHR

    model = MHR.from_files(folder=Path(a.assets), device=torch.device("cpu"), lod=1)
    names = list(model.character.parameter_transform.names)
    lims = [(l.data.minmax.model_parameter_index, l.data.minmax.min, l.data.minmax.max)
            for l in model.character.parameter_limits
            if str(l.type).endswith("MinMax") and l.data.minmax.model_parameter_index < 204]
    out = {}
    for f in a.fits:
        th = np.load(f)["model_parameters"].reshape(-1)
        v = {names[p]: float(max(th[p] - hi, lo - th[p], 0.0)) for p, lo, hi in lims}
        bad = {k: x for k, x in v.items() if x > 1e-4}
        out[f] = {"count": len(lims), "violated": len(bad), "max_violation": max(v.values()),
                  "units": "radians for rotations, model units for scale/length parameters",
                  "worst": sorted(bad.items(), key=lambda kv: -kv[1])[:5]}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
