"""Runs inside the anatomy environment (scripts/setup_anatomy_env.sh, WSL2/Linux): image -> RTMW whole-body
keypoints -> MHR body fitted to them. Never imports the engine.

Usage:
  python anatomy_runner.py --image IMG --assets MHR_ASSETS_DIR --out-dir DIR --result result.json
      [--steps 600 --seed 0 --kp-thr 0.3]

Labels in the result: keypoints AI_GENERATED (detector output), fitted pose/mesh ESTIMATED, reprojection
residuals VERIFIED (measured between the fit and the detections, nothing more). Face keypoints are
detected but not fitted (no face landmark mapping yet: reported as skipped). Depth (MoGe-2) is not used:
reported as skipped. The result JSON is written whether the run succeeds or fails.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import traceback

# COCO-WholeBody 133 (body 0-16, feet 17-22, face 23-90, left hand 91-111, right hand 112-132) -> MHR joint.
# Hand order per side: wrist, thumb CMC/MCP/IP/tip, then index, middle, ring, pinky as MCP/PIP/DIP/tip.
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
BODY = {5: "l_uparm", 6: "r_uparm", 7: "l_lowarm", 8: "r_lowarm", 9: "l_wrist", 10: "r_wrist",
        11: "l_upleg", 12: "r_upleg", 13: "l_lowleg", 14: "r_lowleg", 15: "l_talocrural", 16: "r_talocrural",
        17: "l_ball", 20: "r_ball"}


def hand_map(side: str, base: int) -> dict:
    m = {base: f"{side}_wrist"}
    for f, name in enumerate(FINGERS):
        joints = (["thumb1", "thumb2", "thumb3", "thumb_null"] if name == "thumb"
                  else [f"{name}1", f"{name}2", f"{name}3", f"{name}_null"])
        for k, j in enumerate(joints):
            m[base + 1 + 4 * f + k] = f"{side}_{j}"
    return m


KP_TO_JOINT = {**BODY, **hand_map("l", 91), **hand_map("r", 112)}


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--image", "--assets", "--out-dir", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kp-thr", type=float, default=0.3)
    ap.add_argument("--no-assume-standing", dest="assume_standing", action="store_false")
    a = ap.parse_args()
    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform(),
           "labels": {"keypoints_2d": "AI_GENERATED", "fit": "ESTIMATED", "mesh": "ESTIMATED",
                      "reprojection": "VERIFIED"},
           "skipped": {"face_fit": "no face-landmark to MHR mapping yet",
                       "depth_constraint": "MoGe-2 not installed in the anatomy env"}}
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        ort.preload_dlls()
        import torch
        from rtmlib import Wholebody

        img = cv2.imread(a.image)
        if img is None:
            res["reason"] = f"cannot read image {a.image}"
            return _finish(a, res, 3)
        H, W = img.shape[:2]
        res["image_size"] = [W, H]

        # --- RTMW whole-body keypoints (GPU through onnxruntime) ---------------------------------
        used0 = gpu_used_mb()
        t0 = time.perf_counter()
        wb = Wholebody(mode="performance", backend="onnxruntime", device="cuda", to_openpose=False)
        res["rtmw_load_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        kps, scores = wb(img)
        res["rtmw_s"] = time.perf_counter() - t1
        used1 = gpu_used_mb()
        res["rtmw_gpu_used_delta_mb"] = (used1 - used0) if used0 is not None and used1 is not None else None
        res["rtmw_providers"] = [s.get_providers() for s in _sessions(wb)]
        kps, scores = np.asarray(kps, float), np.asarray(scores, float)
        if kps.ndim != 3 or len(kps) == 0:
            res["reason"] = "no person detected"
            return _finish(a, res, 4)
        best = int(np.argmax(scores.mean(1)))
        kp, sc = kps[best], scores[best]
        res["persons_detected"] = int(len(kps))
        res["keypoints_2d"] = {"xy": kp.tolist(), "score": sc.tolist(), "label": "AI_GENERATED"}
        del wb

        # --- MHR fit ---------------------------------------------------------------------------
        from pathlib import Path
        from mhr.mhr import MHR

        torch.manual_seed(a.seed)
        t2 = time.perf_counter()
        model = MHR.from_files(folder=Path(a.assets), device=torch.device("cpu"), lod=1)
        names = list(model.character.skeleton.joint_names)
        pnames = list(model.character.parameter_transform.names)[:204]
        res["mhr_load_s"] = time.perf_counter() - t2

        idx = [(k, names.index(j)) for k, j in KP_TO_JOINT.items() if sc[k] >= a.kp_thr]
        if len(idx) < 6:
            res["reason"] = f"only {len(idx)} mapped keypoints above {a.kp_thr}"
            return _finish(a, res, 5)
        kp_i = torch.tensor([k for k, _ in idx])
        jt_i = torch.tensor([j for _, j in idx])
        target = torch.tensor(kp[kp_i], dtype=torch.float32)
        w = torch.tensor(sc[kp_i], dtype=torch.float32) ** 2
        hand_kp = torch.tensor([k >= 91 for k, _ in idx])
        vis = kp[sc >= a.kp_thr]
        size = float(max(np.ptp(vis[:, 0]), np.ptp(vis[:, 1]), 1.0))   # person extent in pixels

        # all pose and proportion parameters are free except root translation (the camera shift handles it);
        # free bone lengths let short (e.g. stylised) legs be matched by length rather than by leaning in Z
        pose_mask = torch.tensor([n not in ("root_tx", "root_ty", "root_tz") for n in pnames])
        proportion = torch.tensor([n.startswith("scale_") or n.endswith("_flexible") and "length" in n
                                   or n in ("shoulder_width_flexible", "hip_width_flexible") for n in pnames])
        lims = [(l.data.minmax.model_parameter_index, l.data.minmax.min, l.data.minmax.max)
                for l in model.character.parameter_limits
                if str(l.type).endswith("MinMax") and l.data.minmax.model_parameter_index < 204]
        res["joint_limits_used"] = len(lims)
        lim_i = torch.tensor([p for p, _, _ in lims])
        lim_lo = torch.tensor([lo for _, lo, _ in lims])
        lim_hi = torch.tensor([hi for _, _, hi in lims])
        # Without depth, a 2D fit can lean the whole body in Z for free; priors keep it upright and near rest.
        finger = torch.tensor([any(f in n for f in FINGERS) for n in pnames])
        tilt = torch.tensor([n in ("root_rx", "root_rz") for n in pnames])
        # the data term is in (fraction of person size)^2, about 1e-3 for a 3 % miss: priors are scaled to that
        prior_w = torch.where(finger, torch.tensor(1e-6), torch.tensor(1e-5))
        prior_w = torch.where(tilt, torch.tensor(1e-3), prior_w)
        prior_w = torch.where(proportion, torch.tensor(1e-4), prior_w)
        # ASSUMPTION (not measured): a standing person keeps spine, hips, knees and ankles near one depth plane.
        # Without a depth signal this is what stops the legs from swinging toward the camera.
        planar = torch.tensor([names.index(j) for j in ("c_spine0", "c_spine3", "c_neck", "l_upleg", "r_upleg",
                                                         "l_lowleg", "r_lowleg", "l_talocrural", "r_talocrural")])
        res["assumptions"] = ["standing: torso and leg joints kept near one depth plane (weight 1e-6 per cm^2)"
                              if a.assume_standing else "none"]
        res["priors"] = {"root_tilt": 1e-3, "body_pose": 1e-5, "fingers": 1e-6, "proportions": 1e-4, "identity": 1e-5,
                         "finger_penetration": 1e-3, "finger_penetration_margin_cm": 1.2}
        chains = {s: [[names.index(f"{s}_{j}") for j in (["thumb1", "thumb2", "thumb3", "thumb_null"] if f == "thumb"
                       else [f"{f}1", f"{f}2", f"{f}3", f"{f}_null"])] for f in FINGERS] for s in ("l", "r")}

        def finger_points(J, s):
            pts = []
            for ch in chains[s]:
                p = J[ch]
                t = torch.linspace(0, 1, 4)[:, None, None]
                pts.append((p[:-1][None] * (1 - t) + p[1:][None] * t).reshape(-1, 3))
            return pts

        def penetration(J, margin=1.2):
            pen = J.new_zeros(())
            for s in ("l", "r"):
                pts = finger_points(J, s)
                for i in range(5):
                    for k in range(i + 1, 5):
                        d = torch.cdist(pts[i][1:], pts[k][1:]).min()   # skip the shared palm base point
                        pen = pen + torch.relu(margin - d) ** 2
            return pen

        def run_fit(root_ry):
            params = torch.zeros(1, 204)
            params[0, pnames.index("root_ry")] = root_ry
            theta = params.clone().requires_grad_(True)
            ident = torch.zeros(1, 45, requires_grad=True)
            with torch.no_grad():
                _, st = model(ident, theta, None)
                J = st[0, :, :3]
                px = torch.stack([J[jt_i, 0], -J[jt_i, 1]], 1)
                s0 = (target - target.mean(0)).norm(dim=1).mean() / (px - px.mean(0)).norm(dim=1).mean()
            scale = s0.clone().log().requires_grad_(True)
            shift = (target.mean(0) - s0 * px.mean(0)).clone().requires_grad_(True)
            opt = torch.optim.Adam([theta, ident, scale, shift], lr=0.03)
            hist = []
            for step in range(a.steps):
                hand_w = 0.0 if step < a.steps // 3 else 1.0          # body first, then fingers
                opt.zero_grad()
                _, st = model(ident, theta * pose_mask, None)
                J = st[0, :, :3]
                proj = torch.stack([J[jt_i, 0], -J[jt_i, 1]], 1) * scale.exp() + shift
                err = ((proj - target) ** 2).sum(1) / size ** 2
                ww = torch.where(hand_kp, w * hand_w, w)
                data = (ww * err).sum() / ww.sum().clamp_min(1e-6)
                th = theta[0]
                limit = (torch.relu(th[lim_i] - lim_hi) ** 2 + torch.relu(lim_lo - th[lim_i]) ** 2).sum()
                rest = th - params[0]                                  # the facing init is not penalised
                loss = (data + 1.0 * limit + 1e-3 * penetration(J) + (prior_w * rest ** 2).sum()
                        + 1e-5 * (ident ** 2).sum())
                if a.assume_standing:
                    z = J[planar, 2]
                    loss = loss + 1e-6 * ((z - z.mean()) ** 2).sum()
                loss.backward()
                opt.step()
                if step % 50 == 0 or step == a.steps - 1:
                    hist.append({"step": step, "loss": float(loss), "data": float(data), "limit": float(limit)})
            return theta.detach(), ident.detach(), scale.detach().exp(), shift.detach(), float(data), hist

        t3 = time.perf_counter()
        fits = [run_fit(r) for r in (0.0, 3.14159)]                        # facing the camera or away
        theta, ident, scale, shift, data, hist = min(fits, key=lambda f: f[4])
        res["fit_s"] = time.perf_counter() - t3
        res["fit_history"] = hist
        res["fit_init_losses"] = [f[4] for f in fits]

        with torch.no_grad():
            verts, st = model(ident, theta * pose_mask, None)
        J = st[0, :, :3]
        proj = (torch.stack([J[:, 0], -J[:, 1]], 1) * scale + shift).numpy()

        # --- measured residuals and hand checks ---------------------------------------------------
        resid = {}
        for k, j in KP_TO_JOINT.items():
            if sc[k] >= a.kp_thr:
                resid[k] = float(np.linalg.norm(proj[names.index(j)] - kp[k]))
        res["reprojection"] = {"label": "VERIFIED", "units": "pixels",
                               "note": "distance between projected fitted joints and detected keypoints",
                               "body_mean_px": _mean([v for k, v in resid.items() if k < 91]),
                               "hands_mean_px": _mean([v for k, v in resid.items() if k >= 91]),
                               "person_extent_px": size}
        hands = {}
        for s, base in (("left", 91), ("right", 112)):
            sd = s[0]
            chains_kp = [[base + 1 + 4 * f + k for k in range(4)] for f in range(5)]
            detected = [all(sc[c] >= a.kp_thr for c in ch) for ch in chains_kp]
            hand_px = float(np.linalg.norm(kp[base + 9] - kp[base])) if sc[base] >= a.kp_thr and sc[base + 9] >= a.kp_thr else None
            tips_ok = []
            for f, ch in enumerate(chains_kp):
                if detected[f] and hand_px:
                    jt = names.index(f"{sd}_{'thumb_null' if f == 0 else FINGERS[f] + '_null'}")
                    tips_ok.append(bool(np.linalg.norm(proj[jt] - kp[ch[-1]]) <= 0.35 * hand_px))
            tips3d = J[[names.index(f"{sd}_{'thumb_null' if f == 0 else FINGERS[f] + '_null'}") for f in range(5)]]
            sep = torch.cdist(tips3d, tips3d) + torch.eye(5) * 1e9
            pts = finger_points(J, sd)
            pen_pairs = sum(1 for i in range(5) for k in range(i + 1, 5)          # skeleton axes only
                            if float(torch.cdist(pts[i][1:], pts[k][1:]).min()) < 1.0)
            hands[s] = {"fingers_model": 5, "fingers_detected": int(sum(detected)),
                        "fingertips_matched": int(sum(tips_ok)), "fingertips_checked": len(tips_ok),
                        "hand_size_px": hand_px, "min_fingertip_separation_cm": float(sep.min()),
                        "finger_axis_pairs_closer_than_1cm": pen_pairs,
                        "note": "skeleton-axis distance heuristic, not a mesh collision test",
                        "label": "ESTIMATED"}
        res["hands"] = hands
        th = (theta * pose_mask)[0]
        viol = torch.maximum(th[lim_i] - lim_hi, lim_lo - th[lim_i]).clamp_min(0)
        res["joint_limits"] = {"kind": "soft penalty during the fit", "count": len(lims),
                               "violated": int((viol > 1e-4).sum()), "max_violation": float(viol.max()),
                               "units": "radians for rotations, model units for scale/length parameters"}

        import trimesh
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(verts[0].numpy() / 100.0, model.character.mesh.faces, process=False)   # cm -> m
        mesh.export(out / "mhr_fit.glb")
        np.savez(out / "fit.npz", model_parameters=(theta * pose_mask).numpy(), identity=ident.numpy(),
                 scale=scale.numpy(), shift=shift.numpy(), joints_cm=J.numpy(), joints_px=proj,
                 keypoints=kp, scores=sc, joint_names=np.array(names))
        _overlay(img, kp, sc, proj, names, a.kp_thr, out / "overlay.png")
        res.update(vertices=int(mesh.vertices.shape[0]), faces=int(mesh.faces.shape[0]),
                   mesh_path=str(out / "mhr_fit.glb"), units="metres, +Y up, person faces +Z",
                   torch=torch.__version__, onnxruntime=ort.__version__)
        res["status"] = "success"
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        return _finish(a, res, 1)


def _sessions(wb):
    out = []
    for part in ("det_model", "pose_model"):
        s = getattr(getattr(wb, part, None), "session", None)
        if s is not None:
            out.append(s)
    return out


def _mean(v):
    return float(sum(v) / len(v)) if v else None


def _overlay(img, kp, sc, proj, names, thr, path):
    """Detected hand/body keypoints (green) and projected fitted joints (red)."""
    import cv2

    im = img.copy()
    r = max(2, int(round(max(im.shape[:2]) / 400)))
    for k, j in KP_TO_JOINT.items():
        if sc[k] >= thr:
            cv2.circle(im, tuple(int(x) for x in kp[k]), r + 1, (0, 200, 0), -1)
        p = proj[names.index(j)]
        cv2.circle(im, (int(p[0]), int(p[1])), r, (0, 0, 230), -1)
    cv2.imwrite(str(path), im)


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
