"""LoRA fine-tuning of TripoSG's flow transformer. Runs inside the TripoSG environment.

TripoSG ships inference code only. This trainer rebuilds the training
objective from the released pieces:
  x0    = VAE.encode(surface points + normals of a ground-truth mesh)   (latent, 2048 x 64)
  cond  = DINOv2(front image cropped and padded like TripoSG's prepare_image)
  x_s   = (1 - s) * x0 + s * noise,  s ~ logit-normal,  timestep = 1000 * s
  loss  = || DiT(x_s, timestep, cond) - (x0 - noise) ||^2
which matches RectifiedFlowScheduler.step (x <- x + (s - s_next) * output).
10 % of the steps drop the image (zero embedding) so classifier-free guidance
keeps working. Only LoRA adapters on the attention projections are trained;
the base weights stay frozen and untouched on disk.

Usage (called by vector3d.lab.finetune):
  python triposg_finetune.py --repo DIR --weights DIR --data DIR --out DIR --result result.json
      [--steps 1500 --rank 16 --lr 1e-4 --accum 4 --gt-half auto --vram-fraction 0.9 --seed 0 --save-every 100]
Latents and image tokens are cached in the data dir (latent_cache_*.pt) and a
checkpoint is written every --save-every steps: rerunning the same command
after an interruption resumes instead of starting over.
  python triposg_finetune.py --repo DIR --tiny ...   # random tiny model on CPU: checks the mechanics only

Data dir: one folder per sample with cond.png (RGBA, front view) and
surface.npz (points (N,3), normals (N,3), object frame: +Y up, front +Z).
"""
import argparse
import json
import math
import os
import platform
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--data", "--out", "--result"):
        ap.add_argument(a, required=True)
    ap.add_argument("--weights")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--gt-half", default="auto", help="half extent of GT meshes in model space, or 'auto'")
    ap.add_argument("--vram-fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100, help="checkpoint interval in steps (resumable)")
    ap.add_argument("--tiny", action="store_true", help="random tiny model on CPU, for tests")
    a = ap.parse_args()

    res = {"status": "failed", "python": sys.version.split()[0], "platform": platform.platform(),
           "config": {k: v for k, v in vars(a).items()}}
    try:
        import numpy as np
        import torch

        res.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if a.tiny:
            dev, dtype = torch.device("cpu"), torch.float32
        else:
            if not torch.cuda.is_available():
                res["reason"] = "CUDA not available; refusing to fine-tune TripoSG on CPU"
                return _finish(a, res, 2)
            dev, dtype = torch.device("cuda"), torch.bfloat16
            torch.cuda.set_per_process_memory_fraction(a.vram_fraction, 0)
            torch.cuda.reset_peak_memory_stats()
            res["gpu"] = torch.cuda.get_device_name(0)
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)

        sys.path[:] = [a.repo, os.path.join(a.repo, "scripts")] + [
            p for p in sys.path if os.path.normcase(os.path.abspath(p or os.curdir)) !=
            os.path.normcase(os.path.dirname(os.path.abspath(__file__)))]
        import triposg.models.autoencoders.autoencoder_kl_triposg as ak
        if not hasattr(ak, "fps"):                 # torch_cluster import is commented out upstream
            ak.fps = farthest_point_sampling

        samples = sorted(d for d in os.listdir(a.data) if os.path.isfile(os.path.join(a.data, d, "surface.npz")))
        if not samples:
            res["reason"] = f"no samples in {a.data}"
            return _finish(a, res, 3)
        res["samples"] = len(samples)

        t0 = time.perf_counter()
        vae, dit, embed = load_models(a, dev, dtype)
        res["load_s"] = time.perf_counter() - t0

        gt_half = measure_gt_half(a, samples) if a.gt_half == "auto" else float(a.gt_half)
        res["gt_half"] = gt_half

        # --- cache latents (mean, std) and image embeddings, kept on disk and resumable -------------
        tag = "tiny" if a.tiny else "full"
        cache_path = os.path.join(a.data, f"latent_cache_{tag}_h{gt_half:.4f}.pt")
        store = torch.load(cache_path, weights_only=False) if os.path.isfile(cache_path) else {}
        res["cache_reused"] = sum(s in store for s in samples)
        with torch.no_grad():
            for i, s in enumerate(samples):
                if s in store:
                    continue
                d = os.path.join(a.data, s)
                z = np.load(os.path.join(d, "surface.npz"))
                pts, nrm = normalise(z["points"], gt_half), z["normals"]
                k = np.random.default_rng([a.seed, i]).choice(len(pts), min(len(pts), 20480), replace=False)
                surf = torch.from_numpy(np.concatenate([pts[k], nrm[k]], 1)).float()[None].to(dev, dtype)
                dist = vae.encode(surf).latent_dist
                cond = embed(os.path.join(d, "cond.png"))
                store[s] = (dist.mean.cpu(), dist.std.cpu(), cond.cpu())
                if len(store) % 25 == 0:
                    torch.save(store, cache_path)
        torch.save(store, cache_path)
        cache = [store[s] for s in samples]
        res["cache_s"] = time.perf_counter() - t0 - res["load_s"]
        del vae
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # --- LoRA on the attention projections -------------------------------------------------------
        from peft import LoraConfig, get_peft_model

        dit.requires_grad_(False)
        dit = get_peft_model(dit, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0,
                                             target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        params = [p for p in dit.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        res["trainable_params"] = int(sum(p.numel() for p in params))
        base = dit.base_model.model
        base.gradient_checkpointing = not a.tiny
        dit.train()
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        warm = max(1, a.steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))

        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        os.makedirs(a.out, exist_ok=True)
        ckpt = os.path.join(a.out, "checkpoint.pt")
        log, t1, start = [], time.perf_counter(), 0
        if os.path.isfile(ckpt):                 # resume an interrupted run with the same settings
            ck = torch.load(ckpt, weights_only=False)
            if ck["steps"] == a.steps and ck["rank"] == a.rank:
                set_peft_model_state_dict(dit, ck["lora"])
                for p in params:
                    p.data = p.data.float()
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                start, log = ck["step"], ck["log"]
                rng.bit_generator.state = ck["rng"]
                torch.set_rng_state(ck["torch_rng"])
                res["resumed_from_step"] = start
        for step in range(start, a.steps):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(a.accum):
                mean, std, cond = cache[int(rng.integers(len(cache)))]
                x0 = (mean + std * torch.randn_like(mean)).to(dev, torch.float32)
                noise = torch.randn_like(x0)
                s = torch.sigmoid(torch.randn(1, device=dev))           # logit-normal, as in SD3/TripoSG
                xs = (1 - s) * x0 + s * noise
                c = cond.to(dev, dtype)
                if rng.random() < a.cfg_drop:
                    c = torch.zeros_like(c)
                with torch.autocast(dev.type, dtype=dtype, enabled=dev.type == "cuda"):
                    out = dit(xs.to(dtype), (s * 1000).to(dtype), encoder_hidden_states=c).sample
                loss = torch.nn.functional.mse_loss(out.float(), x0 - noise) / a.accum
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 10 == 0 or step == a.steps - 1:
                row = {"step": step, "loss": total, "lr": sched.get_last_lr()[0], "s": time.perf_counter() - t1}
                if dev.type == "cuda":
                    row["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
                log.append(row)
                print(json.dumps(row), flush=True)
            if (step + 1) % a.save_every == 0 and step + 1 < a.steps:
                torch.save({"lora": get_peft_model_state_dict(dit), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step + 1, "log": log, "steps": a.steps,
                            "rank": a.rank, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, ckpt)
        res["train_s"] = time.perf_counter() - t1
        res["log"] = log
        if dev.type == "cuda":
            res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
            res["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        dit.save_pretrained(a.out)
        if os.path.isfile(ckpt):
            os.remove(ckpt)
        res["lora_dir"] = a.out
        res["status"] = "success"
        return _finish(a, res, 0)
    except Exception as e:  # noqa: BLE001 - every failure must reach the result file
        res["reason"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        return _finish(a, res, 1)


def load_models(a, dev, dtype):
    """(vae, transformer, embed(path) -> image tokens). Tiny mode: random small models."""
    import torch

    if a.tiny:
        from triposg.models.autoencoders import TripoSGVAEModel
        from triposg.models.transformers import TripoSGDiTModel

        vae = TripoSGVAEModel(num_attention_heads=2, width_encoder=64, width_decoder=64,
                              num_layers_encoder=1, num_layers_decoder=1).to(dev)
        dit = TripoSGDiTModel(num_attention_heads=2, width=64, in_channels=64, num_layers=3,
                              cross_attention_dim=32).to(dev)
        proj = torch.randn(3 * 16 * 16, 32) / 30

        def embed(path):
            img = prepare_cond(path).resize((64, 64))
            import numpy as np
            x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255)                  # 64x64x3
            patches = x.unfold(0, 16, 16).unfold(1, 16, 16).reshape(16, -1)              # 16 patches
            return (patches @ proj)[None]

        vae.encode_tokens = 64
        return _tiny_encode(vae), dit, embed

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    pipe = TripoSGPipeline.from_pretrained(a.weights)
    vae = pipe.vae.to(dev, dtype)
    dit = pipe.transformer.to(dev, dtype)
    enc = pipe.image_encoder_dinov2.to(dev, dtype)
    fe = pipe.feature_extractor_dinov2

    def embed(path):
        px = fe(prepare_cond(path), return_tensors="pt").pixel_values.to(dev, dtype)
        return enc(px).last_hidden_state

    return vae, dit, embed


def _tiny_encode(vae):
    """Tiny VAE with fewer tokens, so the CPU test stays fast."""
    import types

    orig = vae.encode

    def encode(self, x, **kw):
        return orig(x, num_tokens=self.encode_tokens)

    vae.encode = types.MethodType(encode, vae)
    return vae


def prepare_cond(path, padding_ratio: float = 0.1):
    """TripoSG's prepare_image for an RGBA input with valid alpha, on CPU: white background, bbox crop, square pad."""
    import numpy as np
    from PIL import Image

    rgba = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
    alpha = rgba[..., 3:]
    rgb = rgba[..., :3] * alpha + (1 - alpha)
    ys, xs = np.nonzero(alpha[..., 0] > 0)
    x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    crop = rgb[y:y + h, x:x + w]
    if w > h:
        lr = int(w * padding_ratio)
        tb = int(lr + (w - h) / 2)
    else:
        tb = int(h * padding_ratio)
        lr = int(tb + (h - w) / 2)
    out = np.pad(crop, ((tb, tb), (lr, lr), (0, 0)), constant_values=1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def normalise(points, half):
    """Centre the bounding box and scale its largest half extent to `half` (TripoSG model space)."""
    import numpy as np

    lo, hi = points.min(0), points.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max() / 2)
    return ((points - c) * (half / r)).astype(np.float32)


def measure_gt_half(a, samples) -> float:
    """Model-space size TripoSG itself produces; read from the dataset's base-model probe if present."""
    probe = os.path.join(a.data, "base_probe.json")
    if os.path.isfile(probe):
        return float(json.load(open(probe))["median_half_extent"])
    return 0.9 if a.tiny else _probe(a, samples, probe)


def _probe(a, samples, probe) -> float:
    """Run the frozen base model on 4 training images and record the half extent of its meshes."""
    import numpy as np
    import torch
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    pipe = TripoSGPipeline.from_pretrained(a.weights).to("cuda", torch.float16)
    halves = []
    for s in samples[:4]:
        out = pipe(image=prepare_cond(os.path.join(a.data, s, "cond.png")),
                   generator=torch.Generator(device="cuda").manual_seed(0), num_inference_steps=25).samples[0]
        v = out[0]
        halves.append(float((v.max(0) - v.min(0)).max() / 2))
    del pipe
    torch.cuda.empty_cache()
    m = float(np.median(halves))
    json.dump({"median_half_extent": m, "halves": halves, "samples": samples[:4]}, open(probe, "w"), indent=1)
    return m


def farthest_point_sampling(points, batch, ratio=0.25, random_start=False):
    """Stand-in for torch_cluster.fps: indices into `points` (flattened over the batch)."""
    import torch

    out = []
    for b in torch.unique(batch):
        idx = torch.nonzero(batch == b).flatten()
        p = points[idx].float()
        k = max(1, int(math.ceil(len(idx) * ratio)))
        chosen = torch.empty(k, dtype=torch.long, device=p.device)
        dist = torch.full((len(idx),), float("inf"), device=p.device)
        cur = int(torch.randint(len(idx), (1,))) if random_start else 0
        for i in range(k):
            chosen[i] = cur
            dist = torch.minimum(dist, ((p - p[cur]) ** 2).sum(1))
            cur = int(dist.argmax())
        out.append(idx[chosen])
    return torch.cat(out)


def _finish(a, res, code: int) -> int:
    with open(a.result, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
