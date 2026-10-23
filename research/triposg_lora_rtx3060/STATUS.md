# TripoSG LoRA fine-tune on ROPY-PC (RTX 3060): paused

Paused 2026-10-23 at about 20:40 local, at the owner's request. Nothing is running.

## Done
- `main` at a72bba6 or later; engine tests: 97 passed, 2 skipped.
- `peft` 0.21.2 present in `C:\ai3d\triposg-env`.
- `bench\data` (6 benchmark objects) exists.
- `bench\ft_data`: 600 training samples built with the committed `distill.py` (369 s). Kept on disk, no need to rebuild.
- Smoke test `bench\ft_smoke` (20 steps): **success**.
  - load 12.7 s, latent/embedding cache 3,624 s (in memory only, not saved)
  - 6.85 s per step, peak VRAM 9,938 MB (reserved 10,934 MB)
  - loss 1.777 (step 0) → 1.750 (step 10) → 1.743 (step 19)
  - gt_half 0.9913 (`bench\ft_data\base_probe.json`)

## Stopped
- Full run `bench\ft_run` (1,300 steps), started 20:17, stopped by me at 20:40 while still caching.
  No checkpoint existed yet: the trainer saves the LoRA only at the end and has no resume.
  Only about 23 minutes of caching were lost.

## To continue (about 3.5 h on this PC)
Close Unity, Blender and other GPU apps first: training peaks near 10 GB of the 12 GB card.

```powershell
cd C:\adbosp-vector3d-engine
git pull origin main
Start-Process .venv\Scripts\python.exe -ArgumentList "-m","vector3d","lab","finetune","--data","bench\ft_data","-o","bench\ft_run","--steps","1300" -RedirectStandardOutput bench\ft_run\console.log -RedirectStandardError bench\ft_run\console.err.log -WindowStyle Hidden
```
About 1 h of caching, then about 2.5 h of training (1,300 × 6.85 s).

Then (one at a time, the GPU cannot hold two):
```powershell
.venv\Scripts\python -m vector3d lab benchmark --data bench\data -o bench\triposg_base_rerun
.venv\Scripts\python -m vector3d lab benchmark --data bench\data -o bench\triposg_lora --lora bench\ft_run\lora
```
Compare the two `summary.json` files per object (chamfer_l1, fscore@0.02, heldout_iou_mean,
normal_consistency, duration_s, peak_vram_mb) and look at crate and bracket compare/turntable images.

## Possible improvement
Caching takes about 6 s per sample and is redone every run. Saving the cache to disk
(for example `bench\ft_data\cache.pt`) would cut every later run by about an hour.
