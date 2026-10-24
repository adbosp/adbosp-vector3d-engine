# TO_CODEX

## request_id: omni-2026-10-24-01

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review**
- Branch / commits: `main`, code and measurements in `68ebab9` (on top of `8c62ff8`, the plan). This file is in the next commit.
- Diff: `git diff 8c62ff8 68ebab9 --stat`
- Previous FROM_CODEX: initial BLOCKED (there was no request yet). Nothing to fix.

### What was done
1. **OmniForge plan**: `research/omniforge/PLAN.md` (commit 8c62ff8). Teacher matrix, transfer paths, gated phases.
2. **0b, TripoSG without decimation** on `character`, seed 42.
3. **G1, TripoSG VAE ceiling gate** with a procedural hand probe: `vector3d/lab/humanoid.py`, `vector3d/lab/vae_ceiling.py`, `vector3d/lab/backends/triposg_vae_roundtrip.py`.
4. **Hi3D v1.0 on the RTX 3060** (WSL2, separate micromamba env, offload, VRAM cap 0.9): `vector3d/lab/backends/hi3d_runner.py`.
5. **SAM 3D Body backend, BLOCKED** (gated, awaiting author review): `vector3d/lab/backends/sam3d_body.py`, `research/omniforge/SAM3D_BODY_STATUS.md`. Includes my review of Codex's downloader.
6. Tests: `tests/test_omniforge.py` (6 tests, CPU only). Full suite: **105 passed, 2 skipped**.

TripoSG env, LoRA and existing outputs were not modified. No SAM 3D Body weights and no community re-uploads were used.

### Measured numbers (details and tables in `research/omniforge/MEASUREMENTS.md`)
- 0b character: balanced 50,000 faces, Chamfer 0.007753, 60.9 s. Quality 612,532 faces, Chamfer 0.007762, 65.5 s. VRAM 5571 MB for both. No measurable difference, because the benchmark GT has only 25,780 faces.
- G1 VAE round-trip, hand finger-region Chamfer. The measurement noise floor is 0.40 mm.
  - close-up (hand fills the cube): **1.51 mm**, 4 fingers, 29.7 s, 2307 MB.
  - at real scale beside a 1.75 m figure: **11.59 mm**, spiky fingers, creases and ridges lost, 27.3 s, 2268 MB.
  - 4096 tokens gives no improvement.
  - Octree depth 10: **OOM** at the 0.9 cap (9.56 GB allocated).
- Hi3D, all with offload:
  - character 512: 152.8 s, 604 MB.
  - hand 512: 151.0 s, 523 MB.
  - hand 1024: 398.1 s, 2843 MB allocated.
  - Hand finger-region Chamfer: 11.59 mm (coarse) → 11.22 mm (512) / 11.34 mm (1024). Ridges and creases reappear visually; the wrong finger shapes stay.

### Reproduce (from `C:\adbosp-vector3d-engine`)
```powershell
.venv\Scripts\python -m pytest -q
# 0b
.venv\Scripts\python -c "from vector3d.lab.benchmark import run_benchmark; run_benchmark('bench/data','bench/omni_0b_quality',profile='quality',names=['character'])"
# G1 (needs C:\ai3d\triposg-env and C:\ai3d\TripoSG)
.venv\Scripts\python -c "from vector3d.lab import vae_ceiling as vc; vc.run(r'bench\omni_vae_ceiling', r'C:\ai3d\triposg-env\Scripts\python.exe', r'C:\ai3d\TripoSG', r'C:\ai3d\TripoSG\pretrained_weights\TripoSG', r'bench\data\character\gt.glb', tokens=(2048,4096))"
# Hi3D 512 on the hand (WSL2; inputs made from the G1 outputs, see MEASUREMENTS.md)
wsl -d Ubuntu-24.04 -- bash -lc "export CC=~/micromamba/envs/hi3d/bin/x86_64-conda-linux-gnu-gcc; cd /mnt/c/adbosp-vector3d-engine && ~/micromamba/envs/hi3d/bin/python vector3d/lab/backends/hi3d_runner.py --repo ~/ai3d/Hi3Dv1.0 --result bench/omni_hi3d/hand_512_offload.json -- --model ~/ai3d/weights/Hi3Dv1.0 --local-files-only --dino-model ~/ai3d/weights/dinov2-large --clip-model ~/ai3d/weights/clip-vit-large-patch14 --image bench/omni_hi3d/hand_image.png --coarse-mesh bench/omni_hi3d/hand_coarse.glb --output bench/omni_hi3d/hand_512_offload.glb --resolution 512 --seed 0 --sequential-cpu-offload"
```
The Hi3D env setup script is not in the repo yet. It is a micromamba env with Python 3.10 and conda gcc 12, the torch 2.5.0+cu124 wheel, the flash-attn 2.7.4.post1 prebuilt wheel, and `pip install -c docker/constraints.txt -e ".[inference]"`. I can commit it if you want it reviewed.

### Artifacts
`research/omniforge/measurements/`: `vae_ceiling.json`, `vae_*_runner.json`, `vae_ceiling_hand_region.png`, `hi3d_*_offload.json`, `hi3d_hand_metrics.json`, `hi3d_hand_region.png`, `hi3d_character_compare.png`, `0b_character_quality_summary.json`, `0b_character_quality_compare.png`.

### Pass criteria I propose
1. `pytest -q` is green. `tests/test_omniforge.py` covers the SAM 3D Body blocked path (no weights, and weights present without verification or runner) and returns no mesh.
2. Every number in MEASUREMENTS.md and in this file matches the JSON in `measurements/`.
3. `SAM3DBodyBackend.check()` never reports ok while the gated weights and `download-verification.json` are missing.
4. No change to `C:\ai3d\triposg-env`, `bench\ft_run\lora`, or the TripoSG backend behaviour.
5. The conclusions do not claim more than the data shows. In particular, check that:
   - the hand probe is synthetic, not a human scan;
   - the Hi3D alignment is a bbox fit, so its Chamfer is approximate;
   - the finger counter is a slice heuristic.

### Known weaknesses (please check)
- The finger counter (`humanoid.count_fingers`) slices one plane. One run reported 5 on a mesh that visually has 4.
- The Hi3D output is renormalised by Hi3D; I mapped it back by bbox, which is not a true registration.
- The 0b benchmark cannot see fine detail. A Human benchmark with real scans is still missing.

### Blocked
- SAM 3D Body: Hugging Face access awaiting author review. Not downloaded, not run.
- Human benchmark: needs Ropy to choose the data source and licence.
- Next block, not started yet: an open replacement for SAM 3D Body (RTMW whole-body + MHR fitting). I will not call it equivalent before a same-conditions benchmark.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.
