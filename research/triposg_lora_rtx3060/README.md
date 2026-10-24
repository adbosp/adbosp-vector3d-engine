# TripoSG LoRA fine-tune on RTX 3060: result

Machine ROPY-PC, NVIDIA GeForce RTX 3060 12 GB, torch 2.6.0+cu124, Windows 11. All numbers below come
from real runs on 2026-10-24; files in this folder are the raw outputs.

## Training (`bench\ft_run`, `train_result.json`)
- Data: 600 procedural samples from `distill.py` (`bench\ft_data`). LoRA rank 16, lr 1e-4, accum 4, 1,300 steps, seed 0.
- status **success**, 10.3 M trainable params, gt_half 0.9913 (`base_probe.json`).
- load 11 s, cache 461 s, train 10,869 s (about 8.4 s/step, about 3 h 10 min in total).
- peak VRAM 4,097 MB (4,292 MB reserved).
- loss: 1.804 at step 0 and 1.425 at step 1299. Mean of the first 10 log rows (steps 0 to 90) is 1.611;
  mean of the last 10 rows is 1.546. Single steps are noisy, so the real drop is small.
- Adapter: `bench\ft_run\lora\adapter_model.safetensors` (41 MB; not committed).

## Benchmark (`lab benchmark`, synthetic_v1, 6 objects, seed 42, run one after the other)
A metrics (raw output placed by silhouette, Chamfer in fractions of the GT bbox diagonal):

| object | run | chamfer_l1 ↓ | fscore@0.02 ↑ | heldout_iou_mean ↑ | normal_consistency ↑ | duration_s | peak_vram_mb |
|---|---|---|---|---|---|---|---|
| crate | base | 0.1131 | 0.1652 | 0.4214 | 0.9884 | 77.5 | 5747 |
| crate | lora | **0.0563** | **0.3878** | **0.7130** | 0.9515 | 100.0 | 6339 |
| chair | base | 0.0151 | 0.7451 | 0.7651 | 0.9558 | 64.5 | 5668 |
| chair | lora | **0.0126** | **0.7891** | **0.8297** | **0.9926** | 70.8 | 5677 |
| mug | base | **0.0083** | **0.9999** | **0.9764** | 0.9852 | 88.4 | 6217 |
| mug | lora | 0.0104 | 0.9719 | 0.9662 | **0.9880** | 95.7 | 6348 |
| character | base | **0.0078** | **0.9658** | **0.8632** | 0.9983 | 60.9 | 5571 |
| character | lora | 0.0080 | 0.9379 | 0.8212 | 0.9983 | 64.6 | 5585 |
| lamp | base | **0.0063** | **0.9887** | **0.9577** | 0.9466 | 58.8 | 5625 |
| lamp | lora | 0.0112 | 0.9510 | 0.8581 | 0.9431 | 66.9 | 5705 |
| bracket | base | 0.0333 | 0.6253 | 0.6211 | 0.9962 | 63.1 | 5623 |
| bracket | lora | **0.0172** | **0.7411** | **0.8228** | 0.9750 | 73.6 | 5865 |
| **mean** | base | 0.0307 | 0.7483 | 0.7675 | 0.9784 | 68.9 | 5742 |
| **mean** | lora | **0.0193** | **0.7965** | **0.8352** | 0.9747 | 78.6 | 5920 |

`compare.md` is the same comparison written by `lab compare`.

**Reproducibility:** the base rerun matches the earlier `research\triposg_rtx3060\summary.json` exactly on
every A metric for all 6 objects (seed 42 is deterministic). Only durations differ (68.9 s vs 74.6 s mean).

## What the images show
- **crate** (`crate_compare_*.png`): base gives a thin slab, the front panel with almost no depth.
  The LoRA gives a real box with depth, but its sides are rough and patchy, and after refinement the front panel has a dent.
- **bracket** (`bracket_compare_*.png`): base turns the knob into a long horizontal rod sticking out of both sides.
  The LoRA keeps a sphere on a correct L shape, but the surfaces are noisy. Neither has the hole in the base plate.
- **mug** (turntables in `bench`): same overall shape in both runs. The LoRA adds small fins inside the rim and a
  visible seam stripe down the back, which explains its small metric drop.

## Verdict
The LoRA fixes the base model's worst failure: missing or wrong depth on hard-surface, box-like objects
(crate, bracket, and chair to a lesser degree). That is the shape family the procedural training data covers.
Mean Chamfer drops by 37% and held-out IoU rises from 0.77 to 0.84. It is **not** a free upgrade: on the three
objects the base model already did well (mug, character, lamp), all four metrics are slightly worse. Surfaces get
noisier (lower normal consistency on crate and bracket, artefacts on the mug), and each object takes about 14% longer.
So: use it as an option for hard-surface or box-like inputs, not as the default. This is 6 synthetic objects and one seed,
so it is a signal, not proof. A larger and more varied benchmark is needed before switching the default.

## End-to-end check (what the owner tests)
`python -m vector3d generate --front bench\data\crate\views\front.png --mode ai --ai-lora -o runs\lora_check`
finished with exit 0 in 149 s and wrote model.glb, model.obj, metadata.json and mask.png.
In metadata, `model.version` is `weights:90facf66b8638c8d+lora:43fd2bfaafbee56a` and `model.lora` is `C:\adbosp-vector3d-engine\bench\ft_run\lora`.
