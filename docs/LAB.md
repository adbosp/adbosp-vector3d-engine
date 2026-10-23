# AI 3D Lab

The lab runs real image-to-3D foundation models, measures them and refines their output with geometry. Model choice and licences: [research/model_matrix.md](../research/model_matrix.md). Measured results: [research/benchmark_report.md](../research/benchmark_report.md).

## Commands
```bash
python -m vector3d lab audit -o audit.json             # GPU, VRAM, driver, CUDA, RAM, disk, compilers. Run first.
python -m vector3d lab dataset -o bench/data           # 6 synthetic objects with exact ground truth
python -m vector3d lab experiment --data bench/data -o bench/exp001
python -m vector3d lab generate --image photo.png -o runs/one --profile balanced [--download]
python -m vector3d lab render model.glb -o runs/check   # quality gates + front/back/left/right + turntable
python -m vector3d lab benchmark --data bench/data -o bench/triposg [--download]
python -m vector3d lab report --experiment bench/exp001 --benchmark bench/triposg --audit audit.json -o research/benchmark_report.md
python -m vector3d lab imagine --data bench/data -o bench/imagine   # imagination loop v0 vs visual hull, CPU
python -m vector3d lab segbench --data bench/data -o bench/seg [--methods classic,isnet,birefnet] [--images photos/]
```

## Module map (`vector3d/lab/`)
| Module | Role |
|---|---|
| `env_audit.py` | hardware check and per-model verdict from the official VRAM figures |
| `profiles.py` | FAST / BALANCED / QUALITY / EXPERIMENTAL settings, hard VRAM cap |
| `backends/base.py` | `Backend.generate(reference_images, camera_metadata=None, prompt=None, quality_profile="balanced")` → `GenerationRecord` (model, version, config, seed, time, peak VRAM, output, status) |
| `backends/triposg.py` + `triposg_runner.py` | TripoSG in its own Python env through a subprocess |
| `render.py` | numba orthographic rasterizer: silhouettes, shaded previews, turntables |
| `metrics.py` | silhouette IoU, Chamfer-L1, F-score, normal consistency, mesh validity |
| `align.py` | placing a model's mesh in the reference frame (silhouette-only, or GT for scoring) |
| `refine.py` | render-and-compare silhouette refinement (first Phase D experiment) |
| `gates.py` | acceptance gates, including Blender round-trip when Blender is on PATH |
| `dataset.py`, `experiments.py`, `benchmark.py`, `report.py` | benchmark set, experiments, A-vs-B benchmark, report |
| `imagine.py` | imagination loop v0: silhouettes kept exact, unseen cross-sections chosen by shading per part ([research/imagination_engine.md](../research/imagination_engine.md)) |
| `segbench.py` | background-removal benchmark: dataset views over plain, gradient, noise and clutter backgrounds; IoU, alpha error, time per image |

## Fine-tuning TripoSG (LoRA)
TripoSG ships inference code only; `backends/triposg_finetune.py` rebuilds the rectified-flow training objective from the released VAE encoder, DINOv2 conditioner and transformer, and trains LoRA adapters on the attention projections (base weights frozen). Training data comes from `finetune.py`: procedural objects with exact geometry, weighted towards the boxy hard-surface shapes TripoSG flattens from one image.
```bash
python -m vector3d lab finetune-data -o bench/ft_data --count 600          # CPU, ~1 s per sample
python -m vector3d lab finetune --data bench/ft_data -o bench/ft_run         # GPU, TripoSG env (needs peft)
python -m vector3d lab benchmark --data bench/data -o bench/triposg_lora --lora bench/ft_run/lora
```
The first `finetune` run probes the base model on 4 training images to learn the size of its model space (`base_probe.json`), so ground truth is scaled the way TripoSG expects. The run writes `train_result.json` (loss curve, peak VRAM, time) next to the adapter. Compare `bench/triposg` (released weights) with `bench/triposg_lora` on the same synthetic_v1 inputs; synthetic_v1 objects are hand-written and never produced by the training generators.

## Rules the code enforces
- A backend that cannot run returns `status: blocked` with the reason. It never returns a stand-in mesh.
- TripoSG refuses to run without CUDA instead of falling back to the CPU silently.
- Reference images are copied into the run folder with their SHA-256 hash and never modified.
- The raw model mesh (`raw_triposg.glb`) is kept; placed and refined meshes are separate files.
- Background removal uses only MIT or Apache-2.0 models (BiRefNet, IS-Net, U²-Net), always named explicitly; rembg's default BRIA RMBG (non-commercial) is never loaded. TripoSG gets the soft alpha, so it skips its own RMBG-1.4 step.
- Labels: model output `AI_GENERATED`, refined output `ESTIMATED`, simulated baselines `SIMULATED_BASELINE`.

## Installing TripoSG on Windows 11 (RTX 3060)
TripoSG pins `numpy==1.22.3` and builds `diso` with CUDA, so give it its own environment:
```powershell
git clone https://github.com/VAST-AI-Research/TripoSG C:\ai3d\TripoSG
py -3.10 -m venv C:\ai3d\triposg-env
C:\ai3d\triposg-env\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
C:\ai3d\triposg-env\Scripts\python -m pip install -r C:\ai3d\TripoSG\requirements.txt
setx V3D_TRIPOSG_DIR C:\ai3d\TripoSG
setx V3D_TRIPOSG_PYTHON C:\ai3d\triposg-env\Scripts\python.exe
```
Building `diso` needs the CUDA toolkit (`nvcc`) and the Visual Studio Build Tools. If it does not build, use WSL2 Ubuntu with the same steps. The first `--download` fetches `VAST-AI/TripoSG` from HuggingFace (several GB). RMBG-1.4 is not needed: the adapter cuts the background itself and passes RGBA.
