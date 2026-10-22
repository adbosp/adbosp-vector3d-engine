# Model matrix: image-to-3D backends for an RTX 3060 12GB on Windows 11

Checked on 2026-10-22 against each project's own README, LICENSE and model card. "Not stated" means the official page does not give the value, so we have not guessed it.
"Observed performance" is empty until we run the model ourselves. The cloud container that wrote this file has no GPU and cannot reach HuggingFace, so **none of these models has been run yet**.

## 1. 3D foundation models (image → shape)

| | TripoSG | Hunyuan3D-2 / 2mini / 2mv | Hunyuan3D-2.1 | TRELLIS.2 | Pixal3D |
|---|---|---|---|---|---|
| Official source | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) |
| Release | 2025-03 (1.5B model + VAE); scribble variant 2025-04 | 2025 (2mini, 2mv added later in the year) | 2025-06-13 | arXiv 2512.14692 (Dec 2025, inferred from the id) | Code and demo May 2026; SIGGRAPH 2026 |
| Repo available | Yes | Yes | Yes | Yes | Yes (`main` = TRELLIS.2 backbone, `paper` = Direct3D-S2) |
| Checkpoints | `VAST-AI/TripoSG`, `VAST-AI/TripoSG-scribble` | `tencent/Hunyuan3D-2` (dit-v2-0, -fast, -turbo, paint), `tencent/Hunyuan3D-2mini` (0.6B), `tencent/Hunyuan3D-2mv` (1.1B) | `tencent/Hunyuan3D-2.1`: shape 3.3B, paint-PBR 2B | `microsoft/TRELLIS.2-4B` (4B) | `TencentARC/Pixal3D` (file names not stated) |
| License | **MIT** | **Tencent Hunyuan 3D 2.0 Community License**: not valid in the EU, UK or South Korea; a separate licence is needed above 1M MAU; outputs may not be used to improve other AI models | **Tencent Hunyuan 3D 2.1 Community License**: same three restrictions, and outputs may not be used or shown outside the licensed territory | **MIT** (nvdiffrast/nvdiffrec keep their own licences) | **MIT** (third-party parts per NOTICE) |
| Required VRAM (official) | "at least 8GB" | 6 GB for shape, 16 GB for shape plus texture | 10 GB for shape, 21 GB for texture, 29 GB for both | "at least 24GB" (tested on A100/H100) | Not stated. `--low_vram` loads models on demand, at resolution 1024 instead of 1536 |
| OS (official) | Not stated | Windows, macOS, Linux | Windows, macOS, Linux | **Linux only** | Not stated; HF demo requirements target H-series GPUs |
| Inputs | 1 image (RGBA alpha is used directly, otherwise RMBG-1.4 runs); scribble variant takes scribble plus prompt | 1 image; 2mv takes a dict `front`/`left`/`back` | 1 image | 1 image | 1 image (`inference.py`); **no `inference_mv.py` or `transforms.json` in the repo** as of this check |
| Outputs | GLB, geometry only, `--faces N` decimation (pymeshlab) | trimesh/GLB; Paint gives an RGB texture | GLB with PBR texture (paint pipeline) | GLB with PBR (O-Voxel), MP4 preview | GLB, PBR claimed |
| Key dependencies | torch, diffusers, `diso` (CUDA build), pymeshlab, **numpy==1.22.3 pin** | custom_rasterizer and differentiable_renderer (CUDA build) for texture | PyTorch 2.5.1+cu124, two CUDA extensions, RealESRGAN weights | flash-attn, nvdiffrast, nvdiffrec, CuMesh, O-Voxel, FlexGEMM; CUDA 12.4 | TRELLIS.2 environment plus natten 0.21.0, utils3d |
| Observed performance | **Not run yet** | Not run yet | Not run yet | Not run yet | Not run yet |
| Integration complexity | Low to medium: one script; risks are compiling `diso` on Windows and the numpy pin (isolated env) | Medium: shape-only is pip installable; texture needs compiled extensions | Medium to high | High: Linux and 24GB | High: TRELLIS.2 stack plus natten |

### Ranked for RTX 3060 12GB / Windows 11

1. **TripoSG**, first backend. MIT licence, official minimum 8GB, single script. We pass RGBA with alpha already cut by our own segmenter, so RMBG-1.4 never runs; its licence (BRIA) restricts commercial use and is not needed for our path. Run it in a separate Python env through a subprocess so its numpy pin cannot break the engine. If `diso` does not build on Windows, the fallback is WSL2 Ubuntu.
2. **Hunyuan3D-2mini / 2mv (shape only)**, the multi-view candidate for Phase C. 6 GB for shape is officially stated. **Licence flag:** it cannot be used if Ropy or the users are in the EU, UK or South Korea, outputs cannot train other models, and the 1M MAU limit applies. It must be opted into explicitly and never enabled by default.
3. **Hunyuan3D-2.1 shape only** (10 GB). Fits on paper. The PBR texture stage (21 GB) does not fit on 12GB, and we will not claim it does. Same licence flags.
4. **Pixal3D (experimental)**. MIT, best image alignment on paper, but no VRAM figure, no multi-view script and a TRELLIS.2/natten stack. Try it only under WSL2 with `--low_vram` and measure.
5. **TRELLIS.2: cloud only.** Officially needs 24GB and Linux. No 12GB support is claimed.

## 2. Mesh-native research (topology, quads, parts)

| Method | Source | Code | Weights | License | Inputs → outputs | Usable for us now |
|---|---|---|---|---|---|---|
| MeshFlow (CVPR 2026, Meta) | [facebookresearch/meshflow](https://github.com/facebookresearch/meshflow), [arXiv 2606.04621](https://arxiv.org/abs/2606.04621) | Yes (inference_dit/vae, evaluate, Gradio) | Yes, `facebook/meshflow`; image conditioning also needs **DINOv3 (gated by Meta)** | LICENSE file present, type **not stated on the page, check before use** | point cloud or mesh (≤4096 vertices) plus optional image → artist-style GLB mesh | Candidate for retopology after shape generation; VRAM not stated |
| QuadGPT (ICLR 2026) | [arXiv 2509.21420](https://arxiv.org/abs/2509.21420) | **No release found** | No | Paper CC BY 4.0 | → quad mesh | No. Implement a documented quad-dominant baseline instead |
| MeshMosaic (CVPR 2026) | [Xrvitd/MeshMosaic](https://gittrend.io/repo/Xrvitd/MeshMosaic) | Yes | Not stated | **AGPL-3.0**: linking it would force our engine to AGPL | patch-based artist mesh | Research only, kept out of the product |
| ExMesh (CVPR 2026) | [moumee/ExMesh](https://github.com/moumee/ExMesh) | Yes | No pretrained checkpoints found (per-scene optimisation) | LICENSE.md present, type not shown | multi-view images, masks, COLMAP, depth priors, coarse mesh → refined mesh | Ideas for Phase D (topology-adaptive refinement); needs nvdiffrast plus PyTorch3D |
| TopoMesh (CVPR 2026) | [arXiv 2603.24278](https://arxiv.org/abs/2603.24278) | Not found | Not found | — | mesh autoencoder | No |
| PartCrafter (NeurIPS 2025) | wgsxm/PartCrafter | Yes | Not confirmed from the page | MIT (per mirror listing, to be confirmed on GitHub) | image → parts | Candidate for Phase E (semantic parts) after checking weights |

## 3. What this decides
- Phase B runs **TripoSG** in its own env, invoked through `vector3d/lab/backends/triposg.py`.
- No Hunyuan model is enabled without an explicit licence acknowledgement (`--accept-license hunyuan`).
- Every unrun cell above stays empty until a real run fills it. Benchmark results go to `research/benchmark_report.md`.
