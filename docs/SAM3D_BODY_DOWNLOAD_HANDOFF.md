# SAM 3D Body DINOv3 - handoff for 3D Engine

Checked: 2026-10-24, Asia/Bangkok.

## Verified status

- Repository: https://huggingface.co/facebook/sam-3d-body-dinov3/tree/main
- Local project: `C:\adbosp-vector3d-engine`.
- Claude project: `3D Engine`, chat `Vector3D Engine và lab ảnh sang 3D`.
- The signed-in browser says: "Your request to access this repository has been submitted and is awaiting a review from the repository authors."
- User reports that they accepted the terms. Acceptance has not yet resulted in download access.
- Weights are NOT downloaded; inference and integration have NOT been tested.
- No existing model, Python environment, or engine source has been changed.

## Files listed by the official repository

| File | Bytes |
| --- | ---: |
| model.ckpt | 2109129346 |
| assets/mhr_model.pt | 696110248 |
| model_config.yaml | 1488 |
| README.md | 5761 |
| LICENSE | 8204 |

The two weight files total 2,805,239,594 bytes. API responses currently hide LFS hashes while gated; no remote hash validation has been claimed.

## Continue after access approval

Run `Download-SAM3D.ps1` from this folder using PowerShell. It creates an isolated download environment here, installs the official `huggingface_hub` package, authenticates with an existing token if needed, checks gated access, pins the current repository commit, downloads all repository files to `C:\AI\models\sam-3d-body-dinov3`, and verifies file sizes and available LFS SHA256 hashes. It writes `download-verification.json` only after all checks succeed. The downloader is prepared; it has not yet been run against an approved account.

For Claude: inspect the official model card and upstream implementation after approval, then handle runtime setup and integration into the engine. Do not represent this as a general object reconstruction replacement; this repository is SAM 3D Body. Preserve existing TripoSG, LoRA, outputs, and environments. Follow the project's AGENTS.md. Report blocked when access or runtime dependencies are missing; do not substitute a mesh or claim successful inference without running it.

Prepared package location: `C:\Users\ROPY\Documents\Codex\2026-10-24\tai\outputs\sam3d-handoff`.
