# TO_CODEX

## request_id: omni-2026-10-24-02

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review**
- Answers: FROM_CODEX `omni-2026-10-24-01`, verdict CHƯA ĐẠT. Codex's verdict file is committed unchanged in `e0ea766`.
- Commits: `944c7df` (fixes), `ed7a5ba` (`.gitattributes`: `*.sh` stays LF). This request is in the commit after that.
- Diff: `git diff cfc5aa5 ed7a5ba --stat`
- Tests: `.venv\Scripts\python -m pytest -q` gives **111 passed, 2 skipped**. `tests/test_omniforge.py` has 12 tests.
- No GPU reruns. No numbers changed; only wording, schema and code paths changed.

### Fix 1 [P2]: conclusions scoped to the experiment
- `research/omniforge/MEASUREMENTS.md`: the G1 and Hi3D sections are rewritten as "Quan sát" (observation) and "Giả thuyết" (hypothesis). They state:
  - one synthetic probe, posterior mean, octree depth 9;
  - 1.51 mm is about 3.8 times the 0.40 mm noise floor and does not quantify 0.5 mm ridge preservation;
  - the visual statements are qualitative;
  - grid error and latent error are not separated (depth 10 OOM);
  - untested: sampled latents, DiT latents, real scans, a fine-tuned decoder, a tiled finer grid.
- `research/omniforge/PLAN.md`, section 2 item 2 and the G1 row: phrased as a hypothesis with first evidence.
- `vector3d/lab/vae_ceiling.py` module docstring: the universal claim is removed.
- Check: `Select-String research\omniforge\*.md,vector3d\lab\*.py -Pattern 'vô ích|no DiT fine-tune|not the bottleneck|không phải nút thắt|sẽ không tạo ra'` returns nothing.
- Note: TO_CODEX `-01` and my chat report to Ropy used the absolute wording. `-01` is kept as history, and I am correcting the chat report in my next message to Ropy.

### Fix 2 [P2]: reference parameters separate from measurements
- `compare_hand()` returns only measured values. GT generator settings come from `reference_params(p)`, whose docstring says recovered crease, ridge and gap are NOT measured.
- JSON schema v2:
  - `measurements/vae_ceiling.json` has `reference_params` at the top, `runs.<tag>.measured`, `runs.<tag>.runner` and `noise_floor`, plus a `schema_migration` note.
  - `measurements/hi3d_hand_metrics.json` has `reference_params`, `<row>.measured`, `<row>.alignment`, `alignment_note` and `finger_count_note`.
  - The v1 files are kept byte-identical as `vae_ceiling_v1.json` and `hi3d_hand_metrics_v1.json`. The migration moved keys only; no value was changed or added.
- Test `test_reference_params_are_not_reported_as_measurements`: compares a hand generated with ridge=0 and crease=0 against the GT. The measured dict has no crease, ridge or gap keys. The reference still reads ridge 0.5 mm. The measured palm-ridge-region Chamfer is larger than the self-compare.

### Fix 3 [P2]: no stale success
- `vae_ceiling.invoke(cmd, result, mesh)` works as follows:
  - It refuses to start if the result or mesh path already exists (`FileExistsError`).
  - It accepts a run only with exit code 0, a result JSON written after the start, `status == success`, and the mesh present.
  - Anything else returns `status: failed` with the `returncode` and a reason ("no result file written", the child's reason, or "mesh missing"). A timeout is also recorded as a failure.
- `vae_ceiling.run()` writes each invocation into a new `out_dir/run_<time>_<id>/`, so earlier runs are never read or overwritten.
- Tests:
  - `test_invoke_rejects_child_that_dies_before_writing` (child exits 1, writes nothing);
  - `test_invoke_refuses_stale_success` (a pre-existing success JSON);
  - `test_invoke_rejects_nonzero_exit_even_with_success_json`;
  - `test_invoke_accepts_fresh_success`.

### Fix 4: reproducibility
- `scripts/setup_hi3d_env.sh` is the WSL2 environment as built:
  - micromamba with Python 3.10 and conda gcc 12; Triton needs `CC`, set on the run line;
  - torch 2.5.0+cu124 and the flash-attn 2.7.4.post1 wheel;
  - `pip install -c docker/constraints.txt -e .[inference]`.
  - Pinned revisions:
    - Hi3D code `13657ad09afaa92e350b461a1cd66399e44a59c4`;
    - weights `hi3d-official/Hi3Dv1.0@d502e938cbedc6e87aba8884614c5ca6d177a4b6`;
    - `facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`;
    - `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`.
  - Weight SHA256:
    - transformer_512 `8ba88311dccf23463c6e10dd0d3bd5e0c5c35d1608c6f466413b6428b5ad8e56`
    - transformer_1024 `a2ca5e44dde51cacecd59b42fceeb5f14e75d55e6dd047337752c4e6f281008c`
    - vae `605d207c77f8db0b33c567a5861252f0090cf3c3604dfda1cfee09e8f0b53f3e`
- Lock file: `research/omniforge/env/hi3d_pip_freeze.txt`.
- Hi3D hand table, now produced by script:
  - `vae_ceiling.prepare_hi3d_hand(vae_run_dir, body, out_dir)` builds the coarse hand crop and the RGBA condition image;
  - `vae_ceiling.bbox_fit` is the approximate uniform-scale and translation fit (test: `test_bbox_fit_maps_scaled_copy_back`);
  - `vae_ceiling.evaluate_hi3d_hand('bench/omni_hi3d')` writes `hand_metrics.json` and `hand_region.png`.
  - A rerun on the existing GLBs reproduces v1: largest absolute difference 1.8e-15, same finger counts.
- Bbox-fitted Chamfer stays marked approximate, and the finger counter stays marked as a one-slice heuristic, in both the JSON and the report.

### Integrity evidence for the TripoSG env and LoRA (no "before" snapshot exists)
- LoRA path+size fingerprint is `43fd2bfaafbee56a`. That equals the `+lora:43fd2bfaafbee56a` in `runs\lora_check\metadata.json`, recorded before this OmniForge work started.
- `adapter_model.safetensors` sha256 is `04b42669aeed401c545100a3f8cbcd2c0433e119a9261d3bb20c1e2b695b2cba`, recorded now as a baseline for future checks.
- TripoSG env `pip freeze` today: `research/omniforge/env/triposg_env_pip_freeze_2026-10-24.txt` (baseline for future checks).
- `git diff c711c91 HEAD -- vector3d/lab/backends/triposg.py vector3d/lab/backends/triposg_runner.py` is empty.

### Reproduce
```powershell
.venv\Scripts\python -m pytest -q tests\test_omniforge.py
.venv\Scripts\python -c "from vector3d.lab import vae_ceiling as vc; vc.evaluate_hi3d_hand('bench/omni_hi3d')"   # needs the local bench GLBs
```

### Pass criteria
1. pytest is green, and the four `invoke` tests cover dead child, stale success, nonzero exit with a success JSON, and fresh success.
2. No unscoped claims remain in MEASUREMENTS.md, PLAN.md or the vae_ceiling docstring.
3. No measured dict carries GT generator parameters, the v2 JSON values equal v1 apart from the key moves, and v1 is preserved.
4. Hi3D setup and evaluation are reproducible from committed scripts with pinned revisions.

### Still blocked / not done
- SAM 3D Body: Hugging Face access is awaiting author review.
- Open replacement (RTMW whole-body + MHR fitting): just started; the environment install failed on a pin (`onnxruntime-gpu==1.20.1` does not exist for this Python). Not ready for review.
- Human benchmark with real scans: needs Ropy to choose the data source and licence.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.
