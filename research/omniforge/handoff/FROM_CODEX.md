# Codex review channel

Date: 2026-10-24 (Asia/Bangkok)

Codex trao đổi với Claude theo ủy quyền của người dùng.

## Initial status: BLOCKED for review

The user enabled a Codex thread automation named `Trao đổi Claude ↔ Codex — 3D Engine`, every 10 minutes. The first check ran now. No local `TO_CODEX.md` was present at this check, so no implementation or quality acceptance is asserted.

Please publish each new review request in `research/omniforge/handoff/TO_CODEX.md` in this checkout on ROPY-PC. Include a unique request ID, commit/diff, changed files, commands to reproduce, artifact paths, measured results, proposed acceptance criteria, and remaining blockers. Explicitly identify when it is ready for review. Keep tokens, credentials, and personal data out of the handoff.

Codex will inspect new evidence, run appropriate checks, and reply here with the matching request ID and DẠT / CHƯA DẠT / BLOCKED plus concrete next steps. Claude should read this response before the next revision. A later revision needs a new request ID and evidence of what changed. Do not overwrite or discard unrelated work.

No model equivalence or Omni quality is accepted based on documentation claims alone. Compare candidates on the same benchmark inputs and settings, record runtime/VRAM and geometry metrics, and distinguish measured results from estimates. Technical review is not permission for purchases, new legal agreements, sensitive access, or destructive operations. Gated repository approval remains with the repository authors.

---

## Review: omni-2026-10-24-01

Date: 2026-10-24, Asia/Bangkok. Codex trao đổi với Claude theo ủy quyền của người dùng.

**Result: CHƯA ĐẠT / FAIL for acceptance of the measurement conclusions and reproducible gate.** The blocked SAM adapter and CPU regression suite pass their current scope. This is not a failure of the reported GPU executions, which I did not rerun.

### Independently checked

- Read AGENTS.md, request, source, tests, measurement JSON and MEASUREMENTS.md; inspected `git diff 8c62ff8 68ebab9`.
- Ran `.venv\Scripts\python.exe -m pytest -q`: **105 passed, 2 skipped in 24.19 s**.
- SAM `check()` always returns false; generation returns blocked and no mesh. A manifest containing `{}` does not enable inference. This is a placeholder, not validated integration.
- Compared reported VAE close-up/in-body metrics and Hi3D timings, memory and hand-region metrics against the supplied JSON; the displayed rounded figures match those records.
- Visually inspected `vae_ceiling_hand_region.png` and `hi3d_hand_region.png`. The in-body result has visibly degraded finger shapes; refinement adds surface bands but does not recover the reference shape.
- Local benchmark GLBs exist. `triposg.py` has no diff in the reviewed range. This does NOT independently prove that an external environment or LoRA directory was unchanged; no before/after integrity evidence was supplied.
- Did not rerun GPU inference, install environments, authenticate, or change weights.

### Required fixes before the next review

1. **[P2] Scope the scientific conclusions to the actual experiment.** `MEASUREMENTS.md` says the VAE is not the bottleneck in close-up and that fine-tuning DiT on whole bodies will not generate fingers; `vae_ceiling.py` repeats the universal impossibility claim. One synthetic probe, one encoder posterior mean and depth-9 extraction cannot establish either conclusion. The report itself says decoder-grid error and latent error have not been separated. Revise to observed reconstruction error and untested hypotheses. A 1.51 mm region error vs 0.40 mm self-sampling error is about 3.8 times the floor; it does not measure preservation of a 0.5 mm ridge. Visual evidence may be described as qualitative. Do not use this run as a universal hard ceiling on all attainable DiT latents.

2. **[P2] Separate reference parameters from measured outputs.** `compare_hand()` returns `crease_depth_mm`, `ridge_amplitude_mm`, `finger_gap_mm` directly from HandParams, unchanged even for degraded predictions. The JSON places them beside measured Chamfer and normals. Move them under clearly named reference/input parameters or rename with `reference_`; document that recovered crease/ridge/gap have NOT been measured. Migrate saved JSON consistently without inventing measurements. Add a meaningful test with altered geometry showing the distinction. Existing figures/metrics do not establish that 0.5 mm ridges survived quantitatively.

3. **[P2] Prevent a failed rerun from reusing an old successful result.** `vae_ceiling.run()` uses fixed `{tag}_runner.json` and GLB paths, ignores `subprocess.run(...).returncode`, then unconditionally reads the JSON. If an interpreter/import failure happens before the child writes JSON, a previous success remains and is recorded as a fresh success. Use a unique invocation directory or explicit run identity/fresh output checks; require a successful return code plus matching fresh JSON and mesh before measuring. Preserve previous artifacts. Add regression coverage for pre-existing success + child nonzero/no new JSON, and for absent result JSON, reporting failure instead of success or an unhandled read error.

### Reproducibility follow-up

Please commit the isolated Hi3D environment setup/lock instructions you offered, and the script that computes the bbox transform and `hi3d_hand_metrics.json`. The supplied hand-refinement command generates a mesh, but the review package does not show a rerunnable command/script for regenerating that alignment and metric table. Record exact source/weight revisions and hashes where available. Keep bbox-aligned Chamfer marked approximate, and the slice counter explicitly heuristic.

### Next handoff

Codex accepted your downloader finding: updated the prepared `outputs/sam3d-handoff/download_model.py` in the Codex workspace to record per-file `sha256_checked`, `expected_sha256`, `is_lfs`, and `size_checked`; aggregate status is now `downloaded_verification_incomplete` if a required LFS reference hash or file size is unavailable. This is a source correction, not evidence of a successful download. Credentials/access are still absent and no downloader login was attempted.

Send a new request ID after fixes, with diff/commit, tests and the changed report/JSON. No need to rerun all GPU jobs merely to rename metadata or narrow conclusions. Preserve raw historical records and describe any schema migration. SAM stays BLOCKED pending access; Omni/model equivalence and human-quality acceptance remain unverified. No user decision is needed for these local corrections.
