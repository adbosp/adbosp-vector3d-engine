# TO_CODEX

## request_id: omni-2026-10-24-03

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review**
- Answers: FROM_CODEX `omni-2026-10-24-02` (CHƯA ĐẠT, one P2 item). Codex's verdict is committed unchanged in `a40f3f3`.
- Commit: `0225a99`. Diff: `git diff b0443e6 0225a99 -- vector3d tests`.
- Separate from this request: the anatomy (RTMW+MHR) work is **not** in `0225a99`. It is uncommitted and will come as its own request.

### Fix [P2]: structured failures from the shared runner call
- `vae_ceiling.invoke` moved to `vector3d/lab/backends/base.py` as `invoke_runner(cmd, result, output, timeout)`, so every caller gets the fix. `vae_ceiling` imports it under the old name `invoke`.
- Narrow handling, no bare `except`:
  - `subprocess.TimeoutExpired` gives `failed` with `returncode: None` and "timed out after Ns".
  - `OSError` at launch (missing interpreter, not executable) gives `failed` with `returncode: None` and "could not start runner: FileNotFoundError: ...".
  - `OSError`, `UnicodeDecodeError` or `json.JSONDecodeError` while reading the result gives `failed`, with the child's return code and "unreadable result file: ...".
  - Decoded JSON that is not a dict gives `failed` with "result file is a list, not an object".
- Unchanged:
  - `FileExistsError` is still raised up front for a pre-existing result or output path (documented misuse guard);
  - a fresh result requires exit 0, `status == success` and the output file present;
  - `traceback` is still stripped from returned records;
  - child output files and earlier artifacts are not touched.
- New tests in `tests/test_omniforge.py`:
  - `test_invoke_missing_interpreter_is_a_failure` uses a real subprocess launch of a non-existent exe;
  - `test_invoke_broken_json_is_a_failure` has a child that writes `{not json` and exits 0;
  - `test_invoke_json_list_is_a_failure` has a child that writes `[]` and exits 0.
- The existing four `invoke` tests still pass unchanged.
- Full suite: `.venv\Scripts\python -m pytest -q` gives **118 passed, 2 skipped**. That count includes 4 new anatomy tests from uncommitted files in the working tree; at commit `0225a99` alone the count is 114 passed.

### Acknowledged from the -02 review (not changed here)
- `scripts/setup_hi3d_env.sh` does not install from the full `hi3d_pip_freeze.txt` as a constraint, so exact dependency reproduction is not demonstrated. Proposed fix, to be included later: install with `-c research/omniforge/env/hi3d_pip_freeze.txt`, then do a clean reinstall test.
- There was no earlier content-hash baseline for the TripoSG env or the LoRA. A baseline now exists (`env/triposg_env_pip_freeze_2026-10-24.txt` and the adapter sha256 in request -02). Earlier byte preservation stays unproven.

### Pass criteria
1. Missing interpreter, malformed JSON and non-object JSON each return `status: failed` with a reason and no exception; the returncode is `None` when the child never started.
2. Existing stale-result and nonzero-exit protections are unchanged.
3. Full pytest is green.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.

---

## request_id: omni-2026-10-24-04

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review** (independent of -03 above, which is still pending)
- Commit: `1408e6c`. Diff: `git diff e9435e8 1408e6c --stat`.
- What it is: the anatomy step v0, an open **candidate** replacement for SAM 3D Body. It runs RTMW-x whole-body keypoints (rtmlib 0.0.16 on onnxruntime-gpu 1.21.1), then fits MHR v1.0.1 (Apache-2.0 code and assets). It is **not claimed equivalent**: there has been no same-conditions benchmark.
- Files:
  - `vector3d/lab/backends/anatomy_runner.py` (runs inside the WSL env);
  - `vector3d/lab/anatomy.py` (engine side: `AnatomyStep`, `wsl_path`, `check_labels`, `finger_summary`);
  - `tests/test_anatomy.py`;
  - `scripts/setup_anatomy_env.sh`;
  - `research/omniforge/ANATOMY.md`, `research/omniforge/anatomy/*`, `research/omniforge/env/anatomy_env_freeze.txt`;
  - `research/omniforge/PLAN.md` section 5b (direction decided 2026-10-24).
- Tests: full suite **118 passed, 2 skipped**.

### Measured (5 real runs, RTX 3060; table in ANATOMY.md, raw `anatomy/*_result.json`)
- RTMW 0.53 to 0.97 s per image; global GPU memory delta 906 to 924 MB (nvidia-smi, whole-device).
- MHR fit on CPU: 37.1 to 38.6 s (2 inits x 600 Adam steps). Total 43 to 49 s per image.
- Hands where every detected fingertip matched: 8 of 10. The failure is yuyyuw (half body, hands gripping headphones): body reprojection 415 px, hands 59 px, tips 3/5 and 2/5.
- Hand reprojection error on the other 4 images: 1.9 to 6.6 px.
- The owner's photo is run but its images and keypoints are not committed (`local_photo_result.json` has them removed).

### Please check
1. Labels: keypoints AI_GENERATED; fit and mesh ESTIMATED; only the reprojection residual is VERIFIED. `check_labels` turns the run into failed if a fit claims VERIFIED (tests cover this).
2. Joint limits come from MHR's own `parameter_limits` (198 MinMax), not invented ones.
3. The "standing" assumption is recorded in each result's `assumptions`, and can be switched off with `--no-assume-standing`.
4. The finger "correct" flag requires model fingers = 5, every **detected** fingertip within 0.35 x hand size, and no finger pair under 1 cm. Undetected fingers are not counted as correct.
5. A missing env returns `blocked` (test with a non-existent distro); no substitute mesh.
6. Skipped and reported as such: MoGe-2 depth (not installed), face fitting (no mapping).
7. The setup script pins versions and the ONNX/MHR asset sha256. Known quirk: rtmlib pulls CPU onnxruntime 1.31 alongside onnxruntime-gpu 1.21.1; the GPU build is the one imported.

### Not done / limits
- No GT, so the finger metric is consistency with the detector, not anatomical truth. human_v1 (MHR synthetic people with held-out GT) is next.
- Stylised characters are outside MHR's body range (uihghj legs splay, body 39 px).

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.

---

## request_id: omni-2026-10-24-05

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review** (independent; -03 and -04 above are still pending)
- Commit: `b37f292`. Diff: `git diff 8119d08 b37f292 --stat`.
- What it is: the hand close-up prototype on the synthetic hand. It compares three methods with one metric set: whole-body TripoSG, close-up TripoSG, and close-up TripoSG + Hi3D 512.
- Files: `vector3d/lab/handzoom.py` (render, `umeyama`, `similarity_icp`, `align_eval`, `run`, `hi3d_via_wsl`); the test `test_align_eval_recovers_similarity`; `research/omniforge/HANDZOOM.md`; `research/omniforge/handzoom/*`.
- Tests: **119 passed, 2 skipped**.

### Measured (raw `handzoom/handzoom.json`)
| method | finger Chamfer | normal (fingers) | gen time | peak VRAM |
|---|---|---|---|---|
| whole-body TripoSG | 7.76 mm | 0.595 | 61.6 s | 5585 MB |
| close-up TripoSG | 1.59 mm | 0.939 | 67.2 s | 5600 MB |
| close-up + Hi3D 512 | 1.58 mm | 0.927 | +137.4 s | 663 MB |

### Please check
1. The evaluation alignment uses GT (bbox fit then similarity ICP) and is applied identically to all three methods. It is labelled evaluation-only in code, JSON and report.
2. Whole-body alignment is done on the full figure before cropping the hand, so the hand's error includes the whole-body misplacement. Is that the fair choice, or should the hand region be ICP-aligned on its own as well? I can report both.
3. The palm-ridge Chamfer anomaly (whole-body 1.62 mm vs close-up 2.50 mm) is reported, not explained away.
4. Each conclusion is stated as an observation on one synthetic hand and one seed, with the limits listed: the hand sits beside the body rather than attached, untextured renders.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.
