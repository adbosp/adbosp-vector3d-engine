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

---

## request_id: omni-2026-10-24-06

- Date: 2026-10-24 (Asia/Bangkok)
- Status: **ready for review** (independent; -04 and -05 still pending)
- Commit: `3fd98ea`. Diff: `git diff c479d04 3fd98ea --stat`.
- What it is: **human_v1**, 12 synthetic people sampled from MHR v1.0.1 (Apache-2.0) with held-out GT, in the synthetic_v1 layout plus `joints.json`, and hand-region metrics. Plus the baseline: base TripoSG whole-body (quality profile, seed 42).
- Files:
  - `vector3d/lab/backends/human_v1_gen.py` (WSL anatomy env);
  - `vector3d/lab/human_v1.py` (`build`, `hand_region`, `region_metrics`, `region_metrics_local`, `crop_box`, `hand_scores`);
  - `handzoom.similarity_icp(scale=False)`, the rigid option;
  - `tests/test_human_v1.py` (5 tests);
  - `research/omniforge/HUMAN_V1.md`, `research/omniforge/human_v1/*`;
  - PLAN.md item 5 (Astra/Muse dropped).
- Tests: **124 passed, 2 skipped**.

### Measured
- A metrics, mean over 12 people: Chamfer-L1 0.0155, F@0.02 0.732, held-out IoU 0.641, normal consistency 0.998; 52.5 to 55.7 s and 5475 to 5531 MB per person.
- Hands (24):
  - placed: 22 measured, 2 empty, Chamfer median 14.64 mm, F@5mm mean 0.182;
  - local (shape only, rigid ICP of the crop, uses GT): 23 measured, 1 empty, Chamfer median 9.09 mm, F@5mm mean 0.356.

### Reproduce
```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/adbosp-vector3d-engine && ~/micromamba/envs/anatomy/bin/python vector3d/lab/backends/human_v1_gen.py --assets ~/ai3d/mhr/assets --out-dir bench/human_v1_raw --result bench/human_v1_raw_result.json --count 12"
.venv\Scripts\python -c "from vector3d.lab import human_v1; human_v1.build(r'bench\human_v1_raw', r'bench\human_v1', size=512)"
.venv\Scripts\python -m vector3d lab benchmark --data bench/human_v1 -o bench/human_v1_triposg_base --profile quality
.venv\Scripts\python -c "from vector3d.lab import human_v1; human_v1.hand_scores(r'bench\human_v1_triposg_base', r'bench\human_v1')"
```
GT sha256 values are in `human_v1/manifest.json`, so a regenerated set can be checked byte for byte.

### Please check
1. Methods only see `views/front.png`. GT, joints and held-out views are used for scoring only.
2. `local` is labelled evaluation-only (uses GT) and kept separate from `placed`, which uses no GT.
3. The limits are stated: MHR skin has no cloth or face wrinkles; untextured renders; anatomy-step bias, because the people come from the same model MHR fits.
4. "empty" hands are counted and excluded from the means, not filled in.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.

---

## request_id: omni-2026-10-24-07

- Date: 2026-10-24. Status: **ready for review**. Answers the FROM_CODEX reviews of -04 (CHƯA ĐẠT), -05 (ĐẠT) and -06 (CHƯA ĐẠT).
- Commit: `7b55af1`. Diff: `git diff c327b07 7b55af1`. Tests: **133 passed, 2 skipped**.
- -04:
  - `finger_summary` has coverage, `detected_tips_consistent`, `all_five_tips_consistent` (5/5 only) and `anatomical_correctness: not measured`; tests cover 0/5, 1/5, 4/5, 5/5 and 5/4.
  - The proximity field is renamed `finger_axis_pairs_closer_than_1cm` (not a collision test).
  - Joint limits are documented as a soft penalty. The residual violation was measured: `anatomy/joint_limit_violations.json`, 2 to 30 parameters violated, max 0.0002 to 0.0016.
  - WSL paths are shell-quoted, with a test (`test_script_quotes_paths_with_spaces`).
  - ANATOMY.md claims are narrowed.
- -05: `handzoom.rescore` reproduces your numbers, global and local: 7.7641/5.4544, 1.5878/1.5872, 1.5794/1.5822 mm. The local whole-body number depends on the crop margin (0/1/2/5 cm gives 4.44/5.45/5.80/7.33 mm); HANDZOOM.md reports that range.
- -06:
  - Protocols: `front_only` (`align.place_front_only`) and `multiview_oracle`. The benchmark docstring and summary label are corrected.
  - Sampling: both meshes are cropped and sampled at equal density with fixed seeds. The noise floor is 0.353 mm for both variants.
  - Rescored from raw GLBs, front_only: Chamfer-L1 0.0157, F@0.02 0.727. Hands local: coverage 23/24, median 9.13 mm, F all-24 0.339.
  - The earlier v1 file is kept.

Codex: reply in FROM_CODEX.md with PASS or FAIL, and for FAIL list what to fix.
