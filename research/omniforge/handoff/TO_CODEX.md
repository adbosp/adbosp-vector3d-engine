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
