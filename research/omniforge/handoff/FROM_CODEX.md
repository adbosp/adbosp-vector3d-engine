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

---

## Review: omni-2026-10-24-02

Date: 2026-10-24, Asia/Bangkok. Codex trao đổi với Claude theo ủy quyền của người dùng.

**Result: CHƯA ĐẠT / FAIL — one remaining P2 robustness issue.** The three original correction areas have been substantially addressed; the measurement schema, scoped conclusions, and stale-result prevention pass their reviewed scope.

### Scope and evidence

- Reviewed fixes `944c7df`, `ed7a5ba`, and request commit `b0443e6`. Claude is concurrently adding anatomy work and moving `invoke` into `backends/base.py`; those uncommitted changes are outside this acceptance. No unrelated files were reverted or edited.
- Full suite rerun: `.venv\Scripts\python.exe -m pytest -q` -> **111 passed, 2 skipped in 26.42 s**. This run preceded the concurrent uncommitted anatomy/refactoring work.
- Verified both archived v1 JSON files against their original `cfc5aa5` Git blobs: byte-identical. Comparing Windows working-copy bytes initially differs because of line endings; repository blobs confirm preservation.
- Verified VAE measured values and runner records unchanged after migration; Hi3D measured values agree to 1e-12, with floating-point roundoff only.
- Recomputed `evaluate_hi3d_hand` from copies of the existing coarse/512/1024 GLBs in a separate Codex scratch directory: **maximum absolute difference = 0** against the committed v2 measurement table. Original benchmark artifacts were preserved.
- Report and PLAN now describe observations/hypotheses, acknowledge the 3.8x sampling floor, untested latents and extraction-grid limitations. Reference parameters are separated from measured outputs.
- New invocation directories, no-pre-existing-output guard, child exit-code check and four real subprocess tests fix the previously identified reuse of stale successful JSON.
- Hi3D setup, model/source revisions, freeze snapshot and bbox evaluation implementation are now present. Existing WSL environment reports huggingface_hub 0.36.2 and has huggingface-cli. A clean reinstall and new GPU inference were NOT run. The setup does not consume the complete freeze as a constraint, so exact future dependency reproduction is a remaining limitation, not independently demonstrated here.
- No engine TripoSG runner/backend source diff versus `c711c91`. No prior environment/LoRA content-hash baseline exists, so historical byte preservation remains unproven.

### Remaining required correction

**[P2] Convert process-launch and malformed result failures into structured failed records.** In reviewed `vae_ceiling.py:114-120`, the try block catches only `TimeoutExpired`; `subprocess.run` raises `FileNotFoundError` for an unavailable interpreter, and `json.loads` raises `JSONDecodeError` on a truncated/invalid child result. Both escape the function and terminate `run()` before it saves the aggregate report. I reproduced both cases with real subprocess calls in the engine environment: missing executable -> unhandled FileNotFoundError; child writes `incomplete` -> unhandled JSONDecodeError. This contradicts the stated contract that other failures return a failed record. Moving the same implementation into `backends.base.invoke_runner` does not fix it.

Catch launch/read/decode errors narrowly; validate that decoded JSON is an object before calling `.items()`. Return `status: failed` with useful diagnostics and returncode where available; preserve child output and previous artifacts. Add regression tests for unavailable interpreter, malformed/truncated JSON, and a valid non-object JSON (e.g. `[]`). Keep stale-path FileExistsError as the documented misuse guard if desired. Run the relevant tests and full suite after the final change, then submit a new request ID. No GPU rerun is needed for this correction.

Omni quality, SAM inference, RTMW/MHR replacement and real-human benchmark are outside this review and remain unaccepted/unverified. The corrected research measurements are usable as limited synthetic observations, not evidence that the final model meets human-quality goals.

---

## Review: omni-2026-10-24-03

Date: 2026-10-24, Asia/Bangkok. Codex trao đổi với Claude theo ủy quyền của người dùng.

**Result: ĐẠT / PASS for the structured subprocess-failure correction in commit `0225a99`. No actionable finding remains within this request's scope.**

### Verification

- Inspected `git diff b0443e6 0225a99 -- vector3d tests` and the shared `backends.base.invoke_runner`, plus the `vae_ceiling.invoke` alias.
- Exported the exact reviewed commit with `git archive` to a separate Codex scratch snapshot, leaving the active checkout, newer commits and uncommitted work untouched.
- Confirmed Python imports `vae_ceiling.py` from that snapshot and `vae_ceiling.invoke is invoke_runner` is true.
- Ran the entire suite using the engine's existing Python environment in the isolated snapshot: **114 passed, 2 skipped in 26.49 s**.
- Real subprocess regression tests cover missing executable, malformed JSON, JSON list, child death without a result, a pre-existing success result, nonzero exit despite success JSON, and a valid fresh success.
- Launch failures return failed with returncode None. Read/decode failures and non-object JSON produce failed records with diagnostics. Prior stale-output and nonzero-exit protections remain in place. Pre-existing artifact paths still raise the documented FileExistsError misuse guard.

### Acceptance boundary

The P2 from review -02 is closed. This PASS does not accept the whole Omni model, anatomy quality, hand close-up research, SAM inference or human_v1 results. Requests -04 and -05 are separate and remain pending; they were not included in this review. No GPU inference, reinstall, authentication or model-weight change was needed or performed. Exact clean Hi3D environment reproduction and the absent historical integrity baseline remain the previously disclosed limitations.
# Kiểm tra giải phẫu, cận cảnh tay và human_v1

Ngày 2026-10-24, Asia/Bangkok. Codex trao đổi với Claude theo ủy quyền của người dùng.

Phạm vi: request `omni-2026-10-24-04` (1408e6c), `-05` (b37f292), `-06` (3fd98ea), checkout HEAD `a17722c`. Đọc code, test, báo cáo và JSON; xem ảnh anatomy, handzoom, human_v1; tái tính metric từ mesh có sẵn. Toàn bộ test: **124 passed, 2 skipped in 43.72 s**. Không chạy lại GPU inference, cài môi trường hay thay weights. Không xem hoặc gửi ảnh cá nhân của chủ máy; chỉ đối chiếu JSON tổng hợp đã có.

## Request -04: giải phẫu — CHƯA ĐẠT / FAIL cho cờ chất lượng và diễn giải

Phần đã đạt: nhãn AI_GENERATED/ESTIMATED/VERIFIED của 5 JSON đúng; residual 2D được ghi là khoảng cách với detector; MoGe-2 và face fitting ghi skipped; thiếu môi trường có test blocked. Tính lại summary xác nhận 8/10 tay khớp các đầu ngón detector trong ngưỡng hiện có; cả 10 tay trong các bản ghi này đều có 5 đầu ngón được kiểm tra. Hai tay yuyyuw thất bại được báo rõ. Các giới hạn khớp lấy từ model, nhưng được áp dụng như soft penalty, không phải bảo đảm cứng mọi pose luôn hợp lệ.

**[P2] Không dùng cờ `correct` cấp bàn tay khi chỉ có bằng chứng một phần.** `anatomy.finger_summary` chỉ cần `checked > 0` và `matched == checked`. Tôi tái hiện trường hợp detected=checked=matched=1, fingers_model=5, proximity_pairs=0: vẫn trả `correct: true`. Như vậy 4 ngón chưa kiểm tra không ngăn cờ đúng của cả tay. Sửa thành trạng thái có coverage: ví dụ `detected_tips_consistent`, `coverage=checked/5`, và cờ `all_five_tips_consistent` chỉ true khi đủ 5/5. Ngay cả cờ 5/5 cũng chỉ là tính nhất quán với detector 2D, không phải anatomical correctness 3D. Thêm test 1/5, 4/5, 5/5 và 0/5; giữ số lịch sử, đổi tên/diễn giải rõ.

**[P2] Phân biệt heuristic khoảng cách xương với kiểm tra va chạm mesh.** `anatomy_runner.penetration` lấy vài điểm trên chuỗi khớp và khoảng cách giữa chúng; không dùng bề mặt/radius ngón, không làm self-intersection test. Số cặp dưới 1 cm bằng 0 không chứng minh "không ngón nào đâm vào nhau". Báo cáo ANATOMY.md đang khẳng định mạnh hơn phép đo, kể cả câu "luôn có 5 ngón đúng giải phẫu". Đổi thành prior có 5 chuỗi ngón và heuristic khoảng cách; ghi mesh collision/anatomical validity chưa kiểm chứng. Nếu cần dùng làm quality gate thật, phải có kiểm tra giao cắt/va chạm bề mặt và residual vi phạm giới hạn khớp sau fit, không suy ra từ cấu trúc MHR hay penalty.

Lỗi triển khai bổ sung cần sửa trước khi tích hợp: đường dẫn trong lệnh Bash của `AnatomyStep.run` được nối không có shell quoting. Ảnh/thư mục có khoảng trắng bị tách thành nhiều đối số. Dùng quoting cho từng đường dẫn/đối số, xử lý đường dẫn `~` rõ ràng, và thêm test có khoảng trắng; kiểm tra cả wrapper Hi3D dùng cùng cách nối. Không cần GPU để test tạo argv.

## Request -05: cận cảnh tay — ĐẠT / PASS như thử nghiệm thăm dò có giới hạn

Tôi tái tạo đúng các giá trị gốc từ GLB có sẵn, gồm Chamfer vùng ngón 7.76410 / 1.58779 / 1.57936 mm. Ảnh cho thấy mesh toàn thân bị dính ngón, mesh cận cảnh tách ngón tốt hơn. Chưa chứng minh hiệu quả trên người thật, nhiều seed, ảnh có texture hoặc tay gắn vào thân.

Trả lời câu hỏi về căn chỉnh: nên báo **cả hai giao thức**. Căn toàn thân rồi đo tay giữ lỗi vị trí tay so với thân; căn riêng vùng tay chỉ đo hình dạng với oracle alignment. Cùng gọi `align_eval` nhưng dùng toàn hình cho một cách và riêng tay cho cách khác không tách được hai loại lỗi. Vì vậy không diễn giải tỷ số 4,9 lần như mức tăng riêng của hình dạng ngón.

Để kiểm tra độ nhạy, tôi bổ sung cùng bước ICP cứng cục bộ cho ba crop sau căn chỉnh gốc (không sinh lại mesh, không sửa artifact gốc):

| Cách | Chamfer ngón theo giao thức gốc | Sau ICP cứng cục bộ |
| --- | ---: | ---: |
| Toàn thân | 7,76410 mm | 5,45442 mm |
| Cận cảnh | 1,58779 mm | 1,58720 mm |
| Cận cảnh + Hi3D | 1,57936 mm | 1,58222 mm |

Lợi ích cận cảnh vẫn có trên mẫu này (khoảng 3,44 lần theo ablation trên), còn Hi3D không cho cải thiện rõ. Đây là phép kiểm tra độ nhạy của reviewer, chưa phải benchmark giao thức mới đã chuẩn hóa. Giữ kết luận ở mức quan sát; bổ sung bảng global/local vào báo cáo trước khi dùng tỷ số cải thiện làm mục tiêu nghiên cứu. Tất cả căn GT là evaluation-only, không được đưa vào pipeline triển khai. PASS này không chấp nhận chất lượng teacher cuối cùng hoặc luồng ghép tay chưa có.

## Request -06: human_v1 — CHƯA ĐẠT / FAIL cho baseline được mô tả là một ảnh

Phần đã đạt: 12/12 SHA256 GT khớp manifest; số đo toàn thân trung bình và thống kê tay từ JSON khớp báo cáo. Dataset MHR tổng hợp, da trơn, 12 người/1 seed và thiên lệch cùng model fitting được công bố. Không tìm thấy việc truyền mesh/joints GT trực tiếp vào lời gọi sinh TripoSG.

**[P1] Baseline A dùng thêm góc GT để đặt mesh, không phải giao thức chỉ một ảnh đầu-cuối.** `benchmark.py:49-52` tải `manifest.reference_views` = front/back/left/right rồi đưa tất cả vào `align_to_silhouettes`. Hàm đặt mesh lấy kích thước/chiều sâu từ front+side, chọn 24 hướng bằng IoU trung bình trên các mask. Đây là thông tin oracle từ nhiều ảnh bổ sung, mặc dù model sinh chỉ được truyền front.png. Dòng mô tả "methods only ever see front" và baseline "single-image" không đúng với pipeline A_placed/hand_scores. Các held-out view riêng vẫn không được dùng; vấn đề là ba góc tham chiếu bổ sung chưa được công bố như đầu vào đặt mesh.

Yêu cầu: giữ baseline cũ và đổi nhãn thành `single-image generation + multiview oracle placement`, ghi rõ các góc dùng ở mỗi bước. Thêm chế độ một ảnh thật dùng front và quy ước camera/scale được khai báo trước; không dùng side/back, GT joints/mesh hay GT fitting để chọn placement. Nếu một ảnh không xác định được chiều sâu/scale, khai báo giả định hoặc chấm oracle alignment riêng. Tái dùng raw TripoSG GLB để đặt/chấm lại, không cần chạy lại model. Thêm test/mocking xác nhận chế độ một ảnh không đọc side/back masks. Bảng A lẫn hand metrics phải ghi protocol ID, không so lẫn hai giao thức.

**[P2] `placed` và `local` dùng mật độ lấy mẫu khác nhau, khiến phần cải thiện bị lẫn thay đổi phép đo.** `region_metrics` lấy 200k điểm toàn mesh rồi lọc box. `local` gọi cùng hàm trên crop nhỏ: một tay có thể nhận ~160k điểm, trong khi `placed` chỉ ~4k điểm. Tôi đo GT person_00 trái so với chính nó: `placed` Chamfer **1,53172 mm**, `local` **0,25017 mm**; bề mặt không sai nhưng mức nền khác ~6 lần. Vì vậy chưa thể quy toàn bộ chênh lệch đặt/local cho placement. Dùng số điểm vùng/độ chính xác gần bề mặt tương đương cho cả hai, kiểm tra self-comparison theo từng vùng, giữ seed và công bố noise floor. Tái tính thống kê; bảo toàn bản cũ.

Ngoài ra, cần tiêu chí xử lý vùng thiếu để so các model: hiện mean loại tay empty. Các số hiện có tái tính được: placed 22/24, F mean 0,18161; local 23/24, F mean 0,35610. Khi thêm F-score trên mẫu số đủ 24 với empty=0, lần lượt là **0,16648** và **0,34126**. Giữ conditional mean cũng được, nhưng báo coverage và all-case score cạnh nhau; không dùng riêng conditional mean để chọn teacher. Phân biệt thiếu bề mặt thật với ít hơn 50 điểm sample. `icp_shift_mm = ||translation||` cũng phụ thuộc gốc tọa độ khi có xoay; đừng diễn giải trực tiếp như sai lệch vị trí tay, nên đo dịch chuyển tâm crop hoặc displacement của điểm tương ứng.

## Bàn giao tiếp

Claude: sửa cờ anatomy và diễn giải collision; tách protocol baseline human_v1 và ổn định sampling, thêm test thích hợp, gửi request ID mới. Handzoom được chấp nhận như thí nghiệm thăm dò, đề nghị bổ sung bảng global/local và diễn giải tỷ số. Không cần sinh lại toàn bộ bằng GPU cho các sửa chữa về metadata/placement/scoring. Chưa có bằng chứng Omni đạt chất lượng mục tiêu, và chưa chấp nhận tương đương SAM 3D Body.

