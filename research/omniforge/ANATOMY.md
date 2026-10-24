# Bước giải phẫu (anatomy): RTMW + MHR, ứng viên mở thay SAM 3D Body (2026-10-24)

**Đây không phải bản tương đương SAM 3D Body.** Chưa có benchmark cùng điều kiện (SAM 3D Body vẫn bị chặn). Mọi kết quả dưới đây là quan sát trên 5 ảnh, chưa phải kết luận.

## Cách làm
- **RTMW-x whole-body** (dự án RTMPose của open-mmlab, Apache-2.0): 133 điểm khớp, gồm 42 điểm hai bàn tay, 6 điểm bàn chân, 68 điểm mặt. Chạy bằng `rtmlib` 0.0.16 trên onnxruntime-gpu (CUDA), detector người là YOLOX-m. Đã ghi SHA256 của các file ONNX trong `scripts/setup_anatomy_env.sh`.
- **MHR** (Meta Momentum Human Rig, code và asset Apache-2.0, bản release v1.0.1, `assets.zip` có sha256 `e4f4f205…`): 127 khớp, mỗi ngón có đủ chuỗi khớp, 18.439 đỉnh (LOD1). Chạy trên CPU torch.
- **Khớp mô hình** (`vector3d/lab/backends/anatomy_runner.py`), tối ưu Adam 600 bước, 2 khởi tạo (quay mặt vào hoặc quay lưng với camera), lấy cái có sai số thấp hơn. Hàm mục tiêu gồm:
  - sai số chiếu 2D (camera trực giao), trọng số là bình phương độ tin cậy của RTMW; thân trước, ngón tay từ bước thứ 1/3 trở đi;
  - **giới hạn khớp của chính MHR** (198 giới hạn min/max lấy từ model, không tự bịa). Đây là **phạt mềm**: sau khi khớp vẫn còn vi phạm nhỏ, đã đo lại từ `fit.npz` (`anatomy/joint_limit_violations.json`): 2–30 tham số vượt giới hạn, mức vượt lớn nhất 0,0002–0,0016 (radian với tham số xoay, đơn vị model với tham số tỉ lệ);
  - phạt khi trục xương của hai ngón khác nhau gần nhau dưới 1,2 cm. Đây là heuristic trên điểm của khung xương, **không phải kiểm tra va chạm bề mặt mesh**;
  - prior tư thế và tỉ lệ cơ thể.
  - **Giả định "đứng"** (bật mặc định, ghi trong `assumptions` của kết quả): thân và chân gần một mặt phẳng chiều sâu. Không có giả định này, ảnh 2D cho phép cả người ngả về phía camera (đã thấy trên ảnh uihghj).
- **Nhãn:** điểm khớp `AI_GENERATED`; tư thế và mesh khớp được `ESTIMATED`; sai số chiếu `VERIFIED` (chỉ là khoảng cách đo được giữa kết quả khớp và điểm phát hiện). `anatomy.check_labels` từ chối kết quả nếu phần khớp được tự gắn `VERIFIED`.
- **Bỏ qua (ghi `skipped`):** ràng buộc chiều sâu MoGe-2 (chưa cài) và khớp khuôn mặt (chưa có ánh xạ điểm mặt sang MHR).
- Engine gọi qua `vector3d/lab/anatomy.py` (`AnatomyStep`), dùng WSL2 và env riêng. Thiếu env thì trả `blocked`, không bao giờ trả mesh thay thế.

## Kết quả đo (RTX 3060, WSL2)

| ảnh | thời gian tổng | RTMW | GPU (delta) | khớp MHR (CPU) | sai số thân | sai số tay | kích thước người | ngón trái | ngón phải |
|---|---|---|---|---|---|---|---|---|---|
| uihghj (hoạt hình, hai tay dang) | 49,1 s | 0,97 s | 916 MB | 37,6 s | 39,1 px | 6,3 px | 962 px | 5/5 ✓ | 5/5 ✓ |
| yywdsd (T-pose, găng tay) | 44,9 s | 0,54 s | 924 MB | 38,6 s | 8,2 px | 1,9 px | 595 px | 5/5 ✓ | 5/5 ✓ |
| yuyyuw (tay ôm tai nghe, chỉ nửa người) | 43,6 s | 0,61 s | 910 MB | 37,3 s | **414,8 px** | **59,3 px** | 1393 px | 3/5 ✗ | 2/5 ✗ |
| jkghed (chibi) | 43,6 s | 0,53 s | 912 MB | 37,5 s | 15,7 px | 6,6 px | 576 px | 5/5 ✓ | 5/5 ✓ |
| ảnh chụp của Ropy (không commit ảnh) | 43,2 s | 0,56 s | 906 MB | 37,1 s | 55,6 px | 5,7 px | 1342 px | 5/5 ✓ | 5/5 ✓ |

Cách đọc bảng:
- "Ngón x/5": số đầu ngón **phát hiện được** mà đầu ngón của MHR sau khi chiếu nằm trong 0,35 × kích thước bàn tay.
- ✓ (bản đầu, theo review Codex `-04`) chỉ nghĩa là **nhất quán với detector 2D**: 5/5 đầu ngón được phát hiện và đều khớp, và không cặp trục ngón nào gần nhau dưới 1 cm. Không phải độ đúng giải phẫu 3D, vì không có GT.
- `anatomy.finger_summary` hiện báo riêng từng thứ: `coverage` (số đầu ngón kiểm được chia 5), `detected_tips_consistent` (mọi đầu ngón kiểm được đều khớp), `all_five_tips_consistent` (chỉ đúng khi 5/5), và `anatomical_correctness: not measured`. Cả 8 bàn tay ✓ trong bảng đều có coverage 5/5.
- "GPU (delta)" là mức tăng bộ nhớ GPU toàn máy (`nvidia-smi`) quanh bước RTMW. Bước khớp MHR chạy trên CPU.

Ảnh: `anatomy/<tên>_hands.png` (ảnh gốc cắt quanh tay, MHR nhìn thẳng và nhìn chéo 60°) và `anatomy/<tên>_overlay.jpg` (xanh: điểm RTMW; đỏ: khớp MHR chiếu lên ảnh).

## Quan sát (chưa phải kết luận)
- Ở 8/10 bàn tay, cả 5/5 đầu ngón phát hiện được đều khớp với detector, và heuristic trục xương không thấy cặp ngón nào gần nhau dưới 1 cm. Va chạm bề mặt mesh **chưa được kiểm**. Sai số chiếu vùng tay là 1,9–6,6 px.
- Về cấu trúc, MHR luôn có 5 chuỗi ngón, kể cả khi ngón bị che (yuyyuw). Đây là một prior có thể giúp "tưởng tượng ra bàn tay", nhưng tính đúng giải phẫu 3D của từng tư thế **chưa được kiểm chứng**. Tư thế ngón chỉ đúng tới mức điểm 2D cho phép, và chưa có chiều sâu thật.
- **Thất bại:** ở yuyyuw (chỉ thấy nửa người, tay nắm vật), phần thân khớp sai hẳn (415 px) nên kéo theo tay sai. Cần chế độ "nửa người" và ràng buộc chiều sâu.
- **Giới hạn:** nhân vật hoạt hình có tỉ lệ ngoài phạm vi cơ thể người (chân rất ngắn). Thân khớp sai 39 px và chân bị dạng ra (ảnh uihghj). MHR là model người thật, nên không nên dùng nó để đo nhân vật cách điệu.
- Mất khoảng 43–49 s mỗi ảnh, trong đó khoảng 37 s là bước khớp trên CPU (2 khởi tạo × 600 bước). Chưa tối ưu tốc độ.

## Chưa làm
- So sánh với SAM 3D Body cùng điều kiện (bị chặn).
- Đo trên dữ liệu có GT (human_v1, xem PLAN.md).
- Ràng buộc chiều sâu (MoGe-2), khớp mặt, chế độ nửa người.
