# OmniForge: đo đạc khởi đầu (2026-10-24, ROPY-PC, RTX 3060 12 GB)

Mọi con số dưới đây lấy từ lần chạy thật. File gốc nằm trong `measurements/`.
Sửa lần 2 (theo review Codex `omni-2026-10-24-01`):
- Các kết luận được thu hẹp đúng phạm vi thí nghiệm.
- Tham số GT tách khỏi số đo.
- Schema JSON lên v2. Bản v1 vẫn giữ trong `*_v1.json`, không đổi giá trị nào.

## 0b. TripoSG không giảm số mặt (character, seed 42)

| profile | số mặt | Chamfer-L1 | F@0.02 | held-out IoU | normal cons. | thời gian | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| balanced (giảm còn 50k mặt) | 50.000 | 0.007753 | 0.9658 | 0.8632 | 0.9983 | 60.9 s | 5571 MB |
| quality (giữ nguyên) | 612.532 | 0.007762 | 0.9653 | 0.8631 | 0.9997 | 65.5 s | 5571 MB |

Quan sát: trên benchmark này không đo được khác biệt. Benchmark không đủ để kết luận về chi tiết nhỏ, vì mesh chuẩn của character chỉ có 25.780 mặt. Cần benchmark Human (GĐ 0 trong PLAN.md).

## G1. Thử nghiệm tái tạo qua VAE TripoSG (một mẫu bàn tay tổng hợp)

Thiết lập:
- Mẫu đo là bàn tay phải dựng từ SDF (`vector3d/lab/humanoid.py`). Đây là mẫu tổng hợp, không phải scan người thật.
- Tham số GT của bộ sinh (`reference_params`, là **đầu vào**, không phải số đo): khe ngón 4 mm, nếp khớp 1,2 mm, vân lòng bàn tay 0,5 mm.
- Encode 200.000 điểm kèm normal bằng VAE TripoSG (fp16, **mean** của posterior, không sample), decode bằng flash decoder với octree depth 9.
- Ngưỡng nhiễu của phép đo (bàn tay so với chính nó): Chamfer 0,40 mm, normal agreement 0,997.

| cảnh | token | Chamfer toàn tay | Chamfer vùng ngón | Chamfer vùng vân tay | normal (ngón) | số ngón (heuristic) | thời gian | VRAM |
|---|---|---|---|---|---|---|---|---|
| closeup (tay đầy khung) | 2048 | 1,42 mm | 1,51 mm | 1,56 mm | 0,933 | 4 | 29,7 s | 2307 MB |
| closeup | 4096 | 1,42 mm | 1,51 mm | 1,56 mm | 0,931 | 4 | 62,1 s | 2309 MB |
| in_body (tay đúng tỉ lệ cạnh người 1,75 m) | 2048 | 12,35 mm | 11,59 mm | 16,03 mm | 0,604 | 4 | 27,3 s | 2268 MB |
| in_body | 4096 | 12,44 mm | 11,91 mm | 16,01 mm | 0,594 | 5 | 58,4 s | 2268 MB |
| in_body, octree depth 10 | 2048 | — | — | — | — | — | — | **OOM** ở cap 0,9 (đã cấp phát 9,56 GB) |

Cách đọc bảng:
- "Số ngón" đếm số vùng rời nhau trên **một** lát cắt ngang (heuristic). Kết quả 5 ở dòng in_body/4096 là do lát cắt bắt phải một mảnh gai; nhìn ảnh vẫn là 4 ngón.
- **Không có** số đo nào về độ sâu nếp khớp, biên độ vân hay khe ngón khôi phục được. "Chamfer vùng vân tay" là sai số khoảng cách trên vùng có vân, không phải biên độ vân.
- Ảnh: `measurements/vae_ceiling_hand_region.png`.

Quan sát và giả thuyết (chưa phải kết luận chung):
- Quan sát: với mẫu này, sai số tái tạo vùng ngón là 1,51 mm khi tay đầy khung, và 11,59 mm khi tay ở tỉ lệ toàn thân. Nhìn ảnh: ở tỉ lệ toàn thân ngón thành gai, không còn thấy nếp và vân. Ở closeup, nhìn ảnh vẫn thấy nếp và vân (chỉ là đánh giá định tính).
- 1,51 mm là khoảng 3,8 lần ngưỡng nhiễu, nên **không** chứng minh được vân 0,5 mm được giữ về mặt định lượng.
- Tăng token lên 4096 không đổi sai số trong thí nghiệm này. Chưa tách được phần lỗi do lưới trích xuất và phần lỗi do latent, vì depth 10 bị OOM.
- Giả thuyết: năng lực biểu diễn chi tiết của VAE bị giới hạn theo tỉ lệ vật trong khung. Nếu đúng, dựng đa tỉ lệ (toàn thân rồi cận cảnh tay/mặt) đáng thử hơn chỉ fine-tune DiT trên ảnh toàn thân.
- Chưa kiểm tra: latent được sample (không lấy mean), latent do DiT sinh ra, scan người thật, decoder đã fine-tune, lưới trích xuất mịn hơn với tile.

## GĐ 1. Hi3D v1.0 trên RTX 3060 (WSL2 Ubuntu 24.04, `--sequential-cpu-offload`, cap 0,9)

Môi trường: `scripts/setup_hi3d_env.sh` (revision code và trọng số, SHA256), lock `env/hi3d_pip_freeze.txt`. Không đụng vào env TripoSG.

| đầu vào | độ phân giải | trạng thái | thời gian | VRAM đỉnh (allocated / reserved) |
|---|---|---|---|---|
| character (mesh TripoSG + ảnh) | 512 | chạy được | 152,8 s | 604 / 838 MB |
| bàn tay in_body từ G1 + ảnh cận cảnh của bàn tay GT | 512 | chạy được | 151,0 s | 523 / 724 MB |
| như trên | 1024 | chạy được | 398,1 s | 2843 / 3874 MB |

Chất lượng vùng tay:
- Hi3D tự chuẩn hoá kích thước đầu ra, nên đầu ra được **khớp bbox** vào mesh thô (`vae_ceiling.bbox_fit`). Đây là phép khớp gần đúng, không phải đăng ký bề mặt.
- Tái lập bằng script: `vae_ceiling.prepare_hi3d_hand(...)` tạo đầu vào, `vae_ceiling.evaluate_hi3d_hand('bench/omni_hi3d')` tính bảng. Kết quả chạy lại khớp bản v1, chênh lệch lớn nhất 1,8e-15.

| | Chamfer vùng ngón (gần đúng) | Chamfer vùng vân tay (gần đúng) | normal (ngón) | số ngón (heuristic) |
|---|---|---|---|---|
| mesh thô (đầu vào Hi3D) | 11,59 mm | 16,03 mm | 0,604 | 4 |
| Hi3D 512 | 11,22 mm | 15,63 mm | 0,604 | 4 |
| Hi3D 1024 | 11,34 mm | 15,76 mm | 0,606 | 5 |

Ảnh: `measurements/hi3d_hand_region.png`, `measurements/hi3d_character_compare.png`.

Quan sát:
- Hi3D chạy được trên 3060 ở 512 và 1024 khi bật offload. Mức 1536 chưa thử.
- Đánh giá định tính từ ảnh: Hi3D thêm các dải vân lên lòng bàn tay và nếp ở khớp, nhưng **không** khôi phục hình dạng ngón của GT; ngón gai vẫn là ngón gai. Số liệu gần đúng chỉ nhích nhẹ.
- Giả thuyết cho thiết kế: Hi3D thích hợp làm tầng thêm chi tiết sau khi đã có cấu trúc đúng, còn cấu trúc ngón cần nguồn khác (prior giải phẫu hoặc dựng cận cảnh). Mới thử trên một mẫu tổng hợp.

## Chưa làm / bị chặn
- SAM 3D Body: bị chặn, đang chờ tác giả duyệt (`SAM3D_BODY_STATUS.md`).
- Benchmark Human với người thật: chưa có dữ liệu, cần Ropy chọn nguồn.
- Hi3D 1536: chưa chạy.
- Không có bằng chứng toàn vẹn trước/sau cho env TripoSG và thư mục LoRA. Chỉ biết code engine không đổi chúng; các file đó không được ghi trong các lệnh đã chạy.
