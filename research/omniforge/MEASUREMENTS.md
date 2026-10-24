# OmniForge: đo đạc khởi đầu (2026-10-24, ROPY-PC, RTX 3060 12 GB)

Mọi con số dưới đây lấy từ lần chạy thật. File gốc nằm trong `measurements/`.

## 0b. TripoSG không giảm số mặt (character, seed 42)

| profile | số mặt | Chamfer-L1 | F@0.02 | held-out IoU | normal cons. | thời gian | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| balanced (giảm còn 50k mặt) | 50.000 | 0.007753 | 0.9658 | 0.8632 | 0.9983 | 60.9 s | 5571 MB |
| quality (giữ nguyên) | 612.532 | 0.007762 | 0.9653 | 0.8631 | 0.9997 | 65.5 s | 5571 MB |

Kết luận: trên benchmark hiện có, giảm mặt **không đo được** thiệt hại. Lý do là mesh chuẩn của character chỉ có 25.780 mặt, không có chi tiết nhỏ nào để mất. Benchmark synthetic_v1 không đo được chi tiết người; cần benchmark Human (GĐ 0 trong PLAN.md).

## G1. Trần chi tiết của VAE TripoSG (mẫu bàn tay thủ tục)

Mẫu: bàn tay phải dựng từ SDF (`vector3d/lab/humanoid.py`): 4 ngón cách nhau 4 mm, nếp khớp sâu 1,2 mm, vân lòng bàn tay biên độ 0,5 mm. Đây là mẫu tổng hợp để đo, không phải scan người thật.
Round-trip: lấy 200.000 điểm bề mặt kèm normal, encode bằng VAE TripoSG (fp16, mean của latent), decode bằng flash decoder như pipeline (octree depth 9). Đo lại theo milimét trong toạ độ bàn tay.
Ngưỡng nhiễu của phép đo (bàn tay so với chính nó): Chamfer 0,40 mm, normal agreement 0,997, đếm được 4 ngón.

| cảnh | token | Chamfer toàn tay | Chamfer vùng ngón | Chamfer vân tay | normal (ngón) | số ngón | thời gian | VRAM |
|---|---|---|---|---|---|---|---|---|
| closeup (tay đầy khung) | 2048 | 1,42 mm | 1,51 mm | 1,56 mm | 0,933 | 4 | 29,7 s | 2307 MB |
| closeup | 4096 | 1,42 mm | 1,51 mm | 1,56 mm | 0,931 | 4 | 62,1 s | 2309 MB |
| in_body (tay đúng tỉ lệ cạnh người 1,75 m) | 2048 | 12,35 mm | 11,59 mm | 16,03 mm | 0,604 | 4 | 27,3 s | 2268 MB |
| in_body | 4096 | 12,44 mm | 11,91 mm | 16,01 mm | 0,594 | 5* | 58,4 s | 2268 MB |
| in_body, octree depth 10 | 2048 | — | — | — | — | — | — | **OOM** ở mức cap 0,9 (đã cấp phát 9,56 GB) |

\* 5 là do lát cắt bắt phải một mảnh gai, nhìn ảnh vẫn là 4 ngón.
Ảnh: `measurements/vae_ceiling_hand_region.png`.

Kết luận:
- **VAE không phải nút thắt khi bàn tay đầy khung hình.** Ngón, nếp khớp và vân tay 0,5 mm đều còn (sai số 1,5 mm, chỉ hơn ngưỡng nhiễu một chút).
- **Ở tỉ lệ toàn thân thì mất chi tiết.** Ngón thành gai nhọn, nếp và vân biến mất, sai số tăng khoảng 8 lần (11,6 mm). Tăng lên 4096 token không cải thiện gì. Thử lưới trích xuất mịn hơn (depth 10) thì hết VRAM, nên chưa tách được phần lỗi do lưới và phần lỗi do latent.
- Hệ quả cho thiết kế: train lại DiT với ảnh toàn thân sẽ không tạo ra ngón tay. Hướng khả thi là **đa tỉ lệ**: dựng toàn thân trước, sau đó dựng riêng vùng tay và mặt ở khung cận cảnh (nơi VAE giữ được chi tiết), rồi ghép lại.

## GĐ 1. Hi3D v1.0 trên RTX 3060 (WSL2 Ubuntu 24.04, `--sequential-cpu-offload`, cap 0,9)

Môi trường riêng: micromamba `hi3d` (Python 3.10, torch 2.5.0+cu124, flash-attn 2.7.4.post1 wheel, gcc conda làm `CC` cho Triton). Không đụng vào env TripoSG.

| đầu vào | độ phân giải | trạng thái | thời gian | VRAM đỉnh (allocated / reserved) |
|---|---|---|---|---|
| character (mesh TripoSG + ảnh) | 512 | chạy được | 152,8 s | 604 / 838 MB |
| bàn tay dính ngón (VAE in_body) + ảnh cận cảnh bàn tay đúng | 512 | chạy được | 151,0 s | 523 / 724 MB |
| như trên | 1024 | chạy được | 398,1 s | 2843 / 3874 MB |

Chất lượng vùng tay (Hi3D tự chuẩn hoá lại kích thước, nên tôi khớp bbox của nó vào bbox mesh thô; cách khớp này gần đúng):

| | Chamfer vùng ngón | Chamfer vân tay | normal (ngón) | số ngón |
|---|---|---|---|---|
| mesh thô (đầu vào Hi3D) | 11,59 mm | 16,03 mm | 0,604 | 4 |
| Hi3D 512 | 11,22 mm | 15,63 mm | 0,604 | 4 |
| Hi3D 1024 | 11,34 mm | 15,76 mm | 0,606 | 5* |

Ảnh: `measurements/hi3d_hand_region.png`, `measurements/hi3d_character_compare.png`.

Kết luận:
- Hi3D **chạy được** trên 3060 ở 512 và 1024 khi bật offload, VRAM còn dư nhiều. 1536 chưa thử.
- Hi3D **thêm chi tiết bề mặt** từ ảnh: vân lòng bàn tay và nếp khớp xuất hiện trở lại, đầu ngón tròn hơn. Nhưng nó **giữ nguyên cấu trúc sai** của mesh thô: ngón gai vẫn là ngón gai. Số liệu chỉ nhích nhẹ.
- Hệ quả: Hi3D làm tầng thêm chi tiết **sau khi** đã có cấu trúc đúng. Cấu trúc ngón phải đến từ prior giải phẫu (khung MHR, xem `SAM3D_BODY_STATUS.md`) hoặc từ dựng cận cảnh.

## Chưa làm / bị chặn
- SAM 3D Body: bị chặn, đang chờ tác giả duyệt quyền truy cập (`SAM3D_BODY_STATUS.md`).
- Benchmark Human với người thật: chưa có dữ liệu (cần Ropy quyết định nguồn).
- Hi3D 1536: chưa chạy.
