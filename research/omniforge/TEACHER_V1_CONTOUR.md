# Omni teacher: cô lập bàn tay theo đường viền (thay đổi duy nhất so với v0), 2026-10-24, RTX 3060

- Code: `eee9a6e` (`--isolation contour`): giữ alpha của chính ảnh trong hình bao lồi điểm khớp đã nới rộng, cắt ở đường cổ tay, chỉ giữ các mảnh có chứa điểm khớp.
- Lệnh chạy: `python -m vector3d lab teacher --data bench/human_v1 -o bench/omni_teacher_contour --size 2048 --isolation contour --reuse-body bench/omni_teacher`.
- Dùng lại thân của v0 (`--reuse-body`), nên **chỉ phần tay thay đổi**. Mốc `body_front_only_hires` của lần chạy này trùng khít với v0 trên mọi metric, xác nhận thân giữ nguyên.
- Cùng profile quality, seed 42, cap VRAM 0,95. 12/12 người thành công; RTMW tìm thấy đủ 24/24 bàn tay. Không dùng GT trong luồng sinh.
- Dữ liệu gốc: `teacher_v1_contour/teacher_scores.json`, `teacher_batch.json`. Ảnh: `teacher_v1_contour/inspect_a.png`, `inspect_b.png`.

## Ba cách cùng đầu vào

| | Chamfer-L1 thân | F@0.02 | held-out IoU | normal cons. |
|---|---|---|---|---|
| chỉ thân | **0,0154** | **0,734** | **0,647** | **0,998** |
| teacher v0 (đĩa tròn) | 0,0164 | 0,720 | 0,631 | 0,993 |
| teacher contour | 0,0157 | 0,730 | 0,638 | 0,992 |

Vùng tay, mỗi ô ghi: độ phủ / Chamfer trung vị (các tay đo được) / F@5mm tính cả 24 tay (tay empty tính F = 0):

| | placed (toàn cục, không dùng GT) | local (chỉ hình dạng, có dùng GT) |
|---|---|---|
| chỉ thân | 22/24 / 14,14 mm / 0,178 | 23/24 / 8,68 mm / 0,347 |
| teacher v0 (đĩa tròn) | 23/24 / 18,04 mm / 0,186 | 24/24 / 11,69 mm / 0,306 |
| teacher contour | 23/24 / **12,56 mm** / **0,196** | 24/24 / **8,51 mm** / **0,380** |
| noise_floor | 0,353 mm | 0,353 mm |

So từng tay với mốc "chỉ thân" (F@5mm; tay empty tính F = 0):

| | local: số tay tốt hơn / kém hơn | trung vị ΔF | placed: tốt hơn / kém hơn |
|---|---|---|---|
| contour | 13 / 11 | +0,035 | 12 / 9 |
| v0 | 8 / 16 | −0,068 | 10 / 12 |

Thời gian:
- Luồng chạy (chỉ 2 lượt TripoSG cận cảnh, thân dùng lại) mất 115–160 s mỗi người. Bước giải phẫu chạy riêng thêm khoảng 38 s.
- Nếu tính cả việc sinh thân như v0 thì cộng thêm khoảng 55–60 s.
- VRAM đỉnh 5409–6022 MB.

## Quan sát (chưa phải kết luận)
- Cô lập theo đường viền **xoá được phần kém đi của v0**, và nhỉnh hơn mốc ở các metric vùng tay: F local 0,380 so với 0,347; trung vị placed 12,56 so với 14,14 mm.
- **Mức lợi nhỏ và không ổn định:** contour tốt hơn mốc ở 13/24 tay và kém hơn ở 11/24. Với 12 người và 1 seed, chưa thể nói đây là cải thiện thật.
- Metric toàn thân vẫn hơi kém mốc (Chamfer 0,0157 so với 0,0154; normal 0,992 so với 0,998). Ghép thêm phần tay không làm thân tốt lên.
- Nhìn ảnh:
  - Ảnh cắt của contour không còn các khối tròn như v0 mà theo đúng viền bàn tay, cắt thẳng ở cổ tay.
  - Khi bàn tay đè lên thân hoặc mặt trong ảnh (person_10, person_06), alpha của ảnh trong vùng hình bao có cả pixel của thân. Vùng cắt vì thế mang theo một mảng thân phẳng, và TripoSG dựng mảng đó thành một tấm sau bàn tay. Điều này xảy ra ở cả v0 lẫn contour.
  - Hình dạng ngón và vị trí phần tay (person_00, 04, 06) gần như giống v0.

## Bước tiếp đề xuất (mỗi lần một thay đổi, đo lại trên human_v1 front_only)
1. **Tách tay khỏi thân khi chồng lên nhau:** tô vùng không phải tay bằng màu nền, dựa vào khoảng cách tới khung ngón 2D hoặc phân đoạn ngữ nghĩa, thay vì giữ toàn bộ alpha.
2. **Đặt phần tay theo khung xương MHR:** cổ tay và đầu ngón 3D cho tỉ lệ và độ sâu.
3. Chạy nhiều seed để có khoảng tin cậy.
