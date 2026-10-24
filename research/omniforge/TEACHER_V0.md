# Omni teacher v0 trên human_v1: kết quả âm (2026-10-24, RTX 3060)

Code: `vector3d/lab/teacher.py` (do phiên cloud viết, commit 43283af). Lệnh chạy trên ROPY-PC:
`python -m vector3d lab teacher --data bench/human_v1 -o bench/omni_teacher --size 2048`.
Dữ liệu gốc: `teacher_v0/teacher_scores.json` (từng người, từng tay), `teacher_v0/teacher_batch.json`. Ảnh: `teacher_v0/inspect.png`.

## Thiết lập
- Đầu vào: ảnh mặt trước render lại ở 2048², kèm camera mặt trước. Không dùng GT trong luồng sinh.
- Luồng chạy:
  - TripoSG (profile quality, seed 42) sinh toàn thân, đặt bằng giao thức front_only;
  - RTMW phát hiện 133 điểm khớp trên chính ảnh đó;
  - với mỗi tay: cắt cận cảnh, cô lập bàn tay bằng các đĩa tròn quanh điểm khớp, chạy TripoSG, đặt theo camera của vùng cắt, khớp độ sâu với thân;
  - ghép: cắt bỏ phần thân trong vùng tay rồi thêm phần tay mới (không kín nước).
- Mốc so sánh cùng đầu vào: `body_front_only_hires`, tức chỉ có phần thân ở trên.

## Kết quả (12 người, 24 bàn tay)

| giao thức | Chamfer-L1 | F@0.02 | held-out IoU | normal cons. |
|---|---|---|---|---|
| chỉ thân (cùng đầu vào) | 0,0154 | 0,734 | 0,647 | 0,998 |
| teacher v0 | 0,0164 | 0,720 | 0,631 | 0,993 |

| vùng tay | giao thức | độ phủ | Chamfer trung vị (tay đo được) | F@5mm, tất cả 24 (empty = 0) |
|---|---|---|---|---|
| placed (toàn cục, không dùng GT) | chỉ thân | 22/24 | 14,14 mm | 0,178 |
| placed (toàn cục, không dùng GT) | teacher v0 | 23/24 | 18,04 mm | 0,186 |
| local (chỉ hình dạng, có dùng GT) | chỉ thân | 23/24 | **8,68 mm** | **0,347** |
| local (chỉ hình dạng, có dùng GT) | teacher v0 | 24/24 | 11,69 mm | 0,306 |
| noise_floor | — | 24/24 | 0,353 mm | 1,000 |

- RTMW tìm thấy **24/24 bàn tay** trên ảnh render xám (đủ số điểm khớp để cắt vùng tay). Không có tay nào phải bù bằng GT.
- Thời gian luồng teacher: 176–207 s mỗi người (1 lượt toàn thân và 2 lượt cận cảnh TripoSG). Bước giải phẫu chạy riêng thêm khoảng 38 s: RTMW khoảng 1 s, khớp MHR khoảng 37 s trên CPU; điểm khớp lấy từ RTMW. VRAM đỉnh của TripoSG 5477–5824 MB; RTMW thêm khoảng 0,91–0,93 GB (delta toàn máy).

## Quan sát
- **Teacher v0 kém hơn mốc cùng đầu vào** ở gần như mọi metric:
  - toàn thân: Chamfer 0,0164 so với 0,0154;
  - tay, chỉ hình dạng: trung vị 11,69 so với 8,68 mm, F tất cả 24 là 0,306 so với 0,347;
  - tay, toàn cục: trung vị 18,04 so với 14,14 mm. F toàn cục nhích lên nhẹ (0,186 so với 0,178), chủ yếu nhờ độ phủ tăng.
- Chỉ độ phủ tốt hơn (24/24 so với 23/24).
- Ảnh `inspect.png` gợi ý ba nguyên nhân (giả thuyết, chưa kiểm chứng):
  1. **Mặt nạ cô lập bàn tay** là hợp các đĩa tròn quanh điểm khớp, không theo đường viền thật của bàn tay. Ảnh cắt đưa vào TripoSG là một khối lồi lõm hình tròn, nên mesh sinh ra cũng thành cục.
  2. **Đặt phần tay:** tỉ lệ chỉ lấy theo chiều cao silhouette của vùng cắt, độ sâu lấy theo trung vị độ sâu của thân. Ở person_00, phần tay mới thành một tấm mỏng lơ lửng tách khỏi cánh tay khi nhìn ngang.
  3. **Ghép:** bàn tay cũ của thân chỉ bị cắt trong dải độ sâu của phần mới, nên có chỗ còn hai lớp bề mặt chồng nhau.
- Thí nghiệm bàn tay đơn trước đó (`HANDZOOM.md`) cho thấy cận cảnh giúp ích **khi ảnh cận cảnh sạch và được căn đúng**. Kết quả này cho thấy cách cắt, cô lập và đặt phần tay của v0 làm mất lợi ích đó. Cần sửa trước khi kết luận gì về hướng đa tỉ lệ.

## Hướng sửa đề xuất (chưa làm)
1. Cô lập bàn tay theo đường viền thật: dùng alpha của chính ảnh trong vùng cắt, giới hạn bởi hình bao lồi của điểm khớp có nới rộng, thay vì hợp các đĩa tròn.
2. Đặt phần tay theo khung xương MHR đã khớp: cổ tay và đầu ngón 3D cho cả tỉ lệ lẫn độ sâu, thay vì trung vị độ sâu của thân.
3. Ghép bằng cách thay thế cả vùng tay của thân trong hộp quanh cổ tay MHR, rồi kiểm tra bằng visual hull.
4. Đo lại từng sửa đổi một trên human_v1 bằng giao thức front_only.
