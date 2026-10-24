# human_v1: bộ đo người tổng hợp có GT, và mốc gốc TripoSG (2026-10-24)

## Bộ dữ liệu
- 12 người lấy mẫu từ **MHR v1.0.1 (Apache-2.0)**: `vector3d/lab/backends/human_v1_gen.py` (env anatomy, seed 0, LOD1 với 18.439 đỉnh và 36.874 mặt), chuyển sang dạng benchmark bằng `vector3d/lab/human_v1.py`.
- Cách lấy mẫu: hình dáng ngẫu nhiên (45 hệ số, σ 0,7), xoay quanh trục đứng ±25°, tay và vai dùng 60% biên độ cho phép, thân và cổ 15%, từng ngón co 80% và xoè 30%, đều nằm trong giới hạn khớp của chính MHR. Có biểu cảm nhẹ, bàn chân chạm đất. Chiều cao 1,71–1,92 m.
- Cùng bố cục với synthetic_v1: `gt.glb`, `joints.json` (127 khớp MHR, đơn vị mét), 4 góc tham chiếu và 4 góc giữ lại để đo, ảnh 512². Các phương pháp chỉ được thấy `views/front.png`.
- Tái lập: `human_v1_gen.py --count 12 --seed 0`, rồi `human_v1.build(raw, out, size=512)`. SHA256 của từng GT nằm trong `human_v1/manifest.json`. Dữ liệu ảnh không commit vì sinh lại được.
- Ảnh: `human_v1/people_front.png`.
- **Giới hạn:**
  - da MHR trơn (không tóc, quần áo, móng hay nếp nhăn), nên bộ này đo được hình khối và ngón tay nhưng **không đo được nếp vải hay nếp mặt**;
  - ảnh render Lambert xám, không texture;
  - người được sinh từ chính model mà bước giải phẫu dùng để khớp, nên bước đó chạy trong phân phối và sẽ được lợi trên bộ này.

## Metric vùng tay
- Hộp vùng tay: bao các khớp cổ tay, các đầu ngón, index1 và pinky1 của MHR, thêm lề 3 cm.
- `placed`: đo trên mesh đã đặt theo silhouette như benchmark A (không dùng GT). Kết quả gồm cả lỗi đặt toàn thân, ví dụ cánh tay lệch theo chiều sâu.
- `local`: chỉ đo hình dạng. Cắt vùng tay, ICP cứng (không đổi tỉ lệ) vào GT, rồi đo. Có dùng GT, chỉ phục vụ đánh giá.
- Chamfer tính bằng mm; F-score ở ngưỡng 5 mm. "empty" nghĩa là mesh dự đoán không có bề mặt nào trong hộp vùng tay.

## Mốc gốc: TripoSG gốc, toàn thân, profile quality, seed 42 (RTX 3060)

| | Chamfer-L1 | F@0.02 | held-out IoU | normal cons. | thời gian | VRAM đỉnh |
|---|---|---|---|---|---|---|
| trung bình 12 người (A) | 0,0155 | 0,732 | 0,641 | 0,998 | 52,5–55,7 s | 5475–5531 MB |

| vùng tay (24 bàn tay) | đo được | empty | Chamfer trung bình | Chamfer trung vị | F@5mm trung bình | F@5mm trung vị |
|---|---|---|---|---|---|---|
| placed | 22 | 2 | 24,11 mm | 14,64 mm | 0,182 | 0,206 |
| local (chỉ hình dạng) | 23 | 1 | 10,62 mm | 9,09 mm | 0,356 | 0,376 |

Dữ liệu gốc: `human_v1/triposg_base_summary.json`, `human_v1/triposg_base_hand_scores.json`. Ảnh so sánh: `human_v1/triposg_base_person_00_compare.png` và `person_03` (bàn tay tệ nhất, 91,8 mm, chủ yếu do cánh tay lệch theo chiều sâu).

## Quan sát
- Ở tỉ lệ toàn thân, sai số vùng tay (trung vị 9,1 mm khi chỉ xét hình dạng) gấp nhiều lần sai số bàn tay cận cảnh trong thí nghiệm tổng hợp (1,6 mm, `HANDZOOM.md`). Hai thí nghiệm khác mẫu tay nên chưa so trực tiếp được; đây là mốc để luồng cận cảnh trên human_v1 phải vượt qua.
- Sai số `placed` lớn hơn `local` nhiều. Một phần lớn lỗi vùng tay đến từ vị trí cánh tay theo chiều sâu, không riêng hình dạng ngón.
- Chỉ 12 người, một seed: chưa đủ để kết luận thống kê.

## Bước tiếp
Luồng teacher trên human_v1, đo bằng đúng metric này:
- toàn thân, sau đó MHR (RTMW+MHR);
- cắt cận cảnh tay theo khớp MHR;
- chạy TripoSG trên từng vùng cận cảnh;
- căn vùng theo MHR, không dùng GT;
- ghép vào thân trong visual hull.
