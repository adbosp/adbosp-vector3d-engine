# human_v1: bộ đo người tổng hợp có GT, và mốc gốc TripoSG (2026-10-24)

## Bộ dữ liệu
- 12 người lấy mẫu từ **MHR v1.0.1 (Apache-2.0)**: `vector3d/lab/backends/human_v1_gen.py` (env anatomy, seed 0, LOD1 với 18.439 đỉnh và 36.874 mặt), chuyển sang dạng benchmark bằng `vector3d/lab/human_v1.py`.
- Cách lấy mẫu: hình dáng ngẫu nhiên (45 hệ số, σ 0,7), xoay quanh trục đứng ±25°, tay và vai dùng 60% biên độ cho phép, thân và cổ 15%, từng ngón co 80% và xoè 30%, đều nằm trong giới hạn khớp của chính MHR. Có biểu cảm nhẹ, bàn chân chạm đất. Chiều cao 1,71–1,92 m.
- Cùng bố cục với synthetic_v1: `gt.glb`, `joints.json` (127 khớp MHR, đơn vị mét), 4 góc tham chiếu và 4 góc giữ lại để đo, ảnh 512².
- Phương pháp sinh chỉ được thấy `views/front.png`. Bước đặt mesh vào khung tham chiếu theo một trong hai giao thức có tên (`human_v1.PROTOCOLS`, đã sửa theo review Codex `-06`):
  - `front_only`: một ảnh từ đầu đến cuối. Chỉ dùng silhouette mặt trước và camera mặt trước (`align.place_front_only`): hướng mặc định của model, tỉ lệ theo chiều cao silhouette, độ sâu đặt ở tâm camera.
  - `multiview_oracle`: cách `lab benchmark` làm cho `A_placed.glb`, dùng thêm silhouette back/left/right. Ba góc thêm đó là thông tin oracle. **Bản báo cáo đầu đã gọi nhầm mốc gốc này là "một ảnh"**.
- Tái lập: `human_v1_gen.py --count 12 --seed 0`, rồi `human_v1.build(raw, out, size=512)`. SHA256 của từng GT nằm trong `human_v1/manifest.json`. Dữ liệu ảnh không commit vì sinh lại được.
- Ảnh: `human_v1/people_front.png`.
- **Giới hạn:**
  - da MHR trơn (không tóc, quần áo, móng hay nếp nhăn), nên bộ này đo được hình khối và ngón tay nhưng **không đo được nếp vải hay nếp mặt**;
  - ảnh render Lambert xám, không texture;
  - người được sinh từ chính model mà bước giải phẫu dùng để khớp, nên bước đó chạy trong phân phối và sẽ được lợi trên bộ này.

## Metric vùng tay
- Hộp vùng tay: bao các khớp cổ tay, các đầu ngón, index1 và pinky1 của MHR, thêm lề 3 cm.
- `placed`: đo trên mesh đã đặt theo giao thức (không dùng GT). Kết quả gồm cả lỗi đặt toàn thân, ví dụ cánh tay lệch theo chiều sâu.
- `local`: chỉ đo hình dạng. Cắt vùng tay (lề 5 cm), ICP cứng (không đổi tỉ lệ) vào GT, rồi đo. Có dùng GT, chỉ phục vụ đánh giá.
- `noise_floor`: GT so với chính nó, cùng cách lấy mẫu. Là mức nhiễu của phép đo.
- Lấy mẫu (sửa theo review `-06` P2): cả hai mesh được cắt theo hộp cộng lề 5 cm, rồi lấy mẫu **cùng mật độ** 2×10⁶ điểm/m² với seed cố định. Nhờ vậy `placed`, `local` và `noise_floor` có cùng mức nhiễu (bản đầu sai lệch khoảng 6 lần). Bề mặt cách hộp quá 5 cm không được tính, nên lỗi rất lớn bị đánh giá thấp hơn thực tế.
- Chamfer tính bằng mm; F-score ở ngưỡng 5 mm. "empty" nghĩa là mesh dự đoán không có bề mặt nào trong hộp vùng tay. Bàn tay empty được đếm vào độ phủ, và tính F = 0 trong cột F "tất cả 24".

## Mốc gốc: TripoSG gốc, toàn thân, profile quality, seed 42 (RTX 3060)

Sinh: 52,5–55,7 s, VRAM đỉnh 5475–5531 MB mỗi người. Chấm lại từ `raw_triposg.glb` cho cả hai giao thức, không chạy lại GPU.

| giao thức | Chamfer-L1 | F@0.02 | held-out IoU | normal cons. |
|---|---|---|---|---|
| **front_only** (một ảnh từ đầu đến cuối) | 0,0157 | 0,727 | 0,639 | 0,998 |
| multiview_oracle (cách đặt của benchmark) | 0,0155 | 0,732 | 0,641 | 0,998 |

| vùng tay, 24 bàn tay | giao thức | độ phủ | Chamfer trung vị (tay đo được) | F@5mm, tất cả 24 (empty = 0) | F@5mm, tay đo được |
|---|---|---|---|---|---|
| placed | **front_only** | 22/24 | 14,37 mm | 0,188 | 0,205 |
| placed | multiview_oracle | 22/24 | 14,35 mm | 0,178 | 0,194 |
| local (chỉ hình dạng) | **front_only** | 23/24 | 9,13 mm | 0,339 | 0,354 |
| local (chỉ hình dạng) | multiview_oracle | 23/24 | 9,13 mm | 0,340 | 0,355 |
| noise_floor (GT so GT) | — | 24/24 | 0,353 mm | 1,000 | 1,000 |

Dữ liệu gốc:
- `human_v1/triposg_base_front_only.json` và `triposg_base_multiview_oracle.json` (từng người, từng tay, kèm tóm tắt);
- `human_v1/triposg_base_summary.json` (đầu ra benchmark nguyên bản);
- `human_v1/triposg_base_hand_scores_v1.json` (bản đầu, cách lấy mẫu cũ, giữ để đối chiếu).

Ảnh so sánh: `human_v1/triposg_base_person_00_compare.png` và `person_03` (bàn tay tệ nhất, chủ yếu do cánh tay lệch theo chiều sâu).

## Quan sát
- Với TripoSG, giao thức front_only và multiview_oracle cho kết quả gần như nhau. Model sinh theo hướng mặc định, nên việc dò 24 hướng không đổi gì. Từ nay mọi so sánh trên human_v1 dùng **front_only**.
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
