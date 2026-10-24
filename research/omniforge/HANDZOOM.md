# Thử luồng cận cảnh tay trên bàn tay tổng hợp (2026-10-24)

Code: `vector3d/lab/handzoom.py`. Lệnh: `handzoom.run('bench/omni_handzoom', 'bench/data/character/gt.glb', hi3d_wsl=handzoom.hi3d_via_wsl(512))`. Kết quả gốc: `handzoom/handzoom.json`, ảnh so sánh `handzoom/hand_region.png`, ảnh đầu vào `handzoom/input_*.png`.

## Thiết lập
- Mẫu là bàn tay thủ tục trong `humanoid.py`, **tổng hợp, không phải người thật**. Ảnh đầu vào là render Lambert xám nhìn thẳng (1024², có alpha).
- TripoSG: profile `quality` (không giảm mặt), seed 42, trọng số `90facf66b8638c8d`, không dùng LoRA. Hi3D v1.0: độ phân giải 512, offload, cap 0,9.
- **Căn chỉnh để đo** (dùng GT, chỉ phục vụ đánh giá, không thuộc luồng sinh): khớp bbox, sau đó ICP tương tự (scale, xoay, tịnh tiến). Cả ba cách dùng cùng một thủ tục. Với cách toàn thân, căn cả hình rồi mới cắt vùng tay.

| cách | Chamfer vùng ngón | Chamfer toàn tay | Chamfer vùng vân | normal (ngón) | số ngón (heuristic) | thời gian sinh | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| TripoSG toàn thân (tay cạnh người 1,75 m) | 7,76 mm | 7,52 mm | 1,62 mm | 0,595 | 11* | 61,6 s | 5585 MB |
| TripoSG cận cảnh (tay đầy khung) | **1,59 mm** | 2,36 mm | 2,50 mm | **0,939** | 4 | 67,2 s | 5600 MB |
| TripoSG cận cảnh + Hi3D 512 | 1,58 mm | 2,37 mm | 2,53 mm | 0,927 | 4 | +137,4 s | 663 MB |

\* Bộ đếm một lát cắt bị nhiễu trên bề mặt hỏng. Nhìn ảnh thì các ngón dính thành một khối như găng tay hở ngón.
Ngưỡng nhiễu của phép đo (tay so với chính nó): Chamfer 0,40 mm.

## Quan sát (một mẫu tổng hợp, chưa phải kết luận)
- **Cận cảnh tốt hơn hẳn toàn thân trên mẫu này:** sai số vùng ngón giảm từ 7,76 mm xuống 1,59 mm (khoảng 4,9 lần), normal agreement của ngón tăng từ 0,60 lên 0,94. Nhìn ảnh: bản cận cảnh tách được 4 ngón, có nếp khớp và vân; bản toàn thân bị dính ngón.
- Kết quả này khớp với thí nghiệm VAE trước đó (tỉ lệ tay trong khung quyết định lượng chi tiết giữ được), giờ được thấy trên **cả pipeline sinh** chứ không chỉ VAE.
- **Hi3D không thêm gì đo được** khi đầu vào đã tốt (1,59 → 1,58 mm, normal hơi giảm), mà tốn thêm 137 s. Với mẫu này, Hi3D chỉ đáng dùng khi mesh thô còn thiếu chi tiết.
- Điểm bất thường: vùng vân tay của bản toàn thân có Chamfer thấp hơn (1,62 so với 2,50 mm). Khả năng là khối "găng tay" phủ kín vùng lòng bàn tay, còn bản cận cảnh hơi lệch theo độ dày sau ICP. Chưa kiểm chứng; metric vùng vân không phản ánh vân có được giữ hay không.
- Giới hạn của thí nghiệm: tay đặt **cạnh** cơ thể chứ không gắn vào cổ tay; ảnh render không có texture; mới có một mẫu và một seed.

## Bước tiếp
Cần human_v1 (người tổng hợp sinh từ MHR, có GT giữ riêng) để đo luồng thật: toàn thân → MHR → cắt cận cảnh tay và mặt theo khớp → TripoSG cận cảnh → căn theo MHR (không dùng GT) → ghép trong hull.
