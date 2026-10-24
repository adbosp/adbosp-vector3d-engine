# Omni teacher: độ sâu bàn tay theo khung xương MHR (một thay đổi so với vòng contour), 2026-10-24, RTX 3060

- Code `76d6cf0`, dùng `--depth skeleton`. Lệnh chạy: `python -m vector3d lab teacher --data bench/human_v1 -o bench/omni_teacher_skel --size 2048 --isolation contour --depth skeleton --reuse-body bench/omni_teacher`.
- Cùng thân với v0, cùng seed 42 và profile quality. 12/12 người thành công.
- **24/24 bàn tay lấy độ sâu từ khung xương**, không tay nào phải quay về độ sâu theo thân.
- Dữ liệu gốc: `teacher_v2_skeleton/teacher_scores.json`.
- **Lưu ý về dữ liệu:** người trong human_v1 được sinh từ MHR, nên khung xương MHR được lợi trên bộ này. Mọi mức lợi ở đây chỉ là cận trên, cho tới khi đo trên người không sinh từ MHR.

Vùng tay, mỗi ô ghi: độ phủ / Chamfer trung vị / F@5mm tính cả 24 tay (tay empty tính F = 0):

| | Chamfer thân | placed | local |
|---|---|---|---|
| chỉ thân | 0,0154 | 22/24 / 14,14 mm / 0,178 | 23/24 / 8,68 mm / 0,347 |
| contour + độ sâu theo thân | 0,0157 | 23/24 / **12,56 mm** / **0,196** | 24/24 / **8,51 mm** / **0,380** |
| contour + độ sâu theo khung xương | 0,0160 | 24/24 / 15,58 mm / 0,168 | 24/24 / 10,25 mm / 0,349 |

So từng tay (F@5mm), khung xương so với contour:
- local: tốt hơn 14, kém hơn 10 (trung vị ΔF +0,010);
- placed: tốt hơn 8, kém hơn 13.

Khung xương so với chỉ thân:
- local: tốt hơn 10, kém hơn 11;
- placed: tốt hơn 6, kém hơn 12.

Thời gian 117–155 s mỗi người (thân dùng lại), cộng khoảng 38 s cho bước giải phẫu. VRAM đỉnh 5402–6022 MB.

## Quan sát
- Lấy độ sâu theo khung xương **kém hơn** lấy theo thân ở metric toàn cục (placed), mặc dù dữ liệu này có lợi cho MHR. Ở metric chỉ xét hình dạng, kết quả lẫn lộn: số tay tốt hơn nhiều hơn, nhưng trung vị và F trung bình thì kém hơn.
- Giả thuyết (chưa kiểm chứng): việc khớp MHR chỉ dựa vào điểm 2D, không có ràng buộc độ sâu, nên độ sâu tương đối giữa tay và thân bị sai. Ở bước giải phẫu trước đây, cũng chính sự mơ hồ về độ sâu này đã làm người bị ngả về phía camera.
- **Chọn cấu hình tốt hơn:** contour + độ sâu theo thân. Đây là gốc cho vòng "band" tiếp theo và là mặc định nên dùng trong giao diện.
