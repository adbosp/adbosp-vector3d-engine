# So sánh trên ảnh Alicia: TripoSG / TRELLIS / Omni (2026-10-24, RTX 3060)

- Đầu vào: bộ ảnh quay vòng (trước, phải, sau; render phối cảnh) và một ảnh mặc hoodie ngoài trời. Đây là render Daz3D của bên thứ ba: **chỉ dùng để đánh giá**, không train và không commit ảnh gốc. Ảnh nằm trên nhánh `inputs/ropy-alicia-2026-10-24`.
- Không có GT. Chỉ đo những gì đo được: độ khớp silhouette (IoU) với góc phải và góc sau mà model không được xem, thời gian, VRAM, số mặt.
- Mọi cách dùng chung một mặt nạ: IS-Net (Apache-2.0), sau khi sửa lỗi tách nền (commit 965246c). Lượt chạy trước dùng mặt nạ theo màu nền, sai hoàn toàn với ảnh hoodie, nên **đã bỏ** lượt đó.
- Cấu hình: profile quality, seed 42, mỗi lần một job trên GPU. Script tính số: `scripts/alicia_analyse.py`. Dữ liệu gốc: `alicia_report.json` và `iou_largest_parts.json`.

## Số đo, ảnh mặt trước

| cách | IoU góc phải | IoU góc sau | trung bình | thời gian sinh | nạp model | VRAM đỉnh | số mặt |
|---|---|---|---|---|---|---|---|
| TripoSG | 0,721 | 0,803 | 0,762 | 61 s | — | 5504 MB | 440k |
| TRELLIS (bản 1) | 0,727* | 0,774 | 0,750* | **21,7 s** | 36 s | 9837 MB | 165k |
| Omni (contour, độ sâu theo thân, tidy) | 0,737 | 0,803 | 0,770 | khoảng 397 s cả lượt | — | không ghi lại | 2,18 triệu |
| TripoSG + 3 góc | 0,729 | 0,788 | 0,759 | 77 s | — | 5504 MB | 440k |
| TRELLIS + 3 góc | 0,762* | 0,818* | 0,790* | 21,8 s | 24 s | 9862 MB | 165k |

- Dòng "+ 3 góc": model đã thấy góc phải và góc sau, nên IoU của hai dòng này **không phải số đo trên góc giữ lại**.
- \*: tính sau khi bỏ các mảnh vụn nhỏ hơn 5% số mặt. TRELLIS + 3 góc có 10 mảnh, trong đó 9 mảnh vụn bay lơ lửng; nếu giữ chúng, IoU chỉ còn 0,222 vì bbox bị kéo giãn. TRELLIS một ảnh: 0,723 → 0,727 với góc phải.
- TripoSG + 3 góc: bước tinh chỉnh theo các góc bị **từ chối** (IoU +11,3% nhưng độ mượt −0,045), nên giữ model đã fit.
- Thời gian Omni gồm 3 lượt TripoSG (thân và 2 bàn tay) cộng bước giải phẫu. Đường Omni không ghi VRAM đỉnh; đó là một thiếu sót.
- Đây là IoU silhouette sau khi chuẩn hoá chiều cao, so render trực giao của ta với ảnh phối cảnh tham chiếu. Chỉ là số gần đúng.

## Ảnh hoodie (không có góc giữ lại, nên chỉ so bằng mắt)

| cách | thời gian sinh | nạp model | VRAM đỉnh | số mặt |
|---|---|---|---|---|
| TripoSG | 67 s | — | 5514 MB | 478k |
| TRELLIS | **18,5 s** | 25 s | 9869 MB | 194k |
| Omni | khoảng 320 s cả lượt | — | không ghi lại | 1,63 triệu |

## Quan sát từ bảng hình (`sheet_*_views.png`, `sheet_*_closeups.png`)
- **TripoSG** cho chi tiết tốt nhất: khuôn mặt (sống mũi, môi, ngôi tóc), sợi tóc, dáng người tự nhiên. Bàn tay mượt nhưng các ngón dính một phần.
- **TRELLIS** sinh nhanh gấp khoảng 3 lần nhưng tốn gần gấp đôi VRAM (9,8 GB). Bề mặt thô hơn, có bậc kiểu voxel ở mặt. Bàn tay vỡ thành cục. Với ảnh hoodie, TRELLIS bắt được chiếc điện thoại trên tay phải và túi áo, nhưng **tự đội mũ hoodie lên** trong khi ảnh để mũ xuống. Khi dùng nhiều ảnh, TRELLIS sinh ra mảnh vụn bay lơ lửng.
- **Omni** có thân và mặt giống TripoSG. Bàn tay cận cảnh bị **loang lổ, đứt mép**, kém hơn TripoSG thường. IoU silhouette nhỉnh hơn một chút (0,770 so với 0,762), nhưng chậm hơn khoảng 4–6 lần.
- Tóm lại, trên bộ ảnh này **TripoSG gốc vẫn là kết quả đẹp nhất về chi tiết**. TRELLIS không hơn TripoSG về chi tiết mặt và tay, nên khó kỳ vọng chưng cất TRELLIS sẽ làm TripoSG chi tiết hơn ở những vùng đó. Còn TRELLIS có giúp được về hình khối tổng thể hay phần lưng không thì phải đo trên dữ liệu có GT (human_v1, hsrd_v1) bằng `lab omni-distill`. Đây là quan sát trên một nhân vật, chưa phải kết luận.
