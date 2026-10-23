# Vòng tưởng tượng (Imagination loop) v0

Code: `vector3d/lab/imagine.py`. Lệnh: `python -m vector3d lab imagine --data bench/data -o bench/imagine`. Kết quả gốc: `research/imagination_v0/summary.json` và ảnh `*_compare.png` (GT / hull / imagined, nhìn chéo 45° và từ trên xuống).

## Ý tưởng

Lấy cảm hứng từ cách não người dựng hình trong đầu, không sao chép một kiến trúc mạng có sẵn:

| Khái niệm nhận thức | Trong engine |
|---|---|
| Tri giác: cái mắt thấy chắc chắn | 4 silhouette (front/back/left/right) là sự thật, nhãn VERIFIED, không bao giờ bị sửa |
| Tưởng tượng: phần không nhìn thấy | mặt cắt ngang của từng lát (từ thoi, tròn, đến vuông) và "bộ phận nhỏ là ống tròn hay tấm dày", nhãn ESTIMATED |
| Mô phỏng trong đầu (mental imagery) | dựng từng giả thuyết thành mesh và render lại đúng 4 góc nhìn |
| Predictive coding: sai số dự đoán | so bóng đổ render với ảnh thật; ánh sáng không được cho trước mà ước lượng chung cho cả cảnh |
| Nhận biết theo bộ phận (Biederman, geons) | quyết định theo từng bộ phận, không theo từng hàng pixel, nên vài hàng thiếu bằng chứng không lật được kết quả |
| Kỳ vọng mặc định (prior) | khi không có bằng chứng thì tưởng tượng mặt cắt tròn |

Điểm toán học then chốt: với silhouette front và side, mỗi lát ngang của visual hull là các hình chữ nhật. Mọi hình lồi chạm đủ 4 cạnh của hình chữ nhật đó giữ nguyên cả hai silhouette. Họ siêu elip |x'|ⁿ + |z'|ⁿ ≤ 1 làm đúng điều này với mọi n. Vì vậy bước tưởng tượng chỉ thay đổi phần không nhìn thấy, còn silhouette vẫn khớp pixel (ref IoU = 1.000 trên cả 6 vật).

## Liên quan đến công trình đã có

Không nhận là "chưa ai làm". Các mảnh riêng lẻ đã có: analysis-by-synthesis/inverse graphics, visual hull, primitive fitting (superquadrics), dựng CAD từ 3 hình chiếu (PlankAssembly ICCV 2023, Ortho2CAD 2026, GaussianCAD 2025), agent viết CAD (Zero-to-CAD 2026). Phần mới ở đây là cách ghép: hull chính xác làm ràng buộc cứng, họ giả thuyết được chứng minh giữ silhouette, chọn giả thuyết bằng bóng đổ với ánh sáng chưa biết, quyết định theo bộ phận, và nhãn tin cậy cho từng bộ phận. Mình chưa tìm thấy bài nào làm đúng tổ hợp này, nhưng chưa tra cứu hệ thống.

## Kết quả (synthetic_v1, 6 vật, CPU 4 nhân)

Chamfer-L1 (càng thấp càng tốt), F-score@0.02 và IoU trung bình của 4 góc nhìn giữ lại không dùng khi dựng (càng cao càng tốt).

| Vật | Visual hull | Imagination v0 | TripoSG A (1 ảnh, RTX 3060) | TripoSG B (A + tinh chỉnh 4 silhouette) |
|---|---|---|---|---|
| crate | 0.0066 / 0.921 / **0.993** | 0.0068 / 0.920 / 0.985 | 0.1131 / 0.165 / 0.421 | 0.0312 / 0.698 / 0.972 |
| chair | 0.0081 / 0.867 / 0.936 | **0.0070 / 0.908 / 0.962** | 0.0151 / 0.745 / 0.765 | 0.0087 / 0.865 / 0.911 |
| mug | 0.0410 / 0.404 / 0.680 | 0.0212 / 0.661 / 0.872 | **0.0083 / 1.000 / 0.976** | 0.0089 / 1.000 / 0.972 |
| character | 0.0161 / 0.707 / 0.662 | 0.0104 / 0.855 / 0.773 | 0.0078 / **0.966 / 0.863** | **0.0074** / 0.958 / 0.848 |
| lamp | 0.0186 / 0.655 / 0.766 | 0.0105 / 0.895 / **0.975** | **0.0063 / 0.989** / 0.958 | 0.0076 / 0.921 / 0.957 |
| bracket | **0.0055** / 0.978 / **0.967** | 0.0056 / **0.979** / 0.962 | 0.0333 / 0.625 / 0.621 | 0.0058 / 0.966 / 0.947 |

Thời gian: imagination 9 đến 16 s/vật trên CPU; TripoSG 62 đến 95 s/vật trên RTX 3060.

Đọc kết quả:
- So với visual hull: tốt hơn rõ ở 4/6 vật (chair, mug, character, lamp), gần bằng ở crate và bracket (hull vốn đã đúng với vật vuông).
- So với TripoSG: thắng ở crate, chair, bracket (vật cứng, đồ nội thất). Thua ở mug và character; lamp ngang ngửa (IoU góc giữ lại cao hơn, Chamfer kém hơn).
- Chỗ thua đều là phần **không có trong 4 ảnh tham chiếu**: lòng cốc rỗng, chụp đèn rỗng (chỉ thấy khi nhìn từ trên hoặc dưới), tay chéo về phía trước của nhân vật. TripoSG đoán được nhờ đã học từ dữ liệu lớn.

## Công bằng của phép so sánh (đọc trước khi trích số)

- Đầu vào khác nhau: TripoSG A chỉ nhận 1 ảnh màu; B và imagination nhận cả 4 view. Imagination còn dùng bóng đổ của 4 view.
- Ảnh synthetic được render bằng mô hình Lambert, cùng họ với mô hình bóng đổ imagination dùng để so (ánh sáng thì không cho trước). Ảnh chụp thật có texture, phản xạ và bóng đổ phức tạp hơn, nên bước so sánh sẽ yếu hơn. Với ảnh thật cần thay bằng normal map từ AI.
- Bản vẽ nét (line art) không có bóng đổ: imagination khi đó chỉ còn prior "mặt cắt tròn".
- Số TripoSG lấy từ lần chạy thật ngày 2026-10-22 (`research/benchmark_report_rtx3060.md`), không chạy lại hôm nay.
- 6 vật là quá ít để kết luận thống kê.

## Giới hạn còn mở

1. Không thấy được hốc lõm mà 4 view ngang không nhìn vào (cốc, chụp đèn).
2. Một vùng liền trong silhouette được coi là một bộ phận, nên tay dính liền thân thành tấm dẹt. Cần tách bộ phận trong 2D (medial axis hoặc phân rã lồi).
3. Lát cắt chỉ theo trục đứng; vật có trục chính nằm ngang chưa được xử lý riêng.
4. Mặt cắt luôn đối xứng trong hình chữ nhật của nó.

## Bước tiếp theo

1. Ghép với TripoSG khi ROPY-PC online: dùng TripoSG làm "trí tưởng tượng có kiến thức" cho phần không nhìn thấy, imagination và hull làm "tri giác" để kiểm chứng. Điểm nào của TripoSG nằm ngoài hull thì chắc chắn sai (VERIFIED) và bị cắt; phần silhouette TripoSG không phủ được thì lấp bằng lát imagination. Mục tiêu là lấy điểm mạnh của cả hai cột trong bảng trên.
2. Tách bộ phận 2D để sửa giới hạn 2.
3. Dùng normal map AI thay cho bóng đổ để chạy với ảnh thật và bản vẽ.
