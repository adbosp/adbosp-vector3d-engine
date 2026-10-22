# Vector3D Engine

Engine dựng hình học 3D từ các hình chiếu trực giao SVG (Front/Back/Left/Right/Top/Bottom). Ưu tiên toán học; AI chỉ lấp phần thiếu.

Xem kiến trúc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
pip install -e .[dev]
pytest
python -m vector3d inspect --front examples/box_front.svg --right examples/box_right.svg --top examples/box_top.svg
```

## Giao diện web

```bash
python -m vector3d ui            # mở http://127.0.0.1:8765/ trong trình duyệt
python -m vector3d ui --port 9000 --no-browser
```

- Nạp 6 hình chiếu SVG (bấm vào ô hoặc kéo thả; kéo nhiều file thì tên chứa `front`/`right`/... tự vào đúng ô), hoặc bấm **Nạp ví dụ**.
- **3D viewer:** các view được đặt lên các mặt của "hộp kính" bao vật thể để kiểm tra trực quan; xoay/zoom bằng chuột, nút Front/Right/Top/Iso, chế độ trực giao, khung dây, bật/tắt từng view.
- Mở mesh **OBJ / GLB / STL** để so với các hình chiếu.
- Xem từng view 2D theo trục world, bảng so khớp kích thước và thang đo từng view.

Trang tải three.js từ `cdn.jsdelivr.net`, nên lần đầu cần có mạng.

Đã có: SVG parser (viewBox, đơn vị, transform lồng nhau, path/arc, use/symbol, layer, ngữ nghĩa `data-v3d-*`), xác định thang đo có độ tin cậy, camera cho 6 hình chiếu, chuẩn hóa tọa độ và kiểm tra nhất quán giữa các view.
