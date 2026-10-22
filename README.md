# Vector3D Engine

Engine dựng hình học 3D từ các hình chiếu trực giao SVG (Front/Back/Left/Right/Top/Bottom). Ưu tiên toán học; AI chỉ lấp phần thiếu.

Xem kiến trúc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
pip install -e .[dev]
pytest
python -m vector3d inspect --front examples/box_front.svg --right examples/box_right.svg --top examples/box_top.svg
```

Đã có: SVG parser (viewBox, đơn vị, transform lồng nhau, path/arc, use/symbol, layer, ngữ nghĩa `data-v3d-*`), xác định thang đo có độ tin cậy, camera cho 6 hình chiếu, chuẩn hóa tọa độ và kiểm tra nhất quán giữa các view.
