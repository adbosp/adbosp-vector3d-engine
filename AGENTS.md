# AGENTS.md: Vector3D Engine

Engine dựng model 3D từ các hình chiếu trực giao (SVG ưu tiên; PNG/JPG được tách silhouette). Python ≥ 3.10, numpy, scikit-image, Pillow.

## Khi người dùng nhờ "dựng model từ ảnh"
Làm theo **docs/AI_PLAYBOOK.md**, không ứng biến: `describe` → viết `hints.json` (chỉ khi cần) → `generate` → kiểm tra `metadata.json` → báo cáo.

## Lệnh
```bash
pip install -e .[dev]           # cài đặt
pytest                          # toàn bộ test, phải xanh trước khi commit
python -m vector3d describe  --front F [--right R --top T ...]           # dữ kiện JSON cho agent
python -m vector3d generate  --front F [...] [--hints H.json] -o out/    # GLB + OBJ + metadata.json
python -m vector3d inspect   --front F [...] [--json]                    # chỉ phân tích, không dựng mesh
python -m vector3d ui                                                    # giao diện web, nút Generate
```

## Bản đồ code
- `vector3d/svg/`: SVG parser (viewBox, đơn vị, transform, path/arc). `vector3d/raster/`: ảnh → silhouette → contour.
- `vector3d/normalize/`: camera 6 view, chuẩn hóa tọa độ, thang đo, kiểm tra giữa các view.
- `vector3d/reconstruct/`: `hull.py` (visual hull), `hints.py` (áp dụng quyết định của AI), `surface.py` (marching cubes, Taubin, kiểm tra mesh).
- `vector3d/export/`: OBJ, GLB. `vector3d/generate.py`: pipeline Generate từng bước. `vector3d/agent.py`: lệnh `describe`.
- `vector3d/ui/`: server HTTP + trang web. `docs/ARCHITECTURE.md`: kiến trúc và quy ước.

## Quy tắc bất biến
- Toán trước, AI sau. Không dùng AI để đoán số đã đo được.
- Mọi dữ liệu mang nhãn `VERIFIED` / `ESTIMATED` / `AI_GENERATED`. Không nâng nhãn.
- World: tay phải, mét, +Y lên, +Z về phía người nhìn Front. Left/Right là **bên của vật thể** (Right: camera ở −X).
- Pixel và đơn vị SVG không phải kích thước thật khi chưa có thang đo.
- Bước chưa làm phải báo `skipped` kèm lý do, không giả kết quả.
- Code mới đi kèm test; giữ phong cách hiện có (dataclass, hàm thuần, docstring ngắn).
