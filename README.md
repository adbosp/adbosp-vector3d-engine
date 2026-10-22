# Vector3D Engine

Engine dựng hình học 3D từ các hình chiếu trực giao SVG (Front/Back/Left/Right/Top/Bottom). Ưu tiên toán học; AI chỉ lấp phần thiếu.

Xem kiến trúc: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Hướng dẫn cho AI agent (Claude, Codex): [AGENTS.md](AGENTS.md) và [docs/AI_PLAYBOOK.md](docs/AI_PLAYBOOK.md).

## Ảnh chụp bất kỳ → model 3D (AI)

Thả một ảnh JPG/PNG bất kỳ (vật thể rõ, ít bị che) vào ô Front rồi bấm **Generate**. Chế độ "Tự động" sẽ chọn **Ảnh chụp** và chạy:

1. **Tách nền**: dùng rembg (U²-Net) nếu đã cài `pip install -e .[ai]`, nếu không thì tách theo màu nền.
2. **Ước lượng chiều sâu bằng AI**: Depth Anything V2 Small (ONNX, chạy trên CPU; lần đầu tự tải khoảng 100 MB về `~/.cache/vector3d`). Nếu không tải được, engine dùng phương án "thổi phồng" silhouette bằng phương trình Poisson và ghi rõ trong metadata.
3. **Normal map** sinh từ bề mặt depth. Ảnh mask, depth và normal hiện ngay trên giao diện và được lưu kèm.
4. **Dựng khối kín**: mặt trước theo depth, mặt sau đối xứng hoặc phẳng (phù điêu). Nếu có thêm ảnh Right/Top thì chúng dùng để gọt và đặt đúng chiều sâu.
5. Marching cubes trên trường khoảng cách (mặt mượt, chính xác dưới mức voxel) → làm mượt → **tô màu từ ảnh** (vertex color trong GLB) → kiểm tra → xuất file.

```bash
python -m vector3d generate --front my_photo.jpg -o out/photo                        # tự nhận chế độ Ảnh chụp
python -m vector3d generate --front toy.png --right toy_side.png --back-side flat --size 0.3 -o out/toy
```

Kết quả mang nhãn `AI_GENERATED`: mặt sau và chiều sâu tuyệt đối là ước lượng. Ảnh chụp thẳng chính diện, nền trơn cho kết quả tốt nhất.

## Generate: ảnh → model 3D

Trong giao diện web (`python -m vector3d ui`): thả ảnh (SVG, PNG hoặc JPG; một ảnh Front là đủ, càng nhiều view càng chính xác), bấm **Generate**, theo dõi từng bước, xem model 3D và tải về GLB, OBJ hoặc metadata.

Từ dòng lệnh:
```bash
python -m vector3d generate --front examples/box_front.svg --right examples/box_right.svg --top examples/box_top.svg -o out/
python -m vector3d generate --front examples/vase_front.png --hints examples/vase_hints.json -o out/vase
```

Các bước: đọc ảnh → chuẩn hóa tọa độ → (AI ngữ nghĩa: chưa làm) → dựng khối từ silhouette (visual hull) → bổ sung phần khuất theo `hints.json` → marching cubes → làm mượt Taubin → kiểm tra (kín nước, khớp silhouette) → xuất file.

Phần "AI" do Claude hoặc Codex đảm nhận: agent chạy `python -m vector3d describe ...` để lấy số liệu, viết `hints.json` (tròn xoay, chiều sâu, đối xứng, khoét lỗ...), rồi engine tự dựng theo đó. Chi tiết xem [docs/AI_PLAYBOOK.md](docs/AI_PLAYBOOK.md).

```bash
pip install -e .[dev]
pytest
python -m vector3d inspect --front examples/box_front.svg --right examples/box_right.svg --top examples/box_top.svg
```

## AI 3D Lab (model 3D thật + đo đạc)
Chạy model image-to-3D mã nguồn mở (bắt đầu với TripoSG, MIT) trong môi trường riêng, ghi lại seed, thời gian, VRAM đỉnh, rồi đo và tinh chỉnh bằng hình học. Xem [docs/LAB.md](docs/LAB.md), [research/model_matrix.md](research/model_matrix.md) và [research/benchmark_report.md](research/benchmark_report.md).
```bash
python -m vector3d lab audit                       # kiểm tra GPU/VRAM/CUDA trước khi cài
python -m vector3d lab generate --image anh.png -o runs/1 --download
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
