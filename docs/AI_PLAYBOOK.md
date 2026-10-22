# AI Playbook: vận hành Vector3D Engine bằng Claude / Codex

Tài liệu này dành cho AI agent (Claude Code, Codex, hoặc agent khác) làm việc trong repo. Đọc hết trước khi dựng model.

## 1. Vai trò của agent

Engine dựng hình học bằng toán. Agent chỉ ra **quyết định** cho những gì toán không xác định được từ dữ liệu, rồi engine thực thi các quyết định đó một cách tất định.

| Agent ĐƯỢC làm | Agent KHÔNG được làm |
|---|---|
| Nhận diện vật thể và bộ phận (lỗ, trụ, tròn xoay, đối xứng) | Sửa trực tiếp file mesh, OBJ hay GLB |
| Ước lượng chiều sâu mà không view nào nhìn thấy | Ghi đè số đo đã có (kích thước, thang đo, tọa độ đã xác minh) |
| Viết `hints.json` và giải thích lý do trong `notes` | Gắn nhãn VERIFIED cho dữ liệu do AI đoán |
| Chạy lệnh, đọc metadata, sửa hints rồi chạy lại | Bỏ qua cảnh báo `silhouette_iou` thấp hoặc `watertight: false` |

Nguyên tắc gốc: *Geometry that can be determined mathematically must be reconstructed mathematically. AI only resolves missing, uncertain or creative geometry.*

## 2. Quy trình chuẩn

```bash
# 0. Cài đặt (một lần)
pip install -e .[dev]

# 1. Lấy dữ kiện đo được (JSON), KHÔNG đoán từ ảnh khi số đã có ở đây
python -m vector3d describe --front in/front.png --right in/right.png --top in/top.png > out/facts.json

# 2. Đọc facts.json, áp các công thức ở mục 4 và quy tắc ở mục 5, rồi viết out/hints.json (schema ở mục 6)
#    Nếu mọi trục đều có view quan sát và không có đặc trưng khuất, có thể bỏ qua hints.

# 3. Dựng model
python -m vector3d generate --front in/front.png --right in/right.png --top in/top.png \
    --hints out/hints.json --resolution 128 -o out/

# 4. Kiểm tra out/metadata.json (mục 7). Chưa đạt thì sửa hints rồi quay lại bước 3, tối đa 3 vòng.
# 5. Báo cáo cho người dùng: đường dẫn model, độ tin cậy, các giả định, các hints đã áp dụng.
```

Tên view theo **bên của vật thể**: `right` là mặt bên phải của chính vật thể, camera đặt ở −X. Nếu người dùng chỉ đưa một ảnh mà không nói nó là view nào, coi đó là `front`.

## 3. Hệ tọa độ

- World là hệ tay phải, đơn vị mét: +X phải, +Y lên, +Z hướng về người nhìn Front (giống glTF).
- Mặt phẳng view: u sang phải, v lên trên. Với SVG: `u = s·(x − x₀)`, `v = −s·(y − y₀)`, trong đó `s` là số mét trên mỗi đơn vị SVG, `(x₀, y₀)` là gốc (marker `v3d-origin` hoặc tâm bounding box).
- Điểm trên view nằm trong world tại: **P = u·R + v·U + d·C**, với R, U, C là các vector trong bảng dưới và d là độ sâu hướng về camera.

| View | R (+u) | U (+v) | C (hướng về camera) | Điểm world (X,Y,Z) hiện trên view tại (u, v) |
|---|---|---|---|---|
| front | +X | +Y | +Z | (X, Y) |
| back | −X | +Y | −Z | (−X, Y) |
| right | +Z | +Y | −X | (Z, Y) |
| left | −Z | +Y | +X | (−Z, Y) |
| top | +X | −Z | +Y | (X, −Z) |
| bottom | +X | +Z | −Y | (X, Z) |

Trong `facts.json`, `u_axis`, `v_axis` và `camera_at` của từng view lặp lại đúng bảng này.

## 4. Bộ công thức

### 4.1 Thang đo
- Có `meters_per_unit` (nguồn `explicit`, `dimension` hoặc `document_units`): dùng luôn.
- Thang đo lan truyền giữa hai view qua trục chung: `s_B = s_A · L_A / L_B`, trong đó L là chiều dài của silhouette trên trục world chung (engine đã tự làm).
- `units == "drawing_units"` nghĩa là chưa có thang đo thật. Mọi số trong hints dùng **cùng đơn vị với facts.json**. Nếu người dùng cho một kích thước thật (ví dụ "cao 30 cm"), ghi vào `notes` và báo lại để người dùng thêm `data-v3d-scale` hoặc kích thước ghi chú vào SVG. Không tự nhân hệ số trong hints.

### 4.2 Ghép điểm giữa các view
Cùng một đỉnh (X, Y, Z) phải thỏa đồng thời:
- `front.u = X`, `top.u = X`, `bottom.u = X`, `back.u = −X`
- `front.v = Y`, `right.v = Y`, `left.v = Y`, `back.v = Y`
- `right.u = Z`, `left.u = −Z`, `top.v = −Z`, `bottom.v = Z`

Dựng lại điểm 3D: lấy X từ front/top, Y từ front/side, Z từ side/top. Khi có nhiều nguồn cho cùng một tọa độ thì lấy trung bình; nếu lệch quá `tolerance · extent` thì đặt nghi vấn và ghi vào `notes`.

### 4.3 Nhận dạng đặc trưng
| Dấu hiệu trong facts.json | Kết luận | Hint |
|---|---|---|
| Có hình tròn tâm (cx, cy), bán kính r ở view A; ở view vuông góc có hai đường nét đứt song song cách nhau 2r, chạy suốt chiều sâu | Lỗ trụ xuyên dọc theo trục sâu của A | `subtract cylinder`, `center = to_world(cx, cy)`, `radius = r`, `axis` = trục sâu của A, `height` ≥ chiều sâu |
| Như trên nhưng nét đứt chỉ dài h | Lỗ mù sâu h | `subtract cylinder`, `height = 2h`, tâm đặt trên bề mặt |
| Front có `about_vertical_centerline ≥ 0.97`, vật thuộc loại tròn xoay (bình, chai, cốc, bánh xe, đèn), hoặc view top là một hình tròn | Vật tròn xoay quanh Y | `revolve {view: "front", axis: "y"}` |
| Silhouette là hình chữ nhật kín toàn bộ extent | Khối đùn, visual hull đã đúng | không cần |
| Chỉ có một view và vật không tròn xoay | Chiều sâu không xác định | `extent` cho trục sâu, ước lượng theo tỉ lệ điển hình của loại vật, ghi lý do vào `notes` |
| Vật đối xứng trái/phải nhưng bản vẽ chỉ có một nửa | Đối xứng gương | `symmetry: ["x"]` |

### 4.4 Ước lượng chiều sâu khi không có view nào quan sát
- Vật tròn xoay thì dùng `revolve`: chiều sâu bằng đường kính profile, engine tự tính.
- Các trường hợp khác dùng tỉ lệ điển hình nhân với kích thước đã đo, ví dụ: ghế sâu ≈ 0.9 × rộng, ô tô dài ≈ 2.3 × rộng, điện thoại dày ≈ 0.11 × rộng. Luôn ghi nguồn của tỉ lệ vào `notes`. Kết quả mang nhãn AI_GENERATED.

## 5. Quy tắc quyết định
1. Số đo có trong facts.json luôn thắng phán đoán.
2. Hint nào cũng phải có lý do trong `notes`, dựa trên số liệu trong facts.json.
3. Dùng ít hint nhất có thể. Không thêm primitive "cho đẹp".
4. `revolve` cần `about_vertical_centerline ≥ 0.97` (hoặc ≥ 0.95 kèm lý do rõ trong `notes`).
5. Không bao giờ dùng `extent` cho một trục mà view đã quan sát (engine sẽ bỏ qua và cảnh báo).

## 6. Schema `hints.json`

```json
{
  "version": 1,
  "author": "claude | codex | human",
  "extent":   {"z": [-0.4, 0.4]},
  "revolve":  {"view": "front", "axis": "y", "center": 0.0},
  "symmetry": ["x"],
  "primitives": [
    {"op": "subtract", "type": "cylinder", "center": [0, 0, 0], "radius": 0.25, "height": 3.2, "axis": "z"},
    {"op": "add",      "type": "box",      "center": [0, 0.6, 0], "size": [0.5, 0.2, 0.5]},
    {"op": "add",      "type": "sphere",   "center": [0, 1.0, 0], "radius": 0.1}
  ],
  "notes": "lý do cho từng quyết định"
}
```

- Mọi tọa độ là tọa độ world, cùng đơn vị với `facts.json` (mét nếu `units == "m"`).
- Thứ tự áp dụng: `extent` (khi dựng lưới) → `revolve` → `symmetry` → `primitives` theo thứ tự trong mảng.
- `revolve.center` là tọa độ của trục quay trên trục bán kính của view (thường là 0 vì gốc nằm ở tâm).
- Engine kiểm tra schema; hint sai sẽ dừng job với thông báo lỗi.

## 7. Kiểm tra kết quả (`metadata.json`)
| Trường | Đạt khi | Nếu không đạt |
|---|---|---|
| `validation.watertight` | `true` | báo lỗi engine, đừng sửa bằng hints |
| `validation.silhouette_iou.<view>` | ≥ 0.98 | hint đang cắt mất phần thấy được (revolve sai trục, primitive subtract quá lớn) hoặc các view mâu thuẫn nhau (xem `views.checks`) |
| `views.checks[].ok` | tất cả `true` | các bản vẽ lệch kích thước; báo người dùng, không tự chỉnh |
| `geometry_confidence` | `estimated` (không hints) hoặc `ai_generated` | ghi rõ cho người dùng |
| `assumptions` | rỗng, hoặc đã được giải thích | nêu từng giả định trong báo cáo |

## 8. Ví dụ

**Hộp có lỗ xuyên** (`examples/box_*.svg`): cả ba trục đều được quan sát, hình tròn ở front đã được gắn `data-v3d-part="hole"` nên engine tự khoét. Không cần hints.

**Bình hoa từ một ảnh** (`examples/vase_front.png`): `describe` cho `unobserved_axes: ["Z"]` và `about_vertical_centerline: 0.999`, nên dùng `revolve` quanh y (`examples/vase_hints.json`):
```bash
python -m vector3d generate --front examples/vase_front.png --hints examples/vase_hints.json -o out/vase
```

## 9. Bước tiếp theo của pipeline
Visual hull là giới hạn trên của hình dạng thật: chỗ lõm bị khuất ở mọi view sẽ không được dựng lại. Các giai đoạn Correspondence, Topology và Surface Fitting (xem `docs/ARCHITECTURE.md`) sẽ thay dần phần này. Khi chúng xong, playbook sẽ có thêm các loại hint mới.
