# Vector3D Engine — Kiến trúc

> Trạng thái: v0.1. Đã triển khai giai đoạn 1–2 (SVG Parser, Coordinate Normalization). Các giai đoạn còn lại mới ở mức thiết kế.

## 1. Nguyên tắc

1. **Toán học trước, AI sau.** Hình học xác định được bằng toán phải được dựng bằng toán. AI chỉ giải quyết phần thiếu, không chắc chắn hoặc sáng tạo.
2. **Mọi dữ liệu đều mang độ tin cậy.** Ba mức `VERIFIED` → `ESTIMATED` → `AI_GENERATED` (`vector3d/core/confidence.py`). Dữ liệu mức thấp không bao giờ ghi đè dữ liệu mức cao. Mỗi giá trị suy luận đều ghi lại nguồn (`source`).
3. **Giữ đường cong chính xác.** Bézier được giữ dạng giải tích suốt pipeline; phép affine biến điểm điều khiển thành điểm điều khiển nên không mất độ chính xác. Chỉ lấy mẫu (sampling) khi một giai đoạn yêu cầu rõ ràng.
4. **Module độc lập, gọi được qua API/MCP.** Mỗi giai đoạn có input/output kiểu rõ ràng, có thể chạy riêng lẻ. Agent (Claude, Codex…) điều khiển từng bước.
5. **Không thay thế Blender.** Engine phân tích, dựng mesh, kiểm tra, xuất file. Blender/Unity/Unreal là hậu kỳ, không phải backend.

## 2. Pipeline

| # | Giai đoạn | Module | Dùng LLM? | Trạng thái |
|---|-----------|--------|-----------|-----------|
| 1 | SVG Geometry Parser | `vector3d.svg.parser` | Không | ✅ |
| 2 | Coordinate Normalization | `vector3d.normalize` | Không | ✅ |
| 3 | AI Semantic Geometry Understanding | `vector3d.semantics` | Có | Thiết kế |
| 4 | Cross-view Correspondence Solver | `vector3d.correspond` | Không | Thiết kế |
| 5 | Confidence-aware 3D Reconstruction | `vector3d.reconstruct` | Không | Thiết kế |
| 6 | Coarse 3D Surface Reconstruction | `vector3d.surface.coarse` | Không | Thiết kế |
| 7 | AI SVG Topology Blueprint | `vector3d.topology.blueprint` | Có | Thiết kế |
| 8 | Topology Graph Compiler | `vector3d.topology.compiler` | Không | Thiết kế |
| 9 | 3D Surface Fitting Solver | `vector3d.surface.fit` | Không | Thiết kế |
| 10 | Geometry + Topology Validation | `vector3d.validate` | Không | Thiết kế |
| 11 | Iterative Refinement | `vector3d.refine` | Có (điều phối) | Thiết kế |
| 12 | Export OBJ / GLB / Metadata | `vector3d.export` | Không | Thiết kế |

Danh sách này nằm trong `vector3d/pipeline.py` (`STAGES`), là nguồn duy nhất để lớp API/MCP liệt kê công cụ.

## 3. Quy ước tọa độ

**World:** hệ tay phải, đơn vị mét. `+X` phải, `+Y` lên, `+Z` hướng về người xem hình chiếu Front (trùng quy ước glTF: mặt trước model nhìn về `+Z`, nên xuất GLB không cần đổi trục).

**Mặt phẳng hình chiếu:** `u` sang phải, `v` lên trên. SVG có trục y hướng xuống nên `v = −y_svg`.

| View | Camera đặt tại | Nhìn theo | `+u` = | `+v` = |
|------|----------------|-----------|--------|--------|
| Front | `+Z` | `−Z` | `+X` | `+Y` |
| Back | `−Z` | `+Z` | `−X` | `+Y` |
| Right | `−X` | `+X` | `+Z` | `+Y` |
| Left | `+X` | `−X` | `−Z` | `+Y` |
| Top | `+Y` | `−Y` | `+X` | `−Z` |
| Bottom | `−Y` | `+Y` | `+X` | `+Z` |

- Với mọi view: `right × up = hướng về camera`, ma trận cơ sở có định thức +1 (đã test).
- **Cặp đối diện** (Front/Back, Left/Right, Top/Bottom) dùng chung hai trục world, trục ngang bị lật và hướng nhìn ngược nhau trên cùng một trục sâu. Chúng là hai quan sát của cùng một trục, không phải hai trục độc lập. Engine dùng cặp đối diện để kiểm chứng chéo (`mirror_to_opposite`, `check_consistency`), và về sau để phân biệt cạnh thấy/khuất.
- **Left/Right theo bên của vật thể** (đã chốt với người dùng). Vật thể nhìn về `+Z`, nên bên phải của vật thể nằm ở `−X`, tức bên trái người nhìn Front. View Right đặt camera ở `−X`.
- Hai view không đối diện luôn chung đúng một trục world (`shared_axes`): Front–Top chung X, Front–Side chung Y, Top–Side chung Z.

## 4. Giai đoạn 1: SVG Geometry Parser

`parse_svg(path) -> SvgDocument`

- **Phần tử:** `path` (đủ M L H V C S Q T A Z, tương đối/tuyệt đối, lặp lệnh ngầm, cờ arc viết dính `a1 1 0 01 2 2`), `line`, `polyline`, `polygon`, `rect` (cả bo góc), `circle`, `ellipse`, `use`/`symbol`, `text`.
- **Arc → cubic** theo SVG 1.1 F.6, mỗi đoạn ≤ 90°. Hình tròn/ellipse giữ thêm `primitive` (tâm, bán kính) để các giai đoạn sau nhận dạng trụ/lỗ chính xác.
- **Transform lồng nhau** (`matrix/translate/scale/rotate/skewX/skewY`) được nhân dồn theo thứ tự đặc tả. Contour đầu ra ở không gian user của root (đơn vị viewBox).
- **Phân cấp layer/group:** `layer_path` lấy từ `inkscape:label` hoặc `id`.
- **Ngữ nghĩa:** `id`, `class`, mọi thuộc tính `data-v3d-*`; `data-v3d-part` và `data-v3d-role` được kế thừa từ group cha. Nét đứt (`stroke-dasharray`) được đánh dấu `line_style=dashed` (thường là cạnh khuất).
- **Phần tử không phải hình học** (`data-v3d-role` = annotation / construction / dimension / origin) được tách riêng khỏi `contours`.
- Phần tử ẩn (`display:none`, `visibility:hidden`) bị bỏ qua. Path lỗi được ghi cảnh báo, không làm hỏng cả file.

### Thang đo (không bao giờ coi pixel là kích thước thật)

Thứ tự ưu tiên, mỗi nguồn kèm độ tin cậy:

1. `data-v3d-scale` trên root (mét / đơn vị user) → `VERIFIED`.
2. Kích thước ghi chú `data-v3d-dimension="250mm"` trên đường đo → `VERIFIED` nếu các kích thước khớp nhau trong 0,5%, ngược lại `ESTIMATED` + cảnh báo.
3. Đơn vị vật lý của tài liệu (`width="200mm"` + viewBox) × `data-v3d-drawing-scale="1:10"` → `VERIFIED`; không có tỉ lệ bản vẽ thì giả định 1:1 và chỉ là `ESTIMATED`.
4. `px` hoặc không đơn vị → **unknown**.

Nếu các nguồn mâu thuẫn, dùng nguồn ưu tiên cao hơn và ghi cảnh báo.

## 5. Giai đoạn 2: Coordinate Normalization

`normalize_views({view: SvgDocument}) -> NormalizedScene`

1. Kiểm tra `data-v3d-view` khai báo trong file khớp với vị trí được truyền vào.
2. **Lan truyền thang đo:** view chưa có thang đo lấy từ view đã có thông qua chiều dài trên trục world chung → `ESTIMATED`, ghi nguồn `propagated`.
3. Nếu không view nào có thang đo vật lý: giữ **drawing units**, lấy view đầu làm chuẩn và quy các view khác về cùng đơn vị đó; `scene.units = "drawing_units"`.
4. **Gốc tọa độ:** phần tử `id="v3d-origin"` hoặc `data-v3d-role="origin"` → `VERIFIED`; nếu không, tâm bounding box → `ESTIMATED`.
5. Biến đổi một affine duy nhất `SVG → (u, v)`; `camera.to_world(uv, depth)` đưa sang world.
6. **Kiểm tra nhất quán:** mọi cặp view so chiều dài trên trục chung; sai lệch > `tolerance` (mặc định 1%) bị gắn cờ.

## 6. Các giai đoạn tiếp theo (thiết kế)

- **3. Semantic Understanding (LLM tùy chọn):** gán nhãn bộ phận (bánh xe, thân, lỗ…), phân loại cạnh (silhouette / cạnh gãy / cạnh khuất / đường trang trí). Đầu vào là đồ thị contour có cấu trúc, không phải ảnh pixel. Kết quả luôn là `AI_GENERATED` và chỉ làm *gợi ý* cho giai đoạn 4.
- **4. Correspondence Solver (thuần toán):** ghép đỉnh/cạnh giữa các view dựa trên tọa độ trùng khớp trên trục chung (ràng buộc kiểu epipolar trực giao), nhất quán cặp đối diện và đồ thị tương thích; giải bằng gán ghép có trọng số (Hungarian / ILP nhỏ).
- **5. Reconstruction:** điểm 3D có ≥ 2 view xác nhận → `VERIFIED`; suy từ đối xứng / quay tròn (lathe) / đùn (extrude) → `ESTIMATED`; phần không có dữ liệu → `AI_GENERATED`, tách riêng để người dùng duyệt.
- **6. Coarse Surface:** visual hull từ giao các silhouette đùn theo trục sâu, sau đó trừ các đặc trưng đã nhận dạng (lỗ trụ…).
- **7–8. Topology Blueprint + Compiler:** AI đề xuất bố cục edge loop / quad flow dưới dạng SVG; compiler kiểm tra và dịch thành đồ thị topology (đỉnh, cạnh, mặt quad) có ràng buộc.
- **9. Surface Fitting:** bình phương tối thiểu có ràng buộc: kéo lưới topology khớp bề mặt thô, giữ đúng các điểm `VERIFIED`.
- **10. Validation:** manifold, normals, tự cắt, mặt suy biến, tỉ lệ quad, sai số tái chiếu (chiếu mesh ngược về từng view, so với SVG gốc bằng khoảng cách Hausdorff).
- **11. Refinement:** lặp 9–10 tới khi sai số tái chiếu < ngưỡng; LLM chỉ điều phối khi cần quyết định sáng tạo.
- **12. Export:** OBJ, GLB (trục đã khớp glTF), và file metadata JSON (nguồn gốc, độ tin cậy từng đỉnh/mặt, báo cáo kiểm tra).

## 7. Giao diện cho agent

- **Python API:** mỗi giai đoạn là một hàm thuần (input/output dataclass), dễ test.
- **CLI:** `python -m vector3d inspect --front f.svg --right r.svg --top t.svg [--json]`.
- **Giao diện web:** `python -m vector3d ui`, server HTTP thuần Python (`vector3d/ui/server.py`) cùng API JSON `POST /api/inspect`, `GET /api/stages`, `GET /api/examples`. Trang dùng three.js cho 3D viewer (xem view trong hộp kính, mở OBJ/GLB/STL). Lớp MCP sau này có thể dùng lại chính các hàm `inspect()` / `report.*`.
- **MCP server (kế hoạch):** mỗi mục trong `STAGES` thành một tool; trạng thái trung gian lưu dạng JSON có ID để agent gọi lại từng bước, so sánh, hoặc chạy lại một giai đoạn.

## 8. Câu hỏi mở

1. Prompt gốc dừng ở mục 3: còn các mục tiếp theo (yêu cầu topology, định dạng metadata, MCP, hiệu năng…) không?
2. Quy ước chiếu: hệ thống có cần đọc bản vẽ kỹ thuật kiểu first-angle/third-angle với nhiều view trong một file SVG không, hay mỗi view một file?
3. Ngôn ngữ/đóng gói: giữ Python + numpy cho lõi (đề xuất), hay cần thêm lõi C++/Rust cho hiệu năng?
4. Repo GitHub nào để đẩy code lên?
