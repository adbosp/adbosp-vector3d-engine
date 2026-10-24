# OmniForge 3D (Omni): kế hoạch nghiên cứu, ưu tiên Human

Viết ngày 2026-10-24 theo yêu cầu của Ropy. Mọi dữ kiện về model bên dưới được tra trên trang chính thức hôm nay; ô nào ghi "chưa rõ" là trang gốc không nói, không đoán.
**Chưa có con số benchmark nào cho OmniForge.** Mọi khẳng định "tốt hơn" phải đến từ benchmark Human (Giai đoạn 0) chạy thật.

## 1. Mục tiêu

Một model duy nhất cho người dùng (OmniForge 3D), giữ tốc độ của TripoSG trên RTX 3060, nhưng:
- bàn tay đủ 5 ngón, đúng giải phẫu, kể cả khi trong ảnh các ngón dính liền (tự "tưởng tượng" ra ngón hợp lý);
- bàn chân, nếp vải, nếp nhăn mặt, mắt, môi, lỗ mũi, lỗ tai có hình học thật, không bị làm mịn mất;
- không train thêm đồ vật vuông tròn (bàn, ghế, hộp). LoRA hộp hiện có (`research/triposg_lora_rtx3060`) giữ nguyên làm tùy chọn.

## 2. Giới hạn kỹ thuật phải nói trước

1. **Không thể copy layer giữa các kiến trúc khác nhau.** TripoSG là VAE dạng tập vector cộng DiT rectified-flow 1,5B. Hi3D v1.0 là VAE thưa (Sparc3D) cộng hai transformer 512/1024. TRELLIS.2 dùng O-Voxel. Trọng số của chúng không nạp được vào TripoSG. Chỉ còn hai đường: **chưng cất đầu ra** (teacher tạo mesh hoặc nhãn, student học theo) và **chưng cất đặc trưng** (student khớp feature ẩn của teacher, khó hơn nhiều).
2. **Giả thuyết: một phần trần chi tiết của TripoSG có thể nằm ở VAE và lưới trích xuất, không chỉ ở DiT.** LoRA hiện tại chỉ chỉnh DiT. Nếu VAE không tái tạo được một bàn tay chi tiết từ chính mesh chuẩn, thì fine-tune riêng DiT khó có khả năng tạo ra chi tiết đó. Đây là cổng kiểm tra G1. Kết quả đầu tiên trên một mẫu tổng hợp nằm trong `MEASUREMENTS.md`; đó là bằng chứng ban đầu, chưa phải chứng minh.
3. **Đang tự làm mất chi tiết:** profile `balanced` giảm mesh từ khoảng 1.013.704 mặt xuống 50.000 mặt (`research/triposg_rtx3060/summary.json`, crate). Ngón tay và nếp vải bị mất ngay ở bước này. Phải đo lại với `--faces -1` trước khi đổ lỗi cho model.
4. Student nhỏ không chắc giữ được toàn bộ năng lực của teacher. Nó có thể nhanh hơn nhưng kém hơn. Mỗi bước đều phải có benchmark trước và sau.
5. Model đóng (Seedance, Meshy, Astra, Muse, Opus 5.5) **không có trọng số**. Chỉ dùng được đầu ra qua API, và điều khoản thường cấm dùng đầu ra để train model cạnh tranh. Ví dụ, điều khoản của Anthropic cấm dùng đầu ra của Claude để train model AI cạnh tranh. Vì vậy Opus 5.5 được dùng làm **bộ điều phối (agent)**, không làm teacher cho trọng số.

## 3. Ma trận teacher (5 tiêu chí)

| Teacher | 1. Năng lực nổi bật | 2. Bù điểm yếu nào của TripoSG | 3. Dạng kiến thức chuyển giao | 4. Khả năng chưng cất vào student khác kiến trúc | 5. Chi phí và trạng thái |
|---|---|---|---|---|---|
| **Hi3D v1.0** ([HF](https://huggingface.co/hi3d-official/Hi3Dv1.0), [code](https://github.com/hi3d-official/Hi3Dv1.0), phát hành 2026-10-23) | Tinh chỉnh hình học: **ảnh + mesh thô → mesh chi tiết** (512³ / 1024³ / 1536³), chỉ hình học, không màu | Mất chi tiết nhỏ (ngón, nếp vải, mặt) | Mesh đầu ra làm target (output distillation) | Tốt qua đầu ra; không copy được trọng số | Trọng số 7,2 GB (2 transformer 3,68 GB + VAE 0,38 GB) cộng DINOv2-L và CLIP ViT-L/14. Chỉ Linux, GPU CC ≥ 8.0 (3060 là 8.6, đạt), CUDA 12.0–12.5, flash-attn. VRAM trên card nhỏ **chưa ai đo**; 1536³ chỉ được kiểm trên H20 96 GB; có `--sequential-cpu-offload`. License Hi3D Open 1.0: **cho phép dùng đầu ra để train model khác** và cho phép fine-tune; thương mại phải ghi "Powered by Hi3D 1.0"; trên 200.000 MAU hoặc 1 triệu USD/tháng phải xin license riêng; không được gọi bản phái sinh là bản Hi3D chính thức. Không có code train. |
| **SAM 3D Body** (Meta, [model card](https://huggingface.co/facebook/sam-3d-body-dinov3)) | Dựng người toàn thân từ 1 ảnh: thân, bàn chân, **tư thế bàn tay**; mesh theo Momentum Human Rig (MHR) | Ngón dính, sai số ngón, tay không đúng giải phẫu | Cấu trúc: khung xương, tham số tay, mesh MHR làm khuôn giải phẫu; nhãn kiểm tra (5 ngón/tay) | Tốt: dùng làm ràng buộc và target vùng tay, không cần copy trọng số | Gated: Ropy phải tự đồng ý điều khoản SAM License trên HF (tôi không được thay mặt chấp nhận điều khoản). Điều khoản về dùng đầu ra để train **chưa đọc được**, phải kiểm trước khi đưa vào dữ liệu train. VRAM chưa rõ. Mức chi tiết khuôn mặt của mesh: trang không nói. |
| **MoGe-2** ([microsoft/MoGe](https://github.com/microsoft/moge)) | Point map có thang đo mét, **normal map sắc nét** từ 1 ảnh | Mặt nhìn thấy được bị mịn (nếp nhăn mặt, nếp vải) | Normal và depth làm loss khi render khả vi; nhãn | Rất tốt: loss ảnh không phụ thuộc kiến trúc | Nhỏ (35M–331M, bản `-normal`), chạy được trên 3060. License: bên thứ ba ghi MIT, **chưa xác nhận trên trang chính thức**. |
| **Qwen3-VL** | Suy luận hình ảnh: nhận bộ phận, đếm ngón, mô tả che khuất | Pipeline hiện "mù ngữ nghĩa" | Nhãn hoặc gợi ý (hints.json), định tuyến MoE | Không chưng cất hình học; dùng như module | Có bản nhỏ chạy local. License từng bản cần kiểm. |
| **TRELLIS.2** (Microsoft, 4B, MIT) | Hình khối 3D tổng quát, biểu diễn O-Voxel | Thêm một teacher hình khối thứ hai để bỏ phiếu | Mesh đầu ra | Tốt qua đầu ra | Cần ≥ 24 GB và Linux theo trang chính thức, nên chỉ chạy trên GPU thuê. |
| **Hunyuan3D 2.x** | Hình khối | — | — | **Loại**: license cấm dùng đầu ra để cải thiện model AI khác và không hiệu lực ở EU/UK/Hàn Quốc (`research/model_matrix.md`) | Không dùng làm teacher |
| **Seedance / Meshy / Astra / Muse** | Đóng, chỉ có API | Có thể: video quay vòng quanh người (đa góc nhìn) | Đầu ra API | Chỉ khi điều khoản cho phép dùng để train | **Chưa xác định**: cần Ropy chỉ rõ đúng sản phẩm nào (Astra và Muse có nhiều sản phẩm trùng tên) và cần đọc điều khoản từng bên |
| **Opus 5.5** | Lập kế hoạch, viết code, điều phối công cụ | Không có vòng tự sửa lỗi | Hành vi agent (bộ điều phối) | Không chưng cất vào trọng số (điều khoản) | Đang dùng sẵn |

## 4. Kiến trúc OmniForge 3D v0

```
ảnh ─┬─ Tri giác (VERIFIED/ESTIMATED): mask, silhouette, MoGe-2 normal/depth, Qwen3-VL nhãn bộ phận,
     │                                SAM 3D Body khung xương + tay MHR
     ├─ Khối thô: TripoSG + LoRA chuyên gia "human"   (student, chạy trên 3060)
     ├─ Chi tiết: Hi3D refine (teacher, giai đoạn đầu) → về sau thay bằng student chưng cất
     ├─ Kiểm tra neuro-symbolic: nằm trong visual hull, 5 ngón/tay khớp SAM 3D Body,
     │                           giới hạn khớp, đối xứng, kín nước
     ├─ Render khả vi: tối ưu dịch chuyển đỉnh để normal và silhouette khớp MoGe-2 và mask
     └─ Agent (vòng sửa lỗi): kiểm tra fail thì đổi seed, refine cục bộ vùng tay hoặc mặt,
                              hoặc ghép tay MHR đã khớp; tối đa N vòng, ghi lý do từng vòng
```

- **MoE trung thực ở v0:** một bộ định tuyến (nhãn Qwen3-VL) chọn giữa các LoRA chuyên gia (human toàn thân; sau này tay, mặt, vải). Đây là "MoE mức LoRA", chưa phải MoE có cổng học trong từng layer; MoE thật chỉ làm khi v0 có số liệu.
- **Chưng cất nhiều teacher, nạp điểm có chọn lọc:** mỗi mẫu train là mesh đã hợp nhất. Hi3D cho chi tiết, vùng tay lấy theo SAM 3D Body nếu teacher hình khối làm dính ngón, MoGe-2 làm loss mặt nhìn thấy. Mẫu nào không qua cổng chất lượng (sai số ngón, lệch silhouette, mesh hỏng) thì **bỏ**, không đưa vào train.
- Nhãn dữ liệu giữ nguyên quy tắc engine: đầu ra teacher và student đều là `AI_GENERATED`; silhouette đo được là `VERIFIED`.

## 5. Các giai đoạn và cổng đi tiếp / dừng

| GĐ | Việc | Chạy ở đâu | Cổng (đo thật) |
|---|---|---|---|
| **0** | Benchmark Human: 10–20 người có mesh chuẩn, gồm cảnh cận tay và mặt. Thêm metric vùng (Chamfer riêng vùng tay và mặt, đếm ngón) | 3060 | Có bộ dữ liệu được phép dùng (cần Ropy quyết định, mục 6) |
| **0b** | Thắng nhanh: chạy TripoSG `--faces -1` và lưới marching cubes dày hơn, đo lại | 3060 | Bao nhiêu chi tiết mất là do giảm mặt, không phải do model |
| **1** | Hi3D chạy thử trên 3060 qua WSL2 Ubuntu 24.04 (đã có sẵn), 512³ có offload, cấp lên 1024³ nếu được | 3060 | Chạy được không, thời gian, VRAM đỉnh, tay và mặt có thêm chi tiết thật không (ảnh và metric vùng) |
| **G1** | Đưa mesh chi tiết của Hi3D qua VAE TripoSG (encode rồi decode) và đo phần chi tiết còn lại | 3060 | Nếu VAE làm mất ngón ở tỉ lệ toàn thân (đã thấy trên 1 mẫu tổng hợp), ưu tiên thử dựng đa tỉ lệ và tầng refine riêng trước khi đầu tư vào LoRA trên DiT; cần kiểm tra thêm bằng scan thật và latent sample |
| **2** | Lớp tri giác: MoGe-2, Qwen3-VL bản nhỏ, SAM 3D Body | 3060 | Đếm ngón đúng trên benchmark |
| **3** | Tạo dữ liệu teacher: ảnh người → TripoSG → Hi3D (và TRELLIS.2 nếu có GPU thuê) → cổng chất lượng → target đã chọn | 3060 hoặc GPU thuê | Tỷ lệ mẫu qua cổng, có xem mẫu bằng mắt |
| **4** | Train LoRA chuyên gia "human" (hoặc student refine, tùy G1) | 3060 | Benchmark Human trước và sau, cùng seed |
| **5** | Ghép OmniForge 3D v0: định tuyến, kiểm tra, render khả vi, agent | 3060 | Thắng TripoSG gốc trên benchmark Human ở metric vùng tay và mặt mà không thua toàn thân |

## 5b. Hướng đã chốt (2026-10-24, Ropy giao toàn quyền về kỹ thuật)

Mục tiêu: kết quả benchmark vượt đối thủ, với tốc độ sinh tương đương.

1. **Bỏ InstantMesh.** Độ phân giải thấp của từng góc nhìn là giới hạn do kiến trúc, nên tinh chỉnh không đáng chi phí.
2. **Hai tầng:**
   - (a) **Pipeline teacher chậm, tự dựng:**
     - TripoSG dựng toàn thân;
     - giải phẫu MHR (RTMW+MHR bây giờ, SAM 3D Body khi được duyệt; xem `ANATOMY.md`);
     - **dựng riêng tay và mặt ở khung cận cảnh**: cắt quanh các khớp MHR, chạy TripoSG trên từng vùng ở chế độ đầy khung, rồi căn mỗi vùng theo MHR;
     - ghép các vùng vào thân, giữ trong visual hull;
     - Hi3D thêm chi tiết bề mặt;
     - bộ trọng tài neuro-symbolic kiểm tra: nằm trong hull, 5 ngón khớp MHR, kín nước.
   - (b) **Về sau:** chưng cất các kết quả đã được trọng tài chấp nhận vào một student nhanh (TripoSG + LoRA theo vùng), để một lượt chạy tiến gần chất lượng của pipeline.
3. **Thứ tự việc:**
   - sửa `invoke_runner` (xong, request `-03`);
   - hoàn thành RTMW+MHR (v0 xong, `ANATOMY.md`);
   - làm thử luồng cận cảnh tay trên bàn tay tổng hợp, so sánh ba cách trên cùng một tay và cùng metric: TripoSG toàn thân, TripoSG cận cảnh, TripoSG cận cảnh + Hi3D;
   - **human_v1**: người tổng hợp sinh từ MHR (Apache-2.0), có GT giữ riêng để đo mọi khẳng định.
4. Mọi kết quả ghi là quan sát hoặc giả thuyết cho tới khi đo trên human_v1.

## 6. Cần Ropy quyết định

1. **Tải Hi3D để chạy thử (GĐ 1):** khoảng 7,2 GB từ `hi3d-official/Hi3Dv1.0`, cộng DINOv2-large (khoảng 1,2 GB) và CLIP ViT-L/14 (khoảng 1,7 GB), cài trong WSL2 Ubuntu-24.04. Ổ C còn 85 GB.
2. **Dữ liệu người:** tự cung cấp ảnh hoặc scan; hoặc dùng bộ công khai (THuman2.x, CustomHumans, 2K2K), phần lớn chỉ cho nghiên cứu và phải gửi đơn xin; hoặc chỉ dùng ảnh do AI tạo. Bộ nào có mesh chuẩn mới làm benchmark được.
3. **GPU thuê** cho TRELLIS.2 và Hi3D 1024³/1536³ nếu 3060 không đủ: có ngân sách không?
4. **SAM 3D Body:** Ropy tự chấp nhận điều khoản trên Hugging Face (gated), rồi cấp token.
5. **Model đóng:** Astra và Muse là sản phẩm nào cụ thể; có chấp nhận chỉ dùng chúng nếu điều khoản cho phép không.

## Nguồn
- Hi3D v1.0: https://huggingface.co/hi3d-official/Hi3Dv1.0, https://github.com/hi3d-official/Hi3Dv1.0, file LICENSE (Hi3D Open License Agreement 1.0); danh sách file qua API HF (tổng 7.735.450.033 byte)
- SAM 3D Body: https://huggingface.co/facebook/sam-3d-body-dinov3
- MoGe-2: https://github.com/microsoft/moge
- Hunyuan3D và TRELLIS.2: `research/model_matrix.md` (tra ngày 2026-10-22)
