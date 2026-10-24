# TRELLIS: kiểm tra giấy phép (2026-10-24)

Tra ngày 2026-10-24 qua GitHub API (`/repos/<owner>/<repo>/license`) và model card trên Hugging Face. Đây là thông tin để quyết định, không phải tư vấn pháp lý.

## TRELLIS v1: `microsoft/TRELLIS-image-large` (bản đang cài)

| thành phần | dùng trong luồng hình học của engine? | giấy phép | thương mại |
|---|---|---|---|
| code `microsoft/TRELLIS` | có | MIT | được |
| trọng số `microsoft/TRELLIS-image-large` | có | MIT | được |
| bộ mã hoá ảnh `dinov2_vitl14_reg` (facebookresearch/dinov2) | có | Apache-2.0 | được |
| xformers | có | BSD-3-Clause | được |
| spconv (spconv-cu120) | có | Apache-2.0 | được |
| kaolin (chỉ một hàm kiểm tra trong FlexiCubes) | có | Apache-2.0 | được |
| utils3d | có | MIT | được |
| **nvdiffrast** | **không** (chỉ dùng cho GLB có texture, engine không dùng) | NVIDIA Source Code License: chỉ dùng phi thương mại | **không** |
| **diffoctreerast** | **không** (renderer) | giấy phép Inria/MPII: chỉ cho nghiên cứu | **không** |
| **diff-gaussian-rasterization** (mip-splatting) | **không** (render Gaussian) | giấy phép Inria/MPII: chỉ cho nghiên cứu | **không** |
| rembg / BiRefNet | **không** (engine tự tách nền và đưa ảnh RGBA vào) | — | — |

Kết luận cho v1: luồng ảnh → mesh hình học mà engine dùng chỉ phụ thuộc các thành phần MIT, Apache-2.0 và BSD. Ba thành phần phi thương mại hoặc chỉ cho nghiên cứu **không được cài** (`scripts/setup_trellis_env.sh`) và không được gọi. Nếu sau này cần texture hoặc render qua nvdiffrast, phải xét lại giấy phép trước.

## TRELLIS.2: `microsoft/TRELLIS.2-4B` (chưa cài)

| thành phần | giấy phép | thương mại |
|---|---|---|
| code, trọng số | MIT | được |
| CuMesh, FlexGEMM | MIT | được |
| O-Voxel | nằm trong repo TRELLIS.2 (MIT) | được |
| **bộ mã hoá ảnh `facebook/dinov3-vitl16-pretrain-lvd1689m`** | **DINOv3 License (Meta), repo gated phải xin quyền** | phải đọc điều khoản; Ropy phải tự gửi yêu cầu truy cập |
| **tách nền mặc định `briaai/RMBG-2.0`** | **CC BY-NC 4.0** | **không** (cần thoả thuận thương mại với BRIA) |
| **nvdiffrast, nvdiffrec** (có trong setup chính thức) | NVIDIA Source Code License: chỉ phi thương mại | **không** |

Kết luận cho TRELLIS.2:
- Không dùng RMBG-2.0. Engine tự tách nền bằng model MIT/Apache rồi đưa ảnh RGBA vào.
- DINOv3 là bắt buộc (bộ mã hoá điều kiện), nên dùng TRELLIS.2 đồng nghĩa phải chấp nhận DINOv3 License. Việc đó cần Ropy tự quyết và tự gửi yêu cầu truy cập.
- nvdiffrast và nvdiffrec chỉ cần cho texture và render. Phần hình học có chạy được mà không có chúng hay không thì **chưa kiểm**.
- Ngoài ra, TRELLIS.2 yêu cầu chính thức GPU từ 24 GB.
