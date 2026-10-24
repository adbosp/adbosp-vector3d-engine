# Kiểm kê model và dữ liệu trên ROPY-PC (2026-10-24)

Đo trước khi xoá, theo yêu cầu "dọn bớt các mô hình không dùng tới trên máy". Ổ C: đã đầy 0 byte lúc 11:5x (WSL chuyển sang chỉ đọc). Xoá cache pip của engine (9,4 GB) và chạy `wsl --shutdown` giúp WSL tự phục hồi. Ổ WSL (`ext4.vhdx`) nằm trên C: và có dung lượng **57,9 GB**.

## Giữ lại (sản phẩm đang dùng, hoặc dữ liệu của Ropy)
| mục | đường dẫn | dung lượng |
|---|---|---|
| repo và trọng số TripoSG | `C:\ai3d\TripoSG` | 7,43 GB |
| môi trường TripoSG | `C:\ai3d\triposg-env` | 5,15 GB |
| LoRA | `bench\ft_run\lora` (trong `bench\ft_run`, 0,04 GB) | 0,04 GB |
| môi trường engine (.venv) | `C:\adbosp-vector3d-engine\.venv` | 0,63 GB |
| trọng số tách nền IS-Net/BiRefNet (rembg) | `%USERPROFILE%\.u2net` | 0,33 GB |
| Depth Anything V2 ONNX và cache của engine | `%USERPROFILE%\.cache\vector3d` | 0,09 GB |
| môi trường giải phẫu (RTMW + MHR, chế độ Omni trong UI dùng) | WSL `~/micromamba/envs/anatomy` | 9,1 GB |
| asset MHR | WSL `~/ai3d/mhr` | 4,7 GB |
| model ONNX của RTMW | WSL `~/.cache/rtmlib` | 0,32 GB |
| micromamba (binary) | WSL `~/ai3d/bin` | 0,02 GB |
| dữ liệu HSRD-100 (Ropy đã duyệt) | `C:\AI\datasets\HSRD-100\data` | 7,23 GB |
| bộ hsrd_v1 đã dựng | `C:\AI\datasets\hsrd_v1` | 1,13 GB |
| human_v1, benchmark và dữ liệu train | `bench\human_v1`, `bench\data`, `bench\ft_data` | khoảng 0,6 GB |
| SAM 3D Body | `C:\AI\models\sam-3d-body-dinov3` | chưa có (chưa tải) |

## Xoá (không dùng tới, hoặc là cache sinh lại được)
| mục | đường dẫn | dung lượng |
|---|---|---|
| môi trường TRELLIS v1 (bị đánh giá là quá yếu) | WSL `~/micromamba/envs/trellis` | 8,3 GB |
| trọng số TRELLIS-image-large | WSL `~/ai3d/weights/TRELLIS-image-large` | 3,1 GB |
| repo TRELLIS | WSL `~/ai3d/TRELLIS` | 1,2 GB |
| torch hub DINOv2 ViT-L/14 reg (chỉ TRELLIS dùng) | WSL `~/.cache/torch` | 1,2 GB |
| bản clone Direct3D-S2 dở dang (hỏng khi ổ đầy) | WSL `~/ai3d/Direct3D-S2` | 64 KB |
| cache pip trong WSL | WSL `~/.cache/pip` | 7,3 GB |
| cache gói micromamba | WSL `~/micromamba/pkgs` (`micromamba clean -a`) | 3,2 GB |
| cache pip của engine | `%LOCALAPPDATA%\pip\Cache` | 9,4 GB (đã xoá khi xử lý sự cố) |
| bản giải nén HSRD-100 (cache, sinh lại tự động từ các file zip) | `C:\AI\datasets\HSRD-100\unpacked` | 7,89 GB |

Mã backend TRELLIS (`vector3d/lab/backends/trellis.py`) được giữ nguyên: thiếu môi trường thì backend trả `blocked`, có test kèm theo.

## Hỏi Ropy trước (chưa xoá)
| mục | đường dẫn | dung lượng |
|---|---|---|
| Hi3D v1.0: môi trường | WSL `~/micromamba/envs/hi3d` | 7,3 GB |
| Hi3D v1.0: trọng số | WSL `~/ai3d/weights/Hi3Dv1.0` | 7,2 GB |
| Hi3D v1.0: bộ mã hoá CLIP-L và DINOv2-L | WSL `~/ai3d/weights/clip-vit-large-patch14`, `dinov2-large` | 1,6 GB + 1,2 GB |
| Hi3D v1.0: repo | WSL `~/ai3d/Hi3Dv1.0` | 0,03 GB |
| các lượt chạy teacher (mesh trung gian; Codex dùng để tính lại số khi review) | `bench\omni_teacher`, `bench\omni_teacher_contour`, `bench\omni_teacher_skel` | 1,15 + 1,62 + 1,62 GB |
| các lượt chạy so sánh Alicia (mesh; bảng hình và số đã lưu ở `research\compare_alicia`) | `bench\compare_alicia`, `bench\compare_alicia2` | 0,53 + 0,58 GB |
| thư mục chạy của UI (có kết quả Ropy tạo bằng giao diện) | `runs\` | 0,57 GB |
| cache uv trong WSL (không phải do tôi tạo) | WSL `~/.cache/uv` | 0,5 GB |

## Kết quả (2026-10-24)
| mục đã xoá | dung lượng giải phóng |
|---|---|
| cache pip của engine (Windows, lúc xử lý sự cố) | 9,4 GB trên C: |
| bản giải nén HSRD-100 (Windows) | 7,89 GB trên C: |
| môi trường, trọng số, repo TRELLIS; torch hub; bản clone Direct3D-S2; cache pip WSL; `micromamba clean -a` | 23,6 GB **bên trong** WSL |

- **Trên C: (Windows): 17,3 GB.** Ổ C: từ 0 GB trống lên **44,7 GB** trống. Phần chênh còn lại là các file tạm hệ thống được giải phóng sau sự cố.
- **Bên trong WSL: 23,6 GB.** File `ext4.vhdx` (57,9 GB) **không tự co lại**, nên chỗ này chỉ dùng lại được cho chính WSL. Muốn trả về cho Windows thì phải nén ổ ảo, ví dụ bật `wsl --manage Ubuntu-24.04 --set-sparse true`, hoặc dùng `Optimize-VHD` với quyền admin. Đây là thay đổi cấu hình hệ thống, nên **chờ Ropy đồng ý**.
- Các mục "Hỏi Ropy" ở trên vẫn giữ nguyên. Riêng Hi3D tổng cộng khoảng 17,3 GB.
