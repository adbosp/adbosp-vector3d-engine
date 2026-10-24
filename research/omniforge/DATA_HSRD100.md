# HSRD-100: dữ liệu người thật mặc quần áo cho Omni (2026-10-24)

- Nguồn: `digitalrealitylab/HSRD-100` trên Hugging Face, revision `9cc5e9c138dd310dc88c95c80e5db0a3ecb7971a`. Gồm 10 người, mỗi người 10 tư thế, scan bằng photogrammetry.
- **Giấy phép: CC BY 4.0.** Được dùng thương mại và được sửa đổi, nhưng **bắt buộc ghi công** (đã ghi trong `NOTICE`).
- **Điều khoản của bộ dữ liệu:** không dùng để giám sát, cho mục đích y tế, hay nhận dạng sinh trắc học. Siêu dữ liệu nhân khẩu học của từng người **không được sao chép** vào repo.
- Văn bản ghi công: "HSRD-100: 100 High-Quality 3D Human Scans Dataset, Digital Reality Lab, 2025, https://huggingface.co/datasets/digitalrealitylab/HSRD-100, CC BY 4.0".

## Đã tải (ngoài repo)
- `C:\AI\datasets\HSRD-100`: 3 file manifest và **100 file `*-Scan-LOD1.zip`** (khoảng 100 nghìn polygon, texture 8K), tổng **7,76 GB**.
- Từng file đã kiểm kích thước và SHA256 theo Hugging Face: `download-verification.json`, trạng thái `downloaded_and_verified`.
- **Không** tải scan gốc 5 triệu polygon, LOD0, ảnh chụp, hay dữ liệu RealityCapture.
- Tái lập: `python scripts/download_hsrd100.py`.

## hsrd_v1 (`vector3d/lab/hsrd_v1.py`, nằm ở `C:\AI\datasets\hsrd_v1`)
- **Chia theo người** (seed 0):
  - train: HSR0015, HSR0023, HSR0027, HSR0028, HSR0040, HSR0042, HSR0112, HSR0161 (80 tư thế);
  - test: HSR0081, HSR0102 (20 tư thế).
  - Người ở tập test không xuất hiện trong tập train.
- **Hệ toạ độ:** bản scan có trục Z hướng lên, mặt hướng về +Y. Chuyển sang hệ của engine bằng (x, y, z) → (−x, z, y): trục +Y lên, mặt hướng +Z, đây là phép quay thật (det +1). Đặt tâm X/Z về 0, bàn chân ở y = 0, giữ đơn vị mét. Đã kiểm bằng mắt trên nhiều người và có test.
- **Ảnh render** dùng texture thật của scan (màu đỉnh lấy mẫu từ texture 8K theo UV). Các archive không có file `.mtl`, nên texture được đọc trực tiếp. Có 4 góc tham chiếu và 4 góc giữ lại, ảnh 512².
- **test:** 20 mẫu, không tăng cường dữ liệu.
- **train:** 240 mẫu (mỗi tư thế 3 bản). Mỗi bản xoay ngẫu nhiên ±20° quanh trục đứng và đổi hướng nguồn sáng chính. Mỗi mẫu có một cặp train `pairs/<mẫu>`: `cond.png`, cộng `surface.npz` lấy **trực tiếp từ bề mặt scan GT** (nhãn `GT_SCAN`, CC BY), cộng `provenance.json`. Màu nền không đổi, vì bước điều kiện của TripoSG thay nền qua kênh alpha.
- Thời gian dựng 459 s trên CPU.

## Chưa làm
- Chạy TripoSG, các ứng viên teacher và `lab omni-distill` trên hsrd_v1.
- Train LoRA từ các cặp GT-scan.
- Hai việc này đang tạm dừng theo yêu cầu của Ropy (2026-10-24).
