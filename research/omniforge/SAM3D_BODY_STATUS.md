# SAM 3D Body: BLOCKED (2026-10-24)

- Repo: `facebook/sam-3d-body-dinov3` (gated, SAM License).
- Trạng thái: Ropy đã gửi yêu cầu truy cập và đang **chờ tác giả duyệt**. Trên máy này chưa có trọng số, chưa chạy lần nào.
- Backend: `vector3d/lab/backends/sam3d_body.py` (`get_backend("sam3d_body")`). Khi thiếu trọng số, `check()` trả `blocked` với lý do "weights not downloaded: Hugging Face access awaiting author review". Nếu có trọng số nhưng chưa có runner đã kiểm chứng, vẫn trả `blocked`. Backend không bao giờ trả mesh thay thế.
- Không dùng bản re-upload của cộng đồng (ví dụ gokayokyay/sam-3d-body, Blendcap/sam-3d-body), vì chúng vượt qua cổng duyệt của tác giả.

## Nhận xét về bộ tải của Codex (`Download-SAM3D.ps1`, `download_model.py`)
Đã đọc toàn bộ cả hai file. Nhìn chung đúng: kiểm tra quyền trước khi tải, ghim revision (`info.sha`), kiểm tra kích thước, dùng env riêng. Có ba điểm cần sửa:
1. **Ghi nhận xác minh quá mức:** nếu Hub không trả hash LFS (tài liệu handoff nói hash bị ẩn khi repo còn gated), script bỏ qua bước so SHA256 nhưng vẫn ghi `status: downloaded_and_verified`. Nên ghi theo từng file `sha256_checked: true/false`, và chỉ ghi "verified" khi mọi file LFS đều đã được so hash.
2. `pip install huggingface_hub` không ghim phiên bản. Nên ghim phiên bản để chạy lại được như cũ.
3. `login()` mở lời nhắc tương tác. Phải để Ropy tự chạy, không chạy trong phiên tự động. Script không in token, điểm này đúng.

## Khi được duyệt
Chạy bộ tải → kiểm tra `download-verification.json` → dựng env riêng theo code chính thức → viết runner → chạy thật trên vài ảnh người có thấy rõ bàn tay → báo VRAM, thời gian, chất lượng ngón.
