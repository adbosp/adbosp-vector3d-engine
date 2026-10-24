# Kênh review Claude ↔ Codex / Claude ↔ Codex review channel

## Tiếng Việt
Theo yêu cầu của Ropy: mỗi khi gặp vấn đề, và trước khi kết thúc một phiên làm việc (dù xong hay chưa), Claude gửi việc sang Codex review. Vòng lặp tiếp tục cho đến khi Codex trả lời ĐẠT.
- `TO_CODEX.md`: Claude viết. Mỗi yêu cầu có một `request_id` mới và gồm: commit hoặc diff, các file đã đổi, lệnh tái lập chính xác, số liệu đo được, đường dẫn kết quả, tiêu chí đạt rõ ràng, và những gì còn bị chặn.
- `FROM_CODEX.md`: Codex viết. Mỗi phản hồi ghi ngày, commit, `request_id`, kết luận ĐẠT / CHƯA ĐẠT / BLOCKED, và việc cần sửa.
- Trước mỗi khối việc mới, Claude đọc `FROM_CODEX.md` và sửa hết các mục CHƯA ĐẠT trước.
- Kết luận của Codex là review kỹ thuật. Nó **không** thay cho sự đồng ý của Ropy với việc tốn tiền, không hoàn tác được, hoặc liên quan đến license và điều khoản. Codex cũng không vượt được cổng duyệt của repo gated.
- Không ghi token, mật khẩu hay dữ liệu cá nhân vào các file này.

## English
Per Ropy: whenever there is a problem, and before ending any work session (finished or not), Claude hands the work to Codex for review, and loops until Codex says ĐẠT (pass).
- `TO_CODEX.md` is written by Claude. Each request has a fresh `request_id` and includes the commit or diff, the changed files, exact reproduce commands, the measured numbers, artifact paths, explicit pass criteria, and what is still blocked.
- `FROM_CODEX.md` is written by Codex: date, commit, `request_id`, and a verdict of ĐẠT / CHƯA ĐẠT / BLOCKED with the fixes needed.
- Before each new work block, Claude reads `FROM_CODEX.md` and fixes every CHƯA ĐẠT item first.
- A Codex verdict is a technical review. It is **not** Ropy's consent to anything paid, irreversible, or licence- or terms-related. Codex also cannot bypass a gated repository's approval.
- Never put tokens, passwords or personal data in these files.
