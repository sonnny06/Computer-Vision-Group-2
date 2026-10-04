# Subagent profiles — Issue #10: Grayscale

Issue: https://github.com/sonnny06/Computer-Vision-Group-2/issues/10

## Đánh giá độ phức tạp

Đánh giá kỹ thuật cho phạm vi một ảnh màu, không phải ước lượng thời gian được cam kết:

| Phần việc | Độ phức tạp | Lý do |
| --- | --- | --- |
| Thuật toán chuyển đổi | Thấp | Một phép chuyển đổi OpenCV, giao diện nhỏ |
| Notebook và tích hợp | Trung bình | Import source, đường dẫn, hiển thị màu, nội dung giải thích |
| QA và môi trường | Trung bình | Kernel sạch, hai thư mục khởi động, lưu/đọc lại PNG |
| Điều phối và review | Trung bình | File dùng chung, đồng bộ giao diện, bằng chứng hoàn thành |

Không cần reasoning high/xhigh cho phạm vi này. Các profile không cố định model,
để kế thừa lựa chọn của phiên/cấu hình điều phối. Low cho source; medium cho
notebook, QA và Lead. Nếu model đã chọn không hỗ trợ mức này, điều phối cần điều chỉnh.

## Bốn thành viên

| Profile | Vai trò | File sở hữu | Reasoning |
| --- | --- | --- | --- |
| `grayscale_lead.toml` | Tech Lead & Reviewer | Chỉ đọc, trả thiết kế và review | medium |
| `grayscale_image_dev.toml` | Lập trình viên xử lý ảnh | `lab_1/src/color_space.py` | low |
| `grayscale_notebook_dev.toml` | Lập trình viên notebook | `lab_1/notebook/03_grayscale.ipynb`, PNG output | medium |
| `grayscale_qa.toml` | QA & môi trường | `requirements.txt`; outputs notebook khi bàn giao quyền ghi | medium |

Giới hạn file là quy tắc phối hợp trong instructions, không phải ACL theo file.
Sandbox và quyền thực tế vẫn do môi trường/phiên cha quyết định.

## Giao diện chung

`bgr_to_grayscale(image_bgr)` nhận `numpy.ndarray` kiểu `uint8`, không rỗng,
shape `(H, W, 3)` theo thứ tự BGR; trả `uint8` shape `(H, W)`, không sửa input.
Sai kiểu/dtype: `TypeError`; shape sai/ảnh rỗng: `ValueError`.

Ảnh minh họa: `lab_1/data/input/02_portrait_astronaut.jpg`.
Output: `lab_1/data/output/02_portrait_astronaut_gray.png`.
Quy ước này là thiết kế đề xuất cho task, không phải chi tiết đã ghi trong issue.

## Điều phối khi được giao triển khai

1. Agent chính giao Lead chốt giao diện và review hiện trạng, nhận bàn giao rồi đóng thread Lead.
2. Chạy image dev, notebook dev và QA chuẩn bị dependencies song song (tối đa ba subagents).
3. Nhận source và notebook. Giao QA thực thi tích hợp sau khi notebook dev ngừng ghi file.
4. Nếu có lỗi, giao lại đúng người sở hữu file; chạy QA lại sau sửa.
5. Mở Lead để review cuối cùng; agent chính tổng hợp kết quả và phần chưa hoàn thành.

Phiên hiện tại có bốn slot tính cả agent chính: không chạy cả bốn subagents đồng thời.
Không cần tạo agent con lồng nhau. Agent chính chịu trách nhiệm Git và GitHub trong
phạm vi người dùng giao; profile không tự push, tạo PR, merge hoặc cập nhật issue.
Review của Lead là nội bộ, không thay thế approval GitHub.

## Định dạng và cách sử dụng

Các file TOML này là profile đã soạn, chưa cài vào cấu hình và chưa khởi chạy agent.
Theo tài liệu OpenAI hiện tại, custom agents của project nằm trong `.codex/agents/`,
mỗi file có `name`, `description`, `developer_instructions` và các tùy chọn phiên.
Khi cần cài, đưa bốn file TOML vào thư mục đó theo quyền ghi của môi trường.
Không ghi đè profile đã có cùng tên mà chưa kiểm tra.

Tài liệu định dạng: https://learn.chatgpt.com/docs/agent-configuration/subagents

Prompt điều phối sau khi profile được nhận diện:

> Triển khai issue #10 theo bốn profile grayscale_lead, grayscale_image_dev,
> grayscale_notebook_dev và grayscale_qa. Chốt giao diện trước, chạy tối đa ba
> subagents song song, bàn giao quyền ghi notebook trước QA, tổng hợp evidence
> kiểm tra và review cuối. Chỉ thực hiện Git/GitHub trong phạm vi đã được giao.

Trong một runtime không tự tải custom agents, agent chính có thể dùng nội dung
`developer_instructions` làm nội dung giao việc qua công cụ spawn, và chọn reasoning
theo profile nếu công cụ hỗ trợ. Không coi việc tạo file là bằng chứng agent đã chạy.

## Mẫu bàn giao

- Giai đoạn và trạng thái: ready / changes required / blocked.
- File sở hữu và thay đổi đã thực hiện.
- Giao diện hoặc quyết định cần thành viên khác dùng.
- Kiểm tra đã chạy, kết quả và bằng chứng; tách rõ phần chưa chạy.
- Lỗi còn lại, cách tái hiện và người cần xử lý.

Việc tạo profile không triển khai thuật toán, không chạy QA và không cập nhật issue.
