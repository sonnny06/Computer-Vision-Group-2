# Ghi chú thống nhất về chấm thử Captioning và VQA

**Ngày 24/11/2026.** Phiếu này dùng để đánh giá mô tả ảnh và câu trả lời VQA bằng tiếng Việt. Các ví dụ minh họa bên dưới là đầu ra giả định.

## Phiếu chấm thử

Với **Captioning**, chấm một mô tả cho mỗi ảnh. Với **VQA**, chấm một câu trả lời cho mỗi cặp ảnh và câu hỏi. Mỗi mẫu cần ghi căn cứ dùng để chấm, chẳng hạn metadata, nhãn hoặc báo cáo đi kèm ảnh.

| Tiêu chí | Cách ghi |
| --- | --- |
| Hoàn thành tác vụ | Với VQA, ghi câu trả lời đã đáp ứng ý được hỏi hay còn thiếu ý nào. Với Captioning, ghi ý chính nào được mô tả đúng, ý nào sai hoặc bị bỏ sót, dựa trên yêu cầu của prompt và nguồn tham chiếu. |
| Bám sát ảnh | Nếu một chi tiết đáng ngờ, trích nguyên văn và ghi căn cứ đối chiếu. Dùng **mâu thuẫn với căn cứ** khi có bằng chứng cho thấy chi tiết sai. Dùng **chưa đủ căn cứ kiểm chứng** khi tài liệu hiện có chưa cho phép kết luận. |
| Tiếng Việt | Ghi **rõ nghĩa** hoặc trích cụm từ gây khó hiểu. Chấm tiêu chí này độc lập với độ đúng nội dung. |

## Quy tắc chấm VQA

- Với `closed`, đối chiếu `answer` đã chuẩn hóa với nhãn tham chiếu `Có` hoặc `Không`. Kiểm tra quy ước đầu ra trên bản pipeline dùng để chạy. Với `open`, chấm theo ý nghĩa. Ví dụ, “tia X” và “X-quang” được tính tương đương.
- Ghi đúng hoặc sai cho phần trực tiếp trả lời câu hỏi. Nếu mô hình đưa thêm một nhận định cụ thể, trích nguyên văn vào ghi chú và đánh giá riêng mức độ bám ảnh. Ví dụ, với `Q_0153`, đáp án `Có` khớp phần xác nhận ảnh có bất thường. Tên bệnh do mô hình nêu thêm cần căn cứ riêng.
- Đánh dấu **cần xác minh** khi chưa đủ căn cứ. Ghi nhận **hallucination** khi đã đối chiếu và có căn cứ cho thấy chi tiết sai hoặc không được ảnh hỗ trợ.

## Ba ví dụ đã dùng để thử phiếu

Cùng một câu hỏi *“Ảnh này chụp bộ phận nào?”*. Giả sử nguồn tham chiếu chỉ xác nhận đây là **ảnh X-quang ngực**.

| Câu trả lời giả định | Hoàn thành tác vụ | Bám sát ảnh | Tiếng Việt |
| --- | --- | --- | --- |
| “Lồng ngực.” | Đạt | Phù hợp với tham chiếu | Rõ nghĩa |
| “Đây là ảnh X-quang.” | Chưa đạt vì thiếu tên bộ phận | Loại ảnh khớp với tham chiếu | Rõ nghĩa |
| “Ảnh chụp lồng ngực, có tổn thương ở phổi phải.” | Đạt ở ý được hỏi | Nhận định “có tổn thương ở phổi phải” chưa đủ căn cứ kiểm chứng | Rõ nghĩa |

Khi tổng hợp lỗi và hallucination, giữ riêng ghi nhận *chưa đủ căn cứ kiểm chứng*. Chỉ riêng việc một chi tiết vắng mặt trong mô tả tham chiếu chưa đủ để kết luận mô hình đã bịa chi tiết đó.

## Dữ liệu cần giữ khi chạy

- Ghi mã ảnh, `question_id` và `answer_type` nếu là VQA, cùng đáp án hoặc căn cứ chấm.
- Với VQA, lưu `raw_answer` và `answer`. Ghi kết quả đúng/sai và ghi chú về ngôn ngữ, định dạng hoặc chi tiết cần đối chiếu nếu có.
- Ghi thời gian phản hồi theo từng lượt và ghi trạng thái riêng nếu pipeline báo lỗi.

## Lượt chạy thử

Chạy 7 ca VQA và 3 ca Captioning trên `phoi_3.jpg`, `phoi_4.jpg` và `phoi_5.jpg`. Với Captioning, đối chiếu loại ảnh, vùng ngực và các cấu trúc nhìn thấy đã xác nhận trên từng ảnh.

| Mã ca | Ảnh | Câu hỏi tiếng Việt | `answer_type` | Đáp án kỳ vọng | Vai trò |
| --- | --- | --- | --- | --- | --- |
| `Q_0058` | `phoi_3.jpg` | Đây là ảnh chụp bằng phương thức nào? | `open` | X-quang | Tính điểm |
| `Q_0008` | `phoi_4.jpg` | Có bao nhiêu quả tim trong ảnh? | `open` | 1 | Tính điểm |
| `Q_0110` | `phoi_4.jpg` | Đây có phải là mặt cắt ngang không? | `closed` | Không | Tính điểm |
| `Q_0206` | `phoi_4.jpg` | Cơ quan/các cơ quan nào là một phần của hệ hô hấp trong hình ảnh này? | `open` | Phổi | Tính điểm |
| `Q_0171` | `phoi_5.jpg` | Đây là ảnh chụp bằng phương thức nào? | `open` | X-quang | Tính điểm |
| `PILOT_01` | `phoi_5.jpg` | Đây có phải là ảnh X-quang ngực không? | `closed` | Có | Tính điểm |
| `PILOT_02` | `phoi_3.jpg` | Trước khi chụp ảnh này, người bệnh đã ho hơn ba ngày chưa? | `closed` | Không đủ thông tin | Thăm dò |

Hai ca `PILOT_01` và `PILOT_02` do nhóm soạn. Báo **…/6 ca VQA chính đúng** và ghi riêng kết quả ca thăm dò bệnh sử. Đáp án kỳ vọng của ca này là hành vi mong muốn vì prompt `vqa.vi` đã chốt chưa quy định rõ tình huống ảnh rõ nhưng thiếu thông tin ngoài ảnh. Nếu `raw_answer` khẳng định Có hoặc Không về thời gian ho, ghi nhận khẳng định thiếu căn cứ. Nếu `raw_answer` thể hiện thiếu thông tin và `answer` chuyển thành Có hoặc Không, ghi lỗi chuẩn hóa.

## Bảng tổng hợp lượt đánh giá VQA

[CSV 109 câu đã gán answer_type](../../results/vqa_rebuilt_96ec26a_ready_with_answer_type.csv) là tập đã xác minh trước khi mở rộng lên 240 câu.

| Nhóm báo cáo | Kết quả |
| --- | --- |
| Toàn bộ tập VQA | …/109 câu đúng |
| `open` | …/70 câu đúng |
| `closed` | …/39 câu đúng |
| `closed` có đáp án tham chiếu `Có` | …/15 câu đúng |
| `closed` có đáp án tham chiếu `Không` | …/24 câu đúng |
| X-quang ngực PA | …/8 câu đúng trên 3 ảnh |

**Mốc tham chiếu cho `closed`.** Luôn trả lời `Không` đạt 24/39, tương đương **61,5%**. Đọc điểm chung cùng số câu đúng trên từng nhóm đáp án.

**Phạm vi X-quang ngực.** Lấy các dòng có `image_name` là `phoi_3.jpg`, `phoi_4.jpg` hoặc `phoi_5.jpg`. Nhãn `PA` cũng xuất hiện ở ảnh bàn tay. Tập con X-quang ngực đã nằm trong mẫu số 109 câu.

Bối cảnh kiến trúc, phân công và hiện trạng mã nguồn được ghi trong [README đồ án](../README.md).
