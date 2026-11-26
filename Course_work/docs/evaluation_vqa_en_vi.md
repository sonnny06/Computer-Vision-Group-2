# Đánh giá VQA Anh–Việt

## Phạm vi và kết quả

Bộ đánh giá gồm **199 cặp câu hỏi Anh–Việt trên 127 ảnh**. Mỗi cặp dùng cùng ảnh và cùng nội dung hỏi. Hai chế độ sử dụng prompt tương ứng với từng ngôn ngữ. Điểm nội dung được tính từ đầu ra Qwen đã lưu và phiếu chấm theo từng ID. [Dữ liệu đánh giá](../data/evaluation/questions.csv)

| Chế độ | Số câu đúng | Tỷ lệ đúng |
| --- | ---: | ---: |
| Tiếng Việt | 140/199 | 70,35% |
| Tiếng Anh | 144/199 | 72,36% |

Trong 199 cặp, **124** cặp đúng ở cả hai chế độ, **16** chỉ đúng ở chế độ tiếng Việt, **20** chỉ đúng ở chế độ tiếng Anh và **39** sai ở cả hai chế độ.

## Giới hạn diễn giải

Thành phần bộ câu hỏi phụ thuộc vào kết quả của chính lượt chạy được chấm. Vì vậy, tỷ lệ đúng chỉ mô tả hiệu năng trên 199 cặp này và có thể khác trên một tập kiểm thử độc lập. Hai chế độ cũng dùng chỉ dẫn prompt khác nhau, nên chênh lệch điểm phản ánh cả cấu hình của từng chế độ.

Hai phán quyết còn chờ rà liên quan đến phạm vi lời từ chối ở `Q_0131` và mức độ cụ thể của đáp án ở `VM240_RAD_1482`. Bảng điểm dùng phán quyết hiện có.
