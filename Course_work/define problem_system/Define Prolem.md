# Tên đề tài : Xây dựng hệ thống AI đa phương thức hỗ trợ phân tích ảnh y tế thông qua mô tả hình ảnh và trả lời câu hỏi bằng tiếng Việt. 


## 1.Vấn đề cần giải quyết : 

Trong lĩnh vực y tế, hình ảnh như X-quang, ảnh CT hoặc MRI chứa nhiều thông tin quan trọng. Tuy nhiên, việc phân tích hình ảnh thường đòi hỏi kiến thức chuyên môn và khả năng diễn giải kết quả.
Nhóm đặt ra vấn đề:
* Người dùng cung cấp một ảnh y tế nhưng không biết ảnh thể hiện những đặc điểm gì.
* Người dùng muốn biết nội dung chính của ảnh thông qua một đoạn mô tả bằng ngôn ngữ tự nhiên.
* Người dùng muốn đặt câu hỏi cụ thể về ảnh, chẳng hạn: "Có dấu hiệu bất thường nào trong ảnh không?" và nhận được câu trả lời dựa trên nội dung hình ảnh.

Vì vậy, nhóm muốn xây dựng một hệ thống AI có khả năng kết hợp thông tin hình ảnh và ngôn ngữ để hỗ trợ diễn giải ảnh y tế.
![kiến trúc dự kiến](https://image4.slideserve.com/9000741/vqa-architecture-l.jpg)


## 2. Input và output của hệ thống

| Thành phần | Input | Output |
| --- | --- | --- |
| Captioning | Ảnh y tế | Câu/đoạn mô tả ảnh |
| VQA | Ảnh y tế + câu hỏi | Câu trả lời bằng văn bản |


Input ảnh X-quang về phổi : 
![Input X-quang về phổi](https://images.openai.com/static-rsc-4/c9QRkhD3ijRGfqu4E37TSMT9rdDGQ5Vnxd7AQ1sWkMo_SY7mW9iSunNpwSWPTCsq2-zRl8KQWzExDxIK_0Kbgxalpi-J6zLi-hhW0nVbgO-fvlAwh7TMplf7D8dxzg9V0vWUweinzsKZaNcHjHv6zD1MGMz-g5NtZHwFJOM9N6kJP9w-557tqZAerjkWGu0L?purpose=fullsize)
**Chức năng 1 – Captioning**
KẾT QUẢ MÔ TẢ MINH HỌA
"Ảnh X-quang ngực cho thấy cấu trúc phổi và vùng tim, cần được đánh giá thêm bởi chuyên gia."

**Chức năng 2 – VQA**
CÂU HỎI
"Ảnh này có phải ảnh X-quang ngực không?"
CÂU TRẢ LỜI MINH HỌA
"Có, đây là ảnh X-quang ngực."



## 3. Đặc tả Dữ liệu và Kiến trúc (Data & Architecture Specification)
* **Dữ liệu đầu vào (Input):**
  * **Visual (Thị giác):** Hình ảnh y khoa (ảnh 2D/3D như X-quang, MRI) hoặc các tập dữ liệu tham khảo phổ biến để pre-train.
  * **Text (Văn bản):** Các câu hỏi truy vấn (dành cho task VQA) dưới dạng văn bản (tiếng Anh/Việt).
* **Kết quả đầu ra (Output):** 
  * Văn bản tự nhiên mô tả các đặc trưng lâm sàng trên ảnh (Captioning).
  * Câu trả lời ngắn gọn hoặc giải thích chi tiết cho câu hỏi đầu vào (VQA).
* **Định hướng Kiến trúc Mô hình:**
  * **Visual Encoder:** Sử dụng CNN (như ResNet) hoặc ViT (Vision Transformer) để trích xuất đặc trưng hình ảnh.
  * **Text Encoder/Decoder:** Sử dụng kiến trúc Transformer để xử lý câu hỏi đầu vào và sinh văn bản đầu ra.
* **Tập dữ liệu dự kiến:** MS COCO Captions, VQA v2 (để thử nghiệm/pre-train) kết hợp với các tập dữ liệu ảnh y khoa đặc thù có gán nhãn văn bản.

## 4. Tiêu chí Đánh giá (Success Metrics)
* **Đối với bài toán Image Captioning:** Sử dụng các thang đo so khớp văn bản tự nhiên như BLEU-1/2/3/4, METEOR, ROUGE-L, và CIDEr để đánh giá mức độ tương đồng giữa câu sinh ra và báo cáo y khoa của chuyên gia.
* **Đối với bài toán VQA:** Sử dụng độ chính xác (Accuracy) theo từng loại câu hỏi (ví dụ: câu hỏi Yes/No, câu hỏi mở, câu hỏi định vị).
* **Chỉ số Y khoa:** Độ nhạy (Sensitivity) và Độ đặc hiệu (Specificity) trong việc phát hiện đúng và mô tả đúng vùng bệnh lý, tránh hiện tượng sinh văn bản "ảo giác" (hallucination).

## 5. Ràng buộc & Thách thức (Constraints & Challenges)
* **Độ chính xác và Tính an toàn:** Ứng dụng y khoa đòi hỏi độ chính xác cực kỳ cao; một chẩn đoán hoặc mô tả sai lệch từ mô hình (AI Hallucination) có thể dẫn đến hậu quả nghiêm trọng.
* **Đặc thù Dữ liệu:** Dữ liệu ảnh y khoa có đi kèm báo cáo/câu hỏi cặp (Image-Text pairs) thường rất khan hiếm, khó thu thập do vấn đề bảo mật thông tin bệnh nhân và tốn kém chi phí để bác sĩ chuyên khoa gán nhãn.
* **Chi phí tính toán:** Các mô hình kết hợp ViT và Transformer yêu cầu tài nguyên phần cứng lớn (GPU VRAM cao) để huấn luyện và tinh chỉnh (fine-tuning).

## 6.Business Problem
Trong lĩnh vực y tế, việc đọc và diễn giải hình ảnh y khoa đòi hỏi kiến thức chuyên môn. Người dùng không chuyên có thể gặp khó khăn khi muốn hiểu nội dung cơ bản của một hình ảnh y tế hoặc tìm kiếm thông tin cụ thể từ hình ảnh đó.
Nhóm xác định ba vấn đề chính:
| ID | Business Problem | Tác động |
|---|---|---|
| BP01 | Khó hiểu nội dung hình ảnh y tế | Người dùng khó tiếp cận thông tin từ ảnh |
| BP02 | Khó tìm kiếm thông tin cụ thể trong ảnh | Phải tự quan sát và diễn giải |
| BP03 | Thiếu công cụ tương tác bằng ngôn ngữ tự nhiên | Người dùng khó đặt câu hỏi trực tiếp về ảnh |

## 7.Business Features
6 nhóm tính năng.
### F01 – Upload Medical Image
Cho phép người dùng tải ảnh y tế lên hệ thống.
- Hỗ trợ định dạng JPG, JPEG, PNG.
- Kiểm tra định dạng và dung lượng.
- Hiển thị ảnh đã tải lên.
![upload image](https://images.openai.com/static-rsc-4/4KAF4lpkp7fRrILJEkGFGyAQXuT9AKNAoY3yjww0fFQm5g90xWsXvrStPns5mQvP9cjbW8C1GpQxj_IgewnN32dCNXptVslk71wkkQbOkX2uC4vNJ2a3xHUUJYPq-aJKGmOOpMBWgQlO2X4KHdNSFdJnnI5akMH5SCMHzsDXYKM4LTSGJqUprmR28zslwMPI?purpose=fullsize)
### F02 – Image Captioning
Tự động tạo mô tả bằng ngôn ngữ tự nhiên từ ảnh y tế.
- Phân tích đặc trưng hình ảnh.
- Sinh đoạn mô tả.
- Hiển thị kết quả cho người dùng.
![Fo2](https://images.openai.com/static-rsc-4/LWzihAjZH6FgEPHkAYD73P1eFUOPdB9uytxsQz8K16XPKIqwD763Jg67gPLW_g79FHwzRIJL-w_7LT9CSdx8z42R5iPCK4zCvAz_JzEDuKUjB0QHQHnILNBD_cD-JKcZ-tR_NnPYJG9N-iGDbEJfjAIZ10hUQ8v4I9ukUxTBzWxTWVzbR2xLqLUEQEg5xf2l?purpose=fullsize)
### F03 – Medical Visual Question Answering
Cho phép người dùng đặt câu hỏi về ảnh y tế.
- Nhập câu hỏi bằng văn bản.
- Phân tích ảnh kết hợp câu hỏi.
- Trả lời dựa trên nội dung hình ảnh.
![fo3](https://images.openai.com/static-rsc-4/eDXNkSVJ0wrGGpTO36iv4o92lF0grdFAdTavvTUz7-Pks51dHGlhli0KuD2-k4FddLgzWV2Qxa9NNgw0EdW4Bfh2YjMEgemOpa830ShG8C9Tjxt8hWJpYjs4-v8YjzFMjeBSTEg-w1dElFl32wPync76DsC4o0mcG4QXYq5CEa76fLKM21Sg1yUi7F8J5spv?purpose=fullsize)

### F04 – Result Visualization
Hiển thị kết quả một cách trực quan.
- Hiển thị ảnh đầu vào.
- Hiển thị mô tả ảnh.
- Hiển thị câu hỏi và câu trả lời.

### F05 – Question History
Lưu lại lịch sử các câu hỏi và câu trả lời trong phiên làm việc.
- Xem lại câu hỏi trước đó.
- Tiếp tục hỏi về ảnh hiện tại.
- Xóa lịch sử phiên làm việc.

### F06 – Model Evaluation
Phục vụ nhóm nghiên cứu đánh giá chất lượng mô hình.
- Xem lại câu hỏi trước đó.
- Tiếp tục hỏi về ảnh hiện tại.
- Xóa lịch sử phiên làm việc
