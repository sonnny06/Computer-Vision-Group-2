# Tổng quan kiến trúc hệ thống 

**Hệ thống dự kiến**

![Flow dự kiến](<image\flow tổng quát.jpg>)

![pipeline VQA](<image\vqa pipeline.jpg>)


## Tiền xử lí :
Resize và chuẩn hóa kích thước: Ảnh đầu vào được đưa về kích thước chuẩn tương thích với DenseNet-121 (224×224 pixels), sử dụng nội suy bilinear để bảo toàn thông tin chi tiết tối đa.

## Bộ mã hóa ảnh - DenseNet-121
DenseNet-121 nhận ảnh y khoa đã được tiền xử lý (kích thước chuẩn, 1 kênh grayscale, chuẩn hóa theo phân phối X-quang) và tạo ra feature map không gian kích thước 7×7×1024. Feature map này giữ lại thông tin vị trí không gian — quan trọng cho các câu hỏi về vị trí tổn thương — thay vì đơn giản hóa thành vector toàn cục

## Bộ mã hóa văn bản - PhoBERT
PhoBERT tokenize câu hỏi tiếng Việt bằng BPE tokenizer và tạo ra chuỗi L biểu diễn ngữ cảnh 768 chiều. Trong khi vector [CLS] đại diện cho ngữ nghĩa cấp câu, toàn bộ chuỗi token được sử dụng để tính co-attention, đảm bảo không mất thông tin ngữ nghĩa cục bộ của từng từ khóa quan trọng.

## Co-Attention Fusion
Lớp co-attention học tương tác có điều kiện hai chiều: câu hỏi quyết định vùng nào của ảnh cần chú ý, và ngược lại. Biểu diễn hợp nhất đầu ra có chiều tương thích với cả hai nhánh giải mã, đảm bảo thông tin ảnh-văn bản được tích hợp đầy đủ

## Bộ giải mã
LSTM Decoder: Bộ giải mã LSTM nhiều lớp sinh câu trả lời token-bytoken trong khi duy trì trạng thái ẩn mang ngữ cảnh của các token đã sinh. Ưu điểm là đơn giản, ổn định khi huấn luyện trên tập dữ liệu nhỏ; nhược điểm là phụ thuộc xa bị suy giảm theo chiều dài chuỗi.

## Tầng 1 – User Interface : 
Đây là nơi người dùng tải ảnh, nhập câu hỏi và xem kết quả.
Giao diện cần có hai chức năng:
- Generate Caption: tạo mô tả ảnh.
- Ask a Question: đặt câu hỏi về ảnh.
Ví dụ:
Người dùng tải lên một ảnh X-quang và nhập câu hỏi:
"What can you observe in this image?"
->  Hệ thống tiếp nhận ảnh và câu hỏi rồi chuyển xuống tầng xử lý.
## Tầng 2 – Input Processing
Đây là tầng tiền xử lý dữ liệu trước khi đưa vào AI.
| Module | Nhiệm vụ |
|---|---|
| Image Preprocessing | Resize, chuẩn hóa ảnh |
| Text Processing | Tokenize, chuyển câu hỏi thành dữ liệu số |

Tại sao phải có tầng này?
Vì mô hình AI không trực tiếp hiểu một file ảnh JPG hoặc một câu văn như con người. Dữ liệu phải được chuyển về định dạng mà mô hình có thể tiếp nhận.
## Tầng 3 - AI Model : 
### Nhánh A – Medical Image Captioning
Mô hình đề xuất: BLIP
Quy trình:
- 1. Nhận ảnh y tế.
- 2. Image Encoder trích xuất đặc trưng ảnh.
- 3. Text Decoder sinh câu mô tả.
- 4. Trả về đoạn văn bản.
Ví dụ đầu ra: "The image shows a chest radiograph with visible lung fields."
### Nhánh B – Medical VQA : 
Mô hình đề xuất: DenseNet-121 + PhoBERT + Decoder
Quy trình:
- 1. DenseNet-121 nhận ảnh và trích xuất đặc trưng hình ảnh.
- 2. PhoBERT nhận câu hỏi và trích xuất đặc trưng ngôn ngữ.
- 3. Feature Fusion kết hợp hai loại đặc trưng.
- 4. LSTM hoặc Transformer Decoder tạo câu trả lời.

``Ví dụ đầu ra: "Yes, this is a chest X-ray image." ``
### Tầng 4 – Output Processing
Tầng cuối cùng chịu trách nhiệm:
- Nhận kết quả từ hai mô hình.
- Chuẩn hóa nội dung trả về.
- Hiển thị mô tả và câu trả lời.
- Thông báo nếu ảnh hoặc câu hỏi không hợp lệ.
![output processing](https://images.openai.com/static-rsc-4/5TYE3wbNLJ9gPxHexkJ7Hm-YpbMcWo3cHk7htbhY0M9qUaDfAqezFUiSJpdGJQQlEfYQt5tOjpEhkN9HZvmEt8dDpMOnYj7IEk2PXBT9GpnNfnEN60X1KqjQWgcltzQRN7jz6kljjhCpC2vqkxRzUV3syYRtEOvFdlJaiuUogcXRMEcGrJ89xE2XDzEmVdWN?purpose=fullsize)

## Cấu trúc thư mục dự kiến :
```text
Medical-VQA-Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── models/
│   ├── captioning/
│   │   └── blip_model.py
│   │
│   └── vqa/
│       ├── image_encoder.py
│       ├── text_encoder.py
│       ├── fusion.py
│       └── vqa_model.py
│
├── src/
│   ├── preprocessing/
│   │   ├── image_processing.py
│   │   └── text_processing.py
│   │
│   ├── training/
│   │   ├── train_captioning.py
│   │   └── train_vqa.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_captioning.py **bỏ qua**
│   │   └── evaluate_vqa.py   ** bỏ qua**
│   │
│   └── inference/
│       ├── captioning_inference.py
│       └── vqa_inference.py
│
├── app/
│   └── app.py
│
├── checkpoints/
│   ├── captioning/
│   └── vqa/
│
├── configs/
├── requirements.txt
└── README.md
```