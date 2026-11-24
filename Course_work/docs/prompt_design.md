# Thiết Kế Prompt & Kiểm Soát Hành Vi Mô Hình (Qwen3-VL)

- **Phụ trách:** Thành viên 3 — Tôn (Prompt Engineering & Model Behavior)
- **Tác vụ chính:** Medical Image Captioning & Medical Visual Question Answering (VQA)
- **Các file liên quan:**
  - `Course_work/configs/prompt_config.yaml` — Lưu trữ template prompt đa ngôn ngữ và tham số cấu hình.
  - `Course_work/src/utils/prompt.py` — Module đọc cấu hình, kiểm tra đầu vào và sinh prompt chuẩn hóa.
  - `Course_work/docs/prompt_design.md` — Tài liệu giải thích thiết kế và báo cáo thử nghiệm prompt.

---

## 1. Cách Qwen3-VL tiếp nhận ảnh và câu hỏi

Qwen3-VL là mô hình ngôn ngữ – thị giác đa phương thức (Vision-Language Model) xử lý đầu vào theo cấu trúc hội thoại (`messages`), trong đó phần nội dung (`content`) của người dùng kết hợp tuần tự đối tượng ảnh và văn bản hướng dẫn:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path_or_pil},
            {"type": "text", "text": prompt_text},
        ],
    }
]