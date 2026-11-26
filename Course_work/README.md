# Medical-VQA — Hệ thống AI Đa phương thức Phân tích Ảnh Y tế (Topic 4)

Hệ thống hỗ trợ phân tích hình ảnh y tế thông qua **Image Captioning**, **Medical Visual Question Answering (VQA)** tiếng Việt và **Khoanh vùng dị thường (ROI)**, tích hợp mô hình `Qwen/Qwen3-VL-2B-Instruct`.

## 1. Cách chạy nhanh giao diện Web (Khuyên dùng)
```powershell
python run_web.py
```
Trình duyệt sẽ tự mở tại `http://127.0.0.1:8000` với đầy đủ 6 chức năng **F01 – F06**, bộ VQA gồm 199 câu hỏi tiếng Việt trên 127 ảnh và bộ Captioning gồm 35 ảnh tham chiếu, khoanh vùng dị thường OpenCV và xuất phiếu báo cáo PDF.

Bộ đánh giá VQA chính gồm [199 cặp câu hỏi Anh–Việt trên 127 ảnh](data/evaluation/questions.csv). CLI hiện chạy phía tiếng Việt theo mặc định. Xem [kết quả đánh giá Anh–Việt](docs/evaluation_vqa_en_vi.md).

## 2. Cách chạy giao diện Streamlit
```powershell
python -m streamlit run app/app.py
```

## 3. Chạy kiểm thử tự động
```powershell
python tests/test_pipeline.py
python tests/test_preprocessing.py
```

## 4. Phân công nhóm 6 thành viên
| Thành viên | Vai trò | Module chính |
|---|---|---|
| **Sơn** | Application Development | `web/index.html`, `web/server.py`, `app/app.py`, `app/components.py`, `src/utils/history.py` |
| **Thái** | Pre-trained Model Integration | `src/models/pretrained_vlm.py`, `configs/model_config.yaml` |
| **Tôn** | Prompt Engineering | `src/utils/prompt.py`, `configs/prompt_config.yaml` |
| **Trí** | Image Preprocessing & Data | `src/preprocessing/image_processing.py`, `src/data/preprocessing.py`, `data/evaluation/` |
| **Phúc** | Model Evaluation & Testing | `src/evaluation/evaluate_captioning.py`, `evaluate_vqa.py`, `results/` |
| **Bão** | Backend & System Integration | `src/engine/pipeline.py`, `schemas.py`, `errors.py`, `src/inference/*.py` |
