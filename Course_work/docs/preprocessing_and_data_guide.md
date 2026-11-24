# Báo Cáo & Hướng Dẫn: Module Image Preprocessing & Data Preparation
**Người phụ trách:** Trí  
**Vai trò:** Thành viên 4 - Image Preprocessing & Data Preparation  
**Môn học:** Computer Vision Coursework — Multimodal AI (Medical Captioning & VQA)  
**Tập dữ liệu sử dụng:** `v1olet/VietMed-VQA` (Hugging Face) & Bộ dữ liệu chấn thương chỉnh hình lâm sàng

---

## 1. Tổng quan các sản phẩm đã hoàn thành

| Sản phẩm | Vị trí file | Trạng thái |
| :--- | :--- | :--- |
| **Module xử lý ảnh** | [image_processing.py](file:///E:/Course_work/Course_work/src/preprocessing/image_processing.py) | ✅ Hoàn thành 100% |
| **Package Export** | [__init__.py](file:///E:/Course_work/Course_work/src/preprocessing/__init__.py) | ✅ Hoàn thành |
| **Bộ kiểm thử tự động** | [test_preprocessing.py](file:///E:/Course_work/Course_work/tests/test_preprocessing.py) | ✅ 14/14 tests Passed |
| **Bộ ảnh y tế chuẩn hóa** | [images/](file:///E:/Course_work/Course_work/data/evaluation/images/) & [data/](file:///E:/Course_work/data/) | ✅ 35 ảnh (7 bộ phận x 5 ảnh) |
| **Metadata mô tả lâm sàng** | [metadata.csv](file:///E:/Course_work/Course_work/data/evaluation/images/metadata.csv) | ✅ Đầy đủ 35 ca bệnh & nhãn |
| **Tập câu hỏi VQA tiếng Việt** | [questions.csv](file:///E:/Course_work/Course_work/data/evaluation/images/questions.csv) | ✅ 235 cặp Hỏi - Đáp tiếng Việt |

---

## 2. Danh mục 35 ảnh y tế theo 7 bộ phận cơ thể

Toàn bộ ảnh được lưu trữ đồng bộ tại cả hai thư mục:
- `Course_work/data/evaluation/images/`
- `data/`

Quy tắc đặt tên chuẩn hóa: `[tên bộ phận]_[số thứ tự].jpg`:

### 🫁 1. Phổi / Lồng ngực (Chest / Lung - 5 ảnh):
| Tên file | Phương thức chụp | Tình trạng / Khảo sát | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `phoi_1.jpg` | CT Cắt ngang (Axial) | Lát cắt ngang qua các buồng tim và nhu mô phổi | SLAKE / VietMed-VQA |
| `phoi_2.jpg` | CT Cắt ngang (Axial) | Lát cắt quai động mạch chủ và cuống mạch phổi | SLAKE / VietMed-VQA |
| `phoi_3.jpg` | X-quang ngực thẳng (PA) | Phế trường hai bên sáng rõ, tim bình thường | SLAKE / VietMed-VQA |
| `phoi_4.jpg` | X-quang ngực thẳng (PA) | Khảo sát khung xương sườn, vòm hoành và phế trường | SLAKE / VietMed-VQA |
| `phoi_5.jpg` | X-quang ngực thẳng (PA) | Đánh giá nhu mô phổi và bờ rốn phổi hai bên | SLAKE / VietMed-VQA |

---

### 🧠 2. Não / Sọ não (Brain / Head - 5 ảnh):
| Tên file | Phương thức chụp | Tình trạng / Khảo sát | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `nao_1.jpg` | Cộng hưởng từ (MRI) | Bán cầu đại não, hệ thống não thất và chất trắng | SLAKE / VietMed-VQA |
| `nao_2.jpg` | Cắt lớp vi tính (CT) | Lát cắt qua vùng hạch nền và rãnh cuộn não | SLAKE / VietMed-VQA |
| `nao_3.jpg` | Cắt lớp vi tính (CT) | Lát cắt qua thân não và thùy thái dương | SLAKE / VietMed-VQA |
| `nao_4.jpg` | Cắt lớp vi tính (CT) | Lát cắt vùng hố sau, tiểu não và não thất IV | SLAKE / VietMed-VQA |
| `nao_5.jpg` | Cộng hưởng từ (MRI) | Lát cắt vòm sọ, thùy trán và thùy đỉnh | SLAKE / VietMed-VQA |

---

### 🫄 3. Bụng / Ổ bụng (Abdomen - 5 ảnh):
| Tên file | Phương thức chụp | Tình trạng / Khảo sát | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `bung_1.jpg` | Cắt lớp vi tính (CT) | Lát cắt quan sát gan, lách, dạ dày và hai thận | SLAKE / VietMed-VQA |
| `bung_2.jpg` | CT cản quang (Contrast) | Nhu mô tụy, động mạch chủ bụng và bó mạch mạc treo | SLAKE / VietMed-VQA |
| `bung_3.jpg` | Cắt lớp vi tính (CT) | Khảo sát nhu mô thận và khoang sau phúc mạc | SLAKE / VietMed-VQA |
| `bung_4.jpg` | Cắt lớp vi tính (CT) | Kiểm tra các quai ruột non và mỡ mạc treo ruột | SLAKE / VietMed-VQA |
| `bung_5.jpg` | Cắt lớp vi tính (CT) | Khảo sát hạ sườn phải, túi mật và đoạn tá tràng | SLAKE / VietMed-VQA |

---

### 🦴 4. Khung chậu (Pelvic Cavity - 5 ảnh):
| Tên file | Phương thức chụp | Tình trạng / Khảo sát | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `khung_chau_1.jpg` | Cắt lớp vi tính (CT) | Khảo sát bàng quang, trực tràng và hốc chậu | SLAKE / VietMed-VQA |
| `khung_chau_2.jpg` | Cắt lớp vi tính (CT) | Đánh giá xương cánh chậu và xương cùng cụt | SLAKE / VietMed-VQA |
| `khung_chau_3.jpg` | Cắt lớp vi tính (CT) | Lát cắt khớp háng, chỏm xương đùi và ổ cối hai bên | SLAKE / VietMed-VQA |
| `khung_chau_4.jpg` | Cắt lớp vi tính (CT) | Khảo sát mức thấp hố ngồi trực tràng và cơ đáy chậu | SLAKE / VietMed-VQA |
| `khung_chau_5.jpg` | Cắt lớp vi tính (CT) | Đánh giá eo trên khung chậu và đại tràng sigma | SLAKE / VietMed-VQA |

---

### 🦒 5. Cổ / Vùng cổ (Neck - 5 ảnh):
| Tên file | Phương thức chụp | Tình trạng / Khảo sát | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `co_1.jpg` | Cắt lớp vi tính (CT) | Lát cắt qua hai thùy tuyến giáp, khí quản và mạch cảnh | SLAKE / VietMed-VQA |
| `co_2.jpg` | Cắt lớp vi tính (CT) | Lát cắt qua thanh quản, dây thanh âm và đốt sống cổ | SLAKE / VietMed-VQA |
| `co_3.jpg` | Cắt lớp vi tính (CT) | Khảo sát khẩu hầu và khoang mô mềm dưới hàm | SLAKE / VietMed-VQA |
| `co_4.jpg` | Cắt lớp vi tính (CT) | Đánh giá cơ ức đòn chũm và các chuỗi hạch cổ | SLAKE / VietMed-VQA |
| `co_5.jpg` | Cắt lớp vi tính (CT) | Lát cắt nền cổ giáp ranh đỉnh lồng ngực | SLAKE / VietMed-VQA |

---

### 💪 6. Xương tay (Arm / Hand - 5 ảnh):
| Tên file | Tư thế chụp | Bệnh lý / Tình trạng | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `tay_1.jpg` | AP | **Bình thường (KHÔNG GÃY)** | Roboflow Bone Dataset |
| `tay_2.jpg` | AP/Lat | Gãy hoàn toàn thân xương quay và xương trụ | Roboflow Bone Dataset |
| `tay_3.jpg` | AP/Lat | Gãy đầu dưới xương quay (Gãy Colles cổ tay) | Roboflow Bone Dataset |
| `tay_4.jpg` | PA | Gãy nứt xương đốt bàn và ngón tay | Stanford MURA Dataset |
| `tay_5.jpg` | AP | Gãy thân xương cánh tay có di lệch gập góc | Automated Fracture Dataset |

---

### 🦵 7. Xương chân (Leg / Foot - 5 ảnh):
| Tên file | Tư thế chụp | Bệnh lý / Tình trạng | Nguồn dữ liệu |
| :--- | :---: | :--- | :--- |
| `chan_1.jpg` | AP | **Bình thường (KHÔNG GÃY)** | Leg Fracture Dataset |
| `chan_2.jpg` | AP | Gãy cả hai xương cẳng chân (xương chày & mác) | Leg Fracture Dataset |
| `chan_3.jpg` | AP | Gãy lún mâm chày vùng khớp gối | Leg Fracture Dataset |
| `chan_4.jpg` | AP/Lat | Gãy mắt cá xương mác vùng cổ chân | Automated Fracture Dataset |
| `chan_5.jpg` | AP | Gãy nhiều mảnh thân xương đùi có di lệch | Automated Fracture Dataset |

*(Kèm 2 file test lỗi kỹ thuật: `ERR_001_corrupted.jpg` và `ERR_002_zero_byte.jpg` để chạy unit test).*

---

## 3. Tập dữ liệu câu hỏi VQA tiếng Việt (`questions.csv`)

File `questions.csv` bao gồm **235 câu hỏi - đáp y khoa tiếng Việt** chuẩn:
- **Cấu trúc trường:** `question_id`, `image_name`, `body_part`, `question_vi`, `answer_vi`, `question_en`, `answer_en`, `modality`, `source`.
- **Phân loại câu hỏi:**
  - Nhận diện bộ phận cơ thể và phương thức chụp (X-ray, CT, MRI).
  - Vị trí cơ quan và bất thường (trái, phải, trên, dưới, rốn phổi, thùy não,...).
  - Khẳng định tổn thương hoặc bệnh lý (Có / Không).
  - Chẩn đoán gãy xương và phân tích di lệch.

---

## 4. Cách chạy kiểm thử Unit Test

Mở terminal tại thư mục gốc hoặc `Course_work` và chạy:
```bash
python -m unittest tests/test_preprocessing.py
```
Kết quả mong đợi: `Ran 14 tests in 0.593s ... OK`.
