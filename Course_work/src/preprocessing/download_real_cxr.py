"""
Script tự động thu thập (cào) bộ ảnh X-quang ngực thực tế từ các nguồn mở y khoa uy tín
(Bộ dữ liệu IEEE COVID/ChestX-ray dataset & NIH ChestX-ray14)
Phụ trách: Trí (Image Preprocessing & Data Preparation)
"""

import os
import csv
import urllib.request
from pathlib import Path
from PIL import Image

# Thư mục đích
BASE_DIR = Path(r"E:\Course_work\Course_work")
IMAGE_DIR = BASE_DIR / "data" / "evaluation" / "images"
METADATA_PATH_1 = BASE_DIR / "data" / "evaluation" / "metadata.csv"
METADATA_PATH_2 = IMAGE_DIR / "metadata.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Danh sách 10 ca ảnh X-quang thực tế từ bệnh nhân thật
DOWNLOAD_LIST = [
    {
        "image_id": "REAL_001",
        "save_name": "REAL_CXR_001_normal_pa.png",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/16745_2_1.PNG",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Normal",
        "clinical_notes_en": "Patient 1 - CXR: Normal. Clear lung fields, no focal consolidation, no pleural effusion.",
        "finding_description": "Hai phế trường sáng rõ, không thấy đám mờ thâm nhiễm hay đông đặc khu trú. Cung tim và trung thất trong giới hạn bình thường. Góc sườn hoành hai bên sáng nhọn.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng tư thế PA bình thường, không ghi nhận tổn thương bệnh lý ở phổi hay màng phổi.",
        "source": "IEEE Open Medical Dataset / Case Study"
    },
    {
        "image_id": "REAL_002",
        "save_name": "REAL_CXR_002_normal_patient2.png",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/16745_3_1.png",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Normal",
        "clinical_notes_en": "Patient 2 - CXR: Normal chest radiograph with clear diaphragmatic contours and normal cardiac size.",
        "finding_description": "Cấu trúc lồng ngực cân đối, nhu mô phổi hai bên bình thường, vòm hoành trơn láng, không có ứ dịch màng phổi.",
        "ground_truth_caption_vi": "Hình ảnh X-quang ngực thẳng bình thường, bóng tim và hai trường phổi không có bất thường.",
        "source": "IEEE Open Medical Dataset / Case Study"
    },
    {
        "image_id": "REAL_003",
        "save_name": "REAL_CXR_003_bacterial_pneumonia_strep.jpg",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/pneumonia-7.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "AP",
        "finding_category": "Bacterial Pneumonia (Streptococcus)",
        "clinical_notes_en": "Extensive consolidation and air bronchograms with loss of the right hemidiaphragm in keeping with right lower lobe pneumonia.",
        "finding_description": "Đông đặc phế nang diện rộng kèm hình ảnh phế quản hơi (air bronchograms), xóa bờ vòm hoành phải, đặc trưng của viêm phổi thùy dưới phổi phải do phế cầu khuẩn (Streptococcus pneumoniae).",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy đám mờ đông đặc thùy dưới phổi phải có hình phế quản hơi, phù hợp viêm phổi thùy do vi khuẩn.",
        "source": "Radiopaedia / IEEE Medical Dataset"
    },
    {
        "image_id": "REAL_004",
        "save_name": "REAL_CXR_004_klebsiella_pneumonia.jpg",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/klebsiella-pneumonia-1.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Bacterial Pneumonia (Klebsiella)",
        "clinical_notes_en": "Tachypneic and febrile. Extensive right upper lobe consolidation.",
        "finding_description": "Đám mờ đông đặc diện rộng tại thùy trên phổi phải, rãnh liên thùy bị đẩy lồi xuống dưới (dấu hiệu rãnh liên thùy phồng kinh điển của viêm phổi Klebsiella).",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thâm nhiễm đông đặc thùy trên phổi phải điển hình của nhiễm khuẩn Klebsiella pneumoniae.",
        "source": "Radiopaedia / IEEE Medical Dataset"
    },
    {
        "image_id": "REAL_005",
        "save_name": "REAL_CXR_005_covid19_vietnam_patient.jpg",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Viral Pneumonia (COVID-19)",
        "clinical_notes_en": "65-year-old man in Vietnam admitted with fever and cough. Chest radiograph shows bilateral patchy infiltrates and ground-glass opacities in both lungs.",
        "finding_description": "Bệnh nhân nam 65 tuổi tại Việt Nam (ca nhiễm COVID-19 đầu năm 2020), hình ảnh các đám mờ kính mờ dạng vết vá rải rác ngoại vi và hai đáy phổi.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy tổn thương viêm phổi virus dạng kính mờ rải rác hai đáy phổi ở bệnh nhân COVID-19.",
        "source": "AuntMinnie / Vietnam COVID-19 Case Report"
    },
    {
        "image_id": "REAL_006",
        "save_name": "REAL_CXR_006_covid19_nejm.jpg",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/nejmc2001573_f1a.jpeg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Viral Pneumonia (COVID-19)",
        "clinical_notes_en": "New England Journal of Medicine case. Diffuse infiltrates in the bilateral lower lungs.",
        "finding_description": "Tổn thương thâm nhiễm lan tỏa hai bên phế trường, tập trung nhiều ở thùy dưới hai phổi.",
        "ground_truth_caption_vi": "Hình ảnh X-quang ngực xuất hiện thâm nhiễm kẽ lan tỏa hai bên phế trường phù hợp viêm phổi cấp do virus.",
        "source": "New England Journal of Medicine (NEJM)"
    },
    {
        "image_id": "REAL_007",
        "save_name": "REAL_CXR_007_pneumocystis_fungal.png",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/pneumocystis-pneumonia-2-PA.png",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Fungal Pneumonia (Pneumocystis jirovecii)",
        "clinical_notes_en": "Hazy, predominantly perihilar mid and upper zone airspace opacification.",
        "finding_description": "Hình mờ mờ mịt hình cánh bướm quanh rốn phổi hai bên (perihilar airspace opacification), đặc trưng của viêm phổi do nấm Pneumocystis jirovecii (PCP).",
        "ground_truth_caption_vi": "Ảnh X-quang ngực có đám mờ dạng kính mờ quanh rốn phổi hai bên, đặc trưng viêm phổi do nấm PCP.",
        "source": "Radiopaedia / IEEE Medical Dataset"
    },
    {
        "image_id": "REAL_008",
        "save_name": "REAL_CXR_008_tuberculosis_miliary.jpg",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000003-1.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Tuberculosis (Lao phổi)",
        "clinical_notes_en": "Hazy micronodular pattern symmetrically involving both lungs (Miliary tuberculosis).",
        "finding_description": "Vô số nốt mờ nhỏ li ti (kích thước hạt kê 1-2mm) phân bố đồng đều đối xứng khắp hai phế trường từ đỉnh đến đáy, biểu hiện điển hình của Lao kê (Miliary Tuberculosis).",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng cho thấy hình ảnh các nốt vi thể rải rác hai phế trường điển hình của bệnh lao kê.",
        "source": "Eurorad / IEEE Medical Dataset"
    },
    {
        "image_id": "REAL_009",
        "save_name": "REAL_CXR_009_severe_ards.png",
        "url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/ARDSSevere.png",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Acute Respiratory Distress Syndrome (ARDS)",
        "clinical_notes_en": "Severe ARDS. Patient intubated with an orogastric (OG) tube in place. Bilateral widespread alveolar opacities.",
        "finding_description": "Hội chứng suy hô hấp cấp tiến triển (ARDS) mức độ nặng, thâm nhiễm phế nang lan tỏa trắng xóa hai phổi (phổi trắng), bệnh nhân đang được đặt ống nội khí quản thở máy.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy tổn thương đông đặc phế nang lan tỏa hai bên trong hội chứng suy hô hấp cấp ARDS nặng.",
        "source": "IEEE Open Medical Dataset"
    },
    {
        "image_id": "REAL_010",
        "save_name": "REAL_CXR_010_cardiomegaly_nih.png",
        "url": "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/00000001_000.png",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Cardiomegaly",
        "clinical_notes_en": "NIH ChestX-ray14 benchmark sample. Cardiomegaly with enlarged cardiac silhouette (CTR > 0.55).",
        "finding_description": "Bóng tim to rõ rệt, chỉ số tim - lồng ngực (Cardiothoracic Ratio) tăng đáng kể (> 0.55), bờ tim trái mở rộng sát thành ngực.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng cho thấy bóng tim to rõ rệt (Cardiomegaly), hai phế trường không thấy thâm nhiễm.",
        "source": "NIH Clinical Center ChestX-ray14"
    }
]

print("=== BẮT ĐẦU CÀO & TẢI ẢNH X-QUANG THỰC TẾ ===")
success_records = []

for item in DOWNLOAD_LIST:
    dest_path = IMAGE_DIR / item["save_name"]
    url = item["url"]
    print(f"Đang tải {item['image_id']} - {item['finding_category']}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        
        with open(dest_path, "wb") as f:
            f.write(data)
            
        # Kiểm tra tính toàn vẹn và lấy kích thước bằng PIL
        with Image.open(dest_path) as img:
            res_str = f"{img.width}x{img.height}"
            
        item["filename"] = item["save_name"]
        item["resolution"] = res_str
        success_records.append(item)
        print(f" -> Thành công: {item['save_name']} ({res_str}, {len(data):,} bytes)")
        
    except Exception as exc:
        print(f" -> Lỗi tải {url}: {exc}")

# Tạo 2 ca kiểm thử lỗi (corner cases) phục vụ bài test hệ thống
err_corrupt_path = IMAGE_DIR / "ERR_001_corrupted.jpg"
with open(err_corrupt_path, "wb") as f:
    f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_CORRUPT_BYTES_XYZ123456789")

err_zero_path = IMAGE_DIR / "ERR_002_zero_byte.jpg"
with open(err_zero_path, "wb") as f:
    pass

print(f"Đã tạo các ca kiểm thử lỗi: {err_corrupt_path.name}, {err_zero_path.name}")

# Ghi lại file metadata.csv chuẩn
fieldnames = [
    "image_id", "filename", "image_type", "view_position",
    "finding_category", "clinical_notes_en", "finding_description",
    "ground_truth_caption_vi", "source", "resolution"
]

for meta_path in [METADATA_PATH_1, METADATA_PATH_2]:
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in success_records:
            row = {k: rec.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"Đã lưu metadata mới tại: {meta_path}")

print(f"\n=== HOÀN TẤT: ĐÃ TẢI THÀNH CÔNG {len(success_records)}/10 ẢNH THẬT ===")
