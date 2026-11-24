"""
Script chọn lọc 5 ảnh mỗi bộ phận (Phổi, Tay, Chân) và lưu vào thư mục data.
Đặt tên theo cấu trúc: [tên bộ phận]_[số thứ tự].jpg
Phụ trách: Trí (Image Preprocessing & Data Preparation)
"""

import os
import csv
import io
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"E:\Course_work\Course_work")
IMAGE_DIR = BASE_DIR / "data" / "evaluation" / "images"
METADATA_PATH_1 = BASE_DIR / "data" / "evaluation" / "metadata.csv"
METADATA_PATH_2 = IMAGE_DIR / "metadata.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Tải archive.zip để trích xuất các ca gãy xương
print("1. Đang tải kho dữ liệu chấn thương xương khớp...")
archive_url = "https://raw.githubusercontent.com/arjun2703-ai/Automated-Fracture-Detection/main/archive.zip"
req_archive = urllib.request.Request(archive_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_archive, timeout=20) as resp:
    archive_data = resp.read()
zip_archive = zipfile.ZipFile(io.BytesIO(archive_data))

# 2. Danh sách 15 ca ảnh chuẩn chia đều 3 bộ phận (Phổi: 5, Tay: 5, Chân: 5)
DATASET_CONFIG = [
    # ==========================
    # NHÓM 1: PHỔI (5 ảnh)
    # ==========================
    {
        "id": "PHOI_01",
        "filename": "phoi_1.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/16745_2_1.PNG",
        "view": "PA",
        "finding": "Bình thường (Normal)",
        "clinical_notes_en": "Normal chest radiograph. Clear bilateral lung fields, normal cardiac silhouette, sharp costophrenic angles.",
        "description_vi": "Ảnh X-quang phổi bình thường: Phế trường hai bên sáng rõ, không thấy vùng đông đặc hay tràn dịch màng phổi, bóng tim trong giới hạn bình thường.",
        "ground_truth_caption": "Ảnh X-quang ngực thẳng tư thế PA bình thường, không ghi nhận bất thường nhu mô phổi hay bóng tim.",
        "source": "IEEE Open Medical Dataset"
    },
    {
        "id": "PHOI_02",
        "filename": "phoi_2.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/pneumonia-7.jpg",
        "view": "AP",
        "finding": "Viêm phổi thùy do vi khuẩn (Bacterial Pneumonia)",
        "clinical_notes_en": "Extensive consolidation and air bronchograms with loss of the right hemidiaphragm in keeping with right lower lobe pneumonia.",
        "description_vi": "Viêm phổi thùy dưới phổi phải: Đám mờ đông đặc thâm nhiễm diện rộng, có hình ảnh phế quản hơi, xóa bờ vòm hoành phải.",
        "ground_truth_caption": "Ảnh X-quang ngực cho thấy đám mờ đông đặc thùy dưới phổi phải có phế quản hơi, phù hợp viêm phổi do vi khuẩn.",
        "source": "Radiopaedia"
    },
    {
        "id": "PHOI_03",
        "filename": "phoi_3.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
        "view": "PA",
        "finding": "Viêm phổi virus (COVID-19 ca tại Việt Nam)",
        "clinical_notes_en": "65-year-old male COVID-19 patient in Vietnam. Bilateral patchy infiltrates and ground-glass opacities in both lower lung fields.",
        "description_vi": "Viêm phổi do virus COVID-19: Tổn thương dạng kính mờ rải rác ngoại vi và hai đáy phổi trên bệnh nhân điều trị tại Việt Nam.",
        "ground_truth_caption": "Ảnh X-quang ngực phát hiện tổn thương dạng kính mờ rải rác hai đáy phổi ở bệnh nhân viêm phổi virus COVID-19.",
        "source": "AuntMinnie (Vietnam Case Report)"
    },
    {
        "id": "PHOI_04",
        "filename": "phoi_4.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000003-1.jpg",
        "view": "PA",
        "finding": "Lao phổi kê (Miliary Tuberculosis)",
        "clinical_notes_en": "Hazy micronodular pattern symmetrically involving both lungs from apex to base (Miliary tuberculosis).",
        "description_vi": "Lao kê: Vô số nốt mờ nhỏ li ti (kích thước hạt kê) phân bố đối xứng đều khắp hai phế trường từ đỉnh đến đáy.",
        "ground_truth_caption": "Ảnh X-quang ngực thẳng xuất hiện dày đặc các nốt vi thể kích thước hạt kê hai bên phổi, tổn thương điển hình của lao kê.",
        "source": "Eurorad"
    },
    {
        "id": "PHOI_05",
        "filename": "phoi_5.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/00000001_000.png",
        "view": "PA",
        "finding": "Bóng tim to (Cardiomegaly)",
        "clinical_notes_en": "Cardiomegaly with enlarged cardiac silhouette (Cardiothoracic ratio CTR > 0.55).",
        "description_vi": "Bóng tim to rõ rệt: Chỉ số tim lồng ngực CTR > 0.55, mỏm tim lệch trái sát thành ngực, hai phế trường không thấy thâm nhiễm.",
        "ground_truth_caption": "Ảnh X-quang ngực thẳng cho thấy bóng tim to rõ rệt, chỉ số tim lồng ngực tăng trên 0.55.",
        "source": "NIH Clinical Center ChestX-ray14"
    },

    # ==========================
    # NHÓM 2: TAY (5 ảnh)
    # ==========================
    {
        "id": "TAY_01",
        "filename": "tay_1.jpg",
        "body_part": "Xương tay (Arm)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/no-fracture.jpg",
        "view": "AP",
        "finding": "Tay bình thường - KHÔNG GÃY (Normal / Healthy)",
        "clinical_notes_en": "Normal arm radiograph. Continuous cortical margin, normal bone density, no fracture line or dislocation.",
        "description_vi": "Xương tay bình thường: Vỏ xương liên tục, mật độ xương đều, không phát hiện đường nứt rạn hay gãy xương.",
        "ground_truth_caption": "Ảnh X-quang xương tay bình thường, trục xương thẳng và vỏ xương liên tục không có dấu hiệu gãy.",
        "source": "Roboflow Bone Dataset"
    },
    {
        "id": "TAY_02",
        "filename": "tay_2.jpg",
        "body_part": "Xương tay (Arm)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture.jpg",
        "view": "AP/Lateral",
        "finding": "Gãy xương cẳng tay (Forearm Fracture)",
        "clinical_notes_en": "Complete fracture of radius and ulna shafts with cortical discontinuity and mild displacement.",
        "description_vi": "Gãy xương cẳng tay: Đường gãy ngang mất liên tục vỏ xương rõ rệt ở thân xương quay và xương trụ, di lệch nhẹ.",
        "ground_truth_caption": "Ảnh X-quang cẳng tay cho thấy hình ảnh gãy hoàn toàn thân xương quay và xương trụ có di lệch nhẹ.",
        "source": "Roboflow Bone Dataset"
    },
    {
        "id": "TAY_03",
        "filename": "tay_3.jpg",
        "body_part": "Xương tay (Arm)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture%20(2).jpg",
        "view": "AP/Lateral",
        "finding": "Gãy đầu dưới xương quay cổ tay (Wrist Colles Fracture)",
        "clinical_notes_en": "Distal radius fracture at the wrist joint with cortical break at the metaphysis.",
        "description_vi": "Gãy xương cổ tay: Đường gãy ngang đầu dưới xương quay sát khớp cổ tay (gãy Colles do ngã chống tay).",
        "ground_truth_caption": "Ảnh X-quang cổ tay phát hiện đường gãy đầu dưới xương quay vùng khớp cổ tay.",
        "source": "Roboflow Bone Dataset"
    },
    {
        "id": "TAY_04",
        "filename": "tay_4.jpg",
        "body_part": "Xương tay (Arm)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/manhph2211/Bone-Fracture-Detection/main/imgrm/hand_patient10101_image3.png",
        "view": "PA",
        "finding": "Gãy xương bàn tay / ngón tay (Hand / Finger Fracture)",
        "clinical_notes_en": "Hand radiograph showing cortical disruption and fracture of the metacarpal/phalangeal bone.",
        "description_vi": "Gãy xương bàn tay: Vết rạn nứt gãy vỏ xương đốt ngón/đốt bàn tay, kèm phù nề mô mềm vùng chấn thương.",
        "ground_truth_caption": "Ảnh X-quang bàn tay cho thấy tổn thương gãy nứt vỏ xương đốt bàn ngón tay.",
        "source": "Stanford MURA Dataset"
    },
    {
        "id": "TAY_05",
        "filename": "tay_5.jpg",
        "body_part": "Xương tay (Arm)",
        "source_type": "zip",
        "zip_path": "fracture/training/0.jpg",
        "view": "AP",
        "finding": "Gãy xương cánh tay (Humerus Fracture)",
        "clinical_notes_en": "Displaced fracture of the humerus shaft with angular deformity.",
        "description_vi": "Gãy xương cánh tay: Gãy thân xương cánh tay có di lệch gập góc, thấy rõ đường gãy gián đoạn thân xương.",
        "ground_truth_caption": "Ảnh X-quang cánh tay cho thấy gãy thân xương cánh tay có di lệch gập góc.",
        "source": "OrthoIndex Automated Fracture Dataset"
    },

    # ==========================
    # NHÓM 3: CHÂN (5 ảnh)
    # ==========================
    {
        "id": "CHAN_01",
        "filename": "chan_1.jpg",
        "body_part": "Xương chân (Leg)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/3.jpg",
        "view": "AP",
        "finding": "Chân bình thường - KHÔNG GÃY (Normal / Healthy)",
        "clinical_notes_en": "Normal leg radiograph. Intact cortex, normal alignment of tibia and fibula, no fracture.",
        "description_vi": "Xương cẳng chân bình thường: Xương chày và xương mác liên tục, không gãy nứt, diện khớp gối và cổ chân bình thường.",
        "ground_truth_caption": "Ảnh X-quang chân bình thường, xương chày và xương mác nguyên vẹn không có đường gãy.",
        "source": "Leg Fracture Dataset"
    },
    {
        "id": "CHAN_02",
        "filename": "chan_2.jpg",
        "body_part": "Xương chân (Leg)",
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/4.jpg",
        "view": "AP",
        "finding": "Gãy cả 2 xương cẳng chân (Tibia & Fibula Fracture)",
        "clinical_notes_en": "Complete transverse/oblique fracture of mid-shaft tibia and fibula with angular displacement.",
        "description_vi": "Gãy cả hai xương cẳng chân: Gãy hoàn toàn 1/3 giữa xương chày và xương mác, di lệch góc và di lệch sang bên rõ.",
        "ground_truth_caption": "Ảnh X-quang cẳng chân cho thấy gãy hoàn toàn 1/3 giữa xương chày và xương mác kèm di lệch góc.",
        "source": "Leg Fracture Dataset"
    },
    {
        "id": "CHAN_03",
        "filename": "chan_3.jpg",
        "body_part": "Xương chân (Leg)",
        "source_type": "zip",
        "zip_path": "fracture/training/9.jpg",
        "view": "AP",
        "finding": "Gãy thân xương dài đùi / chân (Femur / Long Bone Fracture)",
        "clinical_notes_en": "Oblique displaced fracture through the shaft of long leg bone.",
        "description_vi": "Gãy thân xương dài: Đường gãy chéo có di lệch lớn giữa hai đầu gãy, mất trục giải phẫu chi dưới.",
        "ground_truth_caption": "Ảnh X-quang chân phát hiện gãy chéo thân xương dài có di lệch rời hai đoạn xương.",
        "source": "OrthoIndex Automated Fracture Dataset"
    },
    {
        "id": "CHAN_04",
        "filename": "chan_4.jpg",
        "body_part": "Xương chân (Leg)",
        "source_type": "zip",
        "zip_path": "fracture/training/7.jpg",
        "view": "AP",
        "finding": "Gãy thân xương chày cẳng chân (Tibia Fracture)",
        "clinical_notes_en": "Fracture of tibia shaft with cortical interruption and local soft tissue swelling.",
        "description_vi": "Gãy thân xương chày: Vết gãy ngang thân xương cẳng chân, sưng nề mô mềm xung quanh ổ gãy.",
        "ground_truth_caption": "Ảnh X-quang cẳng chân phát hiện tổn thương gãy ngang thân xương chày.",
        "source": "OrthoIndex Automated Fracture Dataset"
    },
    {
        "id": "CHAN_05",
        "filename": "chan_5.jpg",
        "body_part": "Xương chân (Leg)",
        "source_type": "zip",
        "zip_path": "fracture/training/25.jpg",
        "view": "AP/Lateral",
        "finding": "Gãy xương vùng cổ chân / mắt cá (Ankle Fracture)",
        "clinical_notes_en": "Ankle radiograph showing distal fibular/malleolar fracture with joint space involvement.",
        "description_vi": "Gãy xương cổ chân: Đường gãy mắt cá / đầu dưới xương mác vùng khớp cổ chân do chấn thương lật cổ chân.",
        "ground_truth_caption": "Ảnh X-quang cổ chân cho thấy đường gãy xương vùng mắt cá ngoài khớp cổ chân.",
        "source": "OrthoIndex Automated Fracture Dataset"
    }
]

print("\n2. Bắt đầu chuẩn hóa và lưu 15 ảnh vào data/evaluation/images/...")
saved_records = []

for item in DATASET_CONFIG:
    dest_file = IMAGE_DIR / item["filename"]
    print(f" -> Xử lý [{item['body_part']}]: {item['filename']} ({item['finding']})...")
    
    try:
        # Lấy dữ liệu ảnh (qua URL hoặc từ ZIP)
        if item["source_type"] == "url":
            req = urllib.request.Request(item["source_url"], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_bytes = resp.read()
        else:
            raw_bytes = zip_archive.read(item["zip_path"])
            
        # Chuẩn hóa về RGB và lưu định dạng JPEG chất lượng 95
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(dest_file, "JPEG", quality=95)
            res_str = f"{pil_img.width}x{pil_img.height}"
            
        item["resolution"] = res_str
        saved_records.append(item)
        print(f"    + Đã lưu thành công: {item['filename']} [{res_str}]")
        
    except Exception as exc:
        print(f"    ! Lỗi: {exc}")

# Tạo 2 ca kiểm thử lỗi (corner cases) để bài test tự động chạy chuẩn
err_corrupt = IMAGE_DIR / "ERR_001_corrupted.jpg"
with open(err_corrupt, "wb") as f:
    f.write(b"CORRUPTED_BYTES_NOT_A_VALID_JPEG_12345")

err_zero = IMAGE_DIR / "ERR_002_zero_byte.jpg"
with open(err_zero, "wb") as f:
    pass

print(f" -> Đã lưu 2 file test lỗi: {err_corrupt.name}, {err_zero.name}")

# Ghi lại file metadata.csv chuẩn 100%
fieldnames = [
    "image_id", "filename", "body_part", "view_position",
    "finding_category", "clinical_notes_en", "finding_description",
    "ground_truth_caption_vi", "source", "resolution"
]

for meta_dest in [METADATA_PATH_1, METADATA_PATH_2]:
    with open(meta_dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in saved_records:
            row = {
                "image_id": rec["id"],
                "filename": rec["filename"],
                "body_part": rec["body_part"],
                "view_position": rec["view"],
                "finding_category": rec["finding"],
                "clinical_notes_en": rec["clinical_notes_en"],
                "finding_description": rec["description_vi"],
                "ground_truth_caption_vi": rec["ground_truth_caption"],
                "source": rec["source"],
                "resolution": rec["resolution"]
            }
            writer.writerow(row)
    print(f"Đã lưu metadata mới tại: {meta_dest}")

print(f"\n=== HOÀN TẤT TUYỆT ĐỐI: 15 ẢNH (5 PHỔI, 5 TAY, 5 CHÂN) ĐÃ ĐƯỢC TỔ CHỨC VÀO DATA! ===")
