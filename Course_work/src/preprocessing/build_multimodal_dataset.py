"""
Script xây dựng toàn diện tập dữ liệu y tế đa phương thức (Medical Multimodal Dataset):
- 7 bộ phận cơ thể x 5 ảnh chuẩn = 35 ảnh y tế:
    1. Phổi / Lồng ngực (Chest / Lung): phoi_1.jpg .. phoi_5.jpg
    2. Não / Sọ não (Brain / Head): nao_1.jpg .. nao_5.jpg
    3. Bụng / Ổ bụng (Abdomen): bung_1.jpg .. bung_5.jpg
    4. Khung chậu (Pelvic Cavity): khung_chau_1.jpg .. khung_chau_5.jpg
    5. Cổ / Vùng cổ (Neck): co_1.jpg .. co_5.jpg
    6. Xương tay (Arm / Hand): tay_1.jpg .. tay_5.jpg
    7. Xương chân (Leg / Foot): chan_1.jpg .. chan_5.jpg
- Kết nối bộ câu hỏi VQA tiếng Việt từ v1olet/VietMed-VQA (Hugging Face)
- Xuất metadata.csv và questions.csv
Phụ trách: Trí (Image Preprocessing & Data Preparation)
"""

import os
import sys
import csv
import json
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

# Import các hàm tiền xử lý đã hoàn thiện
REPO_ROOT = Path(r"E:\Course_work")
CW_ROOT = REPO_ROOT / "Course_work"
sys.path.insert(0, str(CW_ROOT))
sys.path.insert(0, str(REPO_ROOT))

try:
    from Course_work.src.preprocessing.image_processing import (
        validate_image,
        convert_to_rgb,
        resize_preserving_aspect_ratio,
    )
except ImportError:
    from src.preprocessing.image_processing import (
        validate_image,
        convert_to_rgb,
        resize_preserving_aspect_ratio,
    )

DEST_CW_IMAGES = CW_ROOT / "data" / "evaluation" / "images"
DEST_CW_EVAL = CW_ROOT / "data" / "evaluation"
DEST_ROOT_DATA = REPO_ROOT / "data"

for folder in [DEST_CW_IMAGES, DEST_CW_EVAL, DEST_ROOT_DATA]:
    folder.mkdir(parents=True, exist_ok=True)

print("1. Đang tải và kiểm tra SLAKE metadata và VietMed-VQA...")

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))

slake_train = fetch_json("https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/train.json")
slake_val = fetch_json("https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/validation.json")
slake_test = fetch_json("https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/test.json")

all_slake = {item["qid"]: item for item in (slake_train + slake_val + slake_test)}
viet_test = fetch_json("https://huggingface.co/datasets/v1olet/VietMed-VQA/raw/main/test_translated_SLAKE_vqa.json")

print(f"-> Đã tải {len(all_slake)} SLAKE items và {len(viet_test)} VietMed-VQA items.")

# Mở slake_imgs_temp.zip
zip_path = REPO_ROOT / "slake_imgs_temp.zip"
if not zip_path.exists():
    raise FileNotFoundError(f"Không tìm thấy {zip_path}")

slake_zip = zipfile.ZipFile(zip_path)
zip_namelist = set(slake_zip.namelist())

# Danh sách nguồn ảnh và mô tả lâm sàng cho 7 bộ phận
BODY_PARTS_CONFIG = {
    # 1. PHỔI (Lung / Chest) - SLAKE / VietMed-VQA
    "phoi_1.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab246/source.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "finding_category": "Bình thường / Cắt lớp ngực (Chest CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Chest CT scan slice at cardiac level showing heart chambers, mediastinal structures and lung parenchyma.",
        "finding_description": "Ảnh CT cắt lớp lồng ngực: Hiển thị nhu mô phổi hai bên và các buồng tim, cấu trúc trung thất bình thường.",
        "ground_truth_caption_vi": "Ảnh CT ngực cắt ngang lớp cắt tim, các cấu trúc tim và phổi nằm trong giới hạn giải phẫu bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },
    "phoi_2.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab272/source.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "finding_category": "Cắt lớp ngực trung thất (Chest Mediastinum CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Chest CT imaging showing aortic arch, pulmonary vessels and surrounding lung fields.",
        "finding_description": "Ảnh CT ngực lát cắt quai động mạch chủ và cuống mạch phổi, nhu mô phổi hai bên sáng rõ.",
        "ground_truth_caption_vi": "Ảnh CT lồng ngực hiển thị quai động mạch chủ và các mạch máu lớn trung thất.",
        "source": "SLAKE / VietMed-VQA"
    },
    "phoi_3.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab297/source.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "finding_category": "X-quang ngực thẳng (Chest Radiograph)",
        "view_position": "PA",
        "clinical_notes_en": "Chest X-Ray showing clear bilateral lung fields, normal cardiothoracic ratio and costophrenic angles.",
        "finding_description": "Ảnh X-quang phổi thẳng: Hai phế trường sáng đều, bóng tim bình thường, góc sườn hoành hai bên nhọn.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng PA bình thường, không thấy thâm nhiễm phế trường hay tràn dịch màng phổi.",
        "source": "SLAKE / VietMed-VQA"
    },
    "phoi_4.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab298/source.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "finding_category": "X-quang ngực thẳng (Chest Radiograph)",
        "view_position": "PA",
        "clinical_notes_en": "Frontal chest radiograph showing lungs, clavicles, ribs and diaphragmatic contours.",
        "finding_description": "Ảnh X-quang phổi thẳng kiểm tra khung xương sườn, vòm hoành và phế trường hai bên rõ nét.",
        "ground_truth_caption_vi": "Ảnh X-quang phổi tư thế thẳng ghi nhận giải phẫu lồng ngực và trường phổi bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },
    "phoi_5.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab175/source.jpg",
        "body_part": "Phổi / Lồng ngực (Chest)",
        "finding_category": "X-quang lồng ngực (Chest X-Ray)",
        "view_position": "PA",
        "clinical_notes_en": "Chest radiograph showing bilateral lung parenchyma and mediastinal borders.",
        "finding_description": "Ảnh X-quang ngực quan sát chi tiết hai thùy phổi và bờ cơ tim trung thất.",
        "ground_truth_caption_vi": "Ảnh X-quang lồng ngực đánh giá nhu mô phổi và rốn phổi hai bên.",
        "source": "SLAKE / VietMed-VQA"
    },

    # 2. NÃO (Brain / Head) - SLAKE / VietMed-VQA
    "nao_1.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab456/source.jpg",
        "body_part": "Não / Sọ não (Brain)",
        "finding_category": "Cộng hưởng từ sọ não (Brain MRI)",
        "view_position": "Axial MRI",
        "clinical_notes_en": "Brain MRI T2/FLAIR axial slice displaying cerebral hemispheres, ventricles and white matter.",
        "finding_description": "Ảnh MRI não cắt ngang: Bán cầu đại não hai bên đối xứng, hệ thống não thất rõ ràng, không thấy phù não hay khối chèn ép.",
        "ground_truth_caption_vi": "Ảnh chụp cộng hưởng từ (MRI) não lát cắt ngang thể hiện cấu trúc nhu mô não và hệ thống não thất.",
        "source": "SLAKE / VietMed-VQA"
    },
    "nao_2.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab397/source.jpg",
        "body_part": "Não / Sọ não (Brain)",
        "finding_category": "Cắt lớp vi tính sọ não (Brain CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Brain CT scan non-contrast axial cut showing basal ganglia and surrounding brain parenchyma.",
        "finding_description": "Ảnh CT sọ não không cản quang: Quan sát hạch nền và các rãnh cuộn não, phân biệt chất xám - chất trắng rõ.",
        "ground_truth_caption_vi": "Ảnh CT sọ não cắt ngang vùng hạch nền không phát hiện xuất huyết hay ổ nhồi máu cấp tính.",
        "source": "SLAKE / VietMed-VQA"
    },
    "nao_3.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab398/source.jpg",
        "body_part": "Não / Sọ não (Brain)",
        "finding_category": "Cắt lớp vi tính sọ não (Brain CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Brain CT scan showing midbrain and temporal lobes with normal intracranial density.",
        "finding_description": "Ảnh CT sọ não lát cắt qua thân não và thùy thái dương, tỷ trọng mô não đồng đều.",
        "ground_truth_caption_vi": "Ảnh cắt lớp vi tính sọ não đánh giá cấu trúc thân não và khoang dưới nhện bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },
    "nao_4.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab399/source.jpg",
        "body_part": "Não / Sọ não (Brain)",
        "finding_category": "Cắt lớp vi tính sọ não (Brain CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Axial head CT scan slice showing posterior fossa and cerebellar hemispheres.",
        "finding_description": "Ảnh CT sọ não lát cắt hố sau: Quan sát tiểu não và não thất IV cân đối, xương sọ nguyên vẹn.",
        "ground_truth_caption_vi": "Ảnh CT sọ não cắt ngang vùng hố sau kiểm tra bán cầu tiểu não và cầu não.",
        "source": "SLAKE / VietMed-VQA"
    },
    "nao_5.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab470/source.jpg",
        "body_part": "Não / Sọ não (Brain)",
        "finding_category": "Cộng hưởng từ sọ não (Brain MRI)",
        "view_position": "Axial MRI",
        "clinical_notes_en": "T1/T2 weighted brain MRI showing upper convexities and frontal lobes.",
        "finding_description": "Ảnh MRI não lát cắt đỉnh vòm sọ: Đánh giá vỏ não thùy trán và thùy đỉnh, không có tổn thương khu trú.",
        "ground_truth_caption_vi": "Ảnh MRI sọ não vùng đỉnh vòm đánh giá các rãnh vỏ não và nhu mô thùy trán.",
        "source": "SLAKE / VietMed-VQA"
    },

    # 3. BỤNG (Abdomen) - SLAKE / VietMed-VQA
    "bung_1.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab204/source.jpg",
        "body_part": "Bụng / Ổ bụng (Abdomen)",
        "finding_category": "Cắt lớp vi tính ổ bụng (Abdomen CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Abdominal CT scan showing liver, spleen, stomach and bilateral kidneys.",
        "finding_description": "Ảnh CT cắt lớp ổ bụng: Thấy rõ gan, lách, dạ dày và hai thận, bờ các cơ quan đều đặn.",
        "ground_truth_caption_vi": "Ảnh CT ổ bụng cắt ngang hiển thị nhu mô gan, lách và hai thận bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },
    "bung_2.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab268/source.jpg",
        "body_part": "Bụng / Ổ bụng (Abdomen)",
        "finding_category": "Cắt lớp vi tính ổ bụng (Abdomen CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Contrast-enhanced abdominal CT showing pancreas, aorta and mesenteric vessels.",
        "finding_description": "Ảnh CT ổ bụng tiêm thuốc cản quang: Nhu mô tụy bắt thuốc đều, động mạch chủ bụng bình thường.",
        "ground_truth_caption_vi": "Ảnh CT ổ bụng cản quang đánh giá tụy và bó mạch mạc treo tràng trên.",
        "source": "SLAKE / VietMed-VQA"
    },
    "bung_3.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab259/source.jpg",
        "body_part": "Bụng / Ổ bụng (Abdomen)",
        "finding_category": "Cắt lớp vi tính ổ bụng (Abdomen CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Abdominal CT slice showing renal parenchyma and retroperitoneal space.",
        "finding_description": "Ảnh CT bụng lát cắt ngang hai thận: Vỏ thận và xoang thận hai bên rõ, không thấy sỏi hay ứ nước.",
        "ground_truth_caption_vi": "Ảnh cắt lớp vi tính bụng lát cắt thận và khoang sau phúc mạc bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },
    "bung_4.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab289/source.jpg",
        "body_part": "Bụng / Ổ bụng (Abdomen)",
        "finding_category": "Cắt lớp vi tính ổ bụng (Abdomen CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Abdominal CT slice focused on bowel loops and mesenteric fat planes.",
        "finding_description": "Ảnh CT bụng quan sát các quai ruột non và mỡ mạc treo, không có hơi tự do hay dịch ổ bụng.",
        "ground_truth_caption_vi": "Ảnh CT ổ bụng cắt ngang kiểm tra các quai ruột và thành bụng trước.",
        "source": "SLAKE / VietMed-VQA"
    },
    "bung_5.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab243/source.jpg",
        "body_part": "Bụng / Ổ bụng (Abdomen)",
        "finding_category": "Cắt lớp vi tính ổ bụng (Abdomen CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Abdominal CT slice showing gall bladder, lower liver edge and duodenum.",
        "finding_description": "Ảnh CT bụng lát cắt qua túi mật và tá tràng: Thành túi mật mỏng, lòng dịch đồng nhất.",
        "ground_truth_caption_vi": "Ảnh CT ổ bụng đánh giá vùng hạ sườn phải, túi mật và tá tràng.",
        "source": "SLAKE / VietMed-VQA"
    },

    # 4. KHUNG CHẬU (Pelvic Cavity) - SLAKE / VietMed-VQA
    "khung_chau_1.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab616/source.jpg",
        "body_part": "Khung chậu (Pelvic Cavity)",
        "finding_category": "Cắt lớp vi tính tiểu khung (Pelvis CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Pelvic CT scan slice showing bladder, rectum and pelvic sidewalls.",
        "finding_description": "Ảnh CT tiểu khung cắt ngang: Bàng quang căng đều, trực tràng và các cơ đáy chậu đối xứng.",
        "ground_truth_caption_vi": "Ảnh CT khung chậu cắt ngang quan sát bàng quang, trực tràng và hốc chậu.",
        "source": "SLAKE / VietMed-VQA"
    },
    "khung_chau_2.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab612/source.jpg",
        "body_part": "Khung chậu (Pelvic Cavity)",
        "finding_category": "Cắt lớp vi tính tiểu khung (Pelvis CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Pelvic CT slice demonstrating iliac bones, sacrum and pelvic floor musculature.",
        "finding_description": "Ảnh CT tiểu khung quan sát xương cánh chậu hai bên và xương cùng cụt nguyên vẹn.",
        "ground_truth_caption_vi": "Ảnh cắt lớp vi tính khung chậu kiểm tra cấu trúc xương chậu và cơ đáy chậu.",
        "source": "SLAKE / VietMed-VQA"
    },
    "khung_chau_3.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab608/source.jpg",
        "body_part": "Khung chậu (Pelvic Cavity)",
        "finding_category": "Cắt lớp vi tính tiểu khung (Pelvis CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Pelvic cavity CT scan showing femoral heads and acetabulum articulation.",
        "finding_description": "Ảnh CT khung chậu lát cắt ngang ổ cối và chỏm xương đùi hai bên cân đối.",
        "ground_truth_caption_vi": "Ảnh CT tiểu khung đánh giá khớp háng và ổ cối xương chậu hai bên.",
        "source": "SLAKE / VietMed-VQA"
    },
    "khung_chau_4.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab619/source.jpg",
        "body_part": "Khung chậu (Pelvic Cavity)",
        "finding_category": "Cắt lớp vi tính tiểu khung (Pelvis CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Pelvic CT cut at lower pelvic level displaying prostate/vaginal region and ischiorectal fossa.",
        "finding_description": "Ảnh CT tiểu khung mức thấp kiểm tra vùng hố ngồi trực tràng và đáy chậu.",
        "ground_truth_caption_vi": "Ảnh CT khung chậu lát cắt thấp đánh giá cơ thắt và mô mềm đáy chậu.",
        "source": "SLAKE / VietMed-VQA"
    },
    "khung_chau_5.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab603/source.jpg",
        "body_part": "Khung chậu (Pelvic Cavity)",
        "finding_category": "Cắt lớp vi tính tiểu khung (Pelvis CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Pelvic CT scan cut demonstrating sigmoid colon and upper pelvic inlet.",
        "finding_description": "Ảnh CT tiểu khung qua đại tràng sigma và eo trên khung chậu, lòng đại tràng thông thoáng.",
        "ground_truth_caption_vi": "Ảnh CT cắt lớp khung chậu hiển thị eo trên và đại tràng sigma bình thường.",
        "source": "SLAKE / VietMed-VQA"
    },

    # 5. CỔ (Neck) - SLAKE / VietMed-VQA
    "co_1.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab410/source.jpg",
        "body_part": "Cổ / Vùng cổ (Neck)",
        "finding_category": "Cắt lớp vi tính vùng cổ (Neck CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Neck CT scan slice at thyroid level displaying thyroid gland, trachea and cervical vessels.",
        "finding_description": "Ảnh CT vùng cổ cắt ngang: Hai thùy tuyến giáp kích thước bình thường, khí quản chính giữa, động mạch cảnh thông suốt.",
        "ground_truth_caption_vi": "Ảnh CT vùng cổ cắt ngang qua tuyến giáp, khí quản và bó mạch cảnh hai bên.",
        "source": "SLAKE / VietMed-VQA"
    },
    "co_2.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab431/source.jpg",
        "body_part": "Cổ / Vùng cổ (Neck)",
        "finding_category": "Cắt lớp vi tính vùng cổ (Neck CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Neck CT scan slice displaying larynx, vocal cords and cervical spine vertebra.",
        "finding_description": "Ảnh CT cổ lát cắt qua thanh quản và dây thanh âm, thân đốt sống cổ nguyên vẹn.",
        "ground_truth_caption_vi": "Ảnh CT cắt lớp vùng cổ đánh giá thanh quản và đốt sống cổ.",
        "source": "SLAKE / VietMed-VQA"
    },
    "co_3.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab434/source.jpg",
        "body_part": "Cổ / Vùng cổ (Neck)",
        "finding_category": "Cắt lớp vi tính vùng cổ (Neck CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Neck CT slice at pharyngeal level showing oropharynx and submandibular soft tissue.",
        "finding_description": "Ảnh CT cổ lát cắt qua họng miệng và tuyến dưới hàm, khoang sau họng sáng rõ.",
        "ground_truth_caption_vi": "Ảnh CT vùng cổ lát cắt khẩu hầu và cấu trúc mô mềm dưới hàm.",
        "source": "SLAKE / VietMed-VQA"
    },
    "co_4.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab422/source.jpg",
        "body_part": "Cổ / Vùng cổ (Neck)",
        "finding_category": "Cắt lớp vi tính vùng cổ (Neck CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Neck CT slice showing cervical lymph node stations and sternocleidomastoid muscles.",
        "finding_description": "Ảnh CT cổ quan sát cơ ức đòn chũm hai bên và các nhóm hạch cổ không phì đại.",
        "ground_truth_caption_vi": "Ảnh cắt lớp vi tính vùng cổ đánh giá cơ cổ và các chuỗi hạch bạch huyết.",
        "source": "SLAKE / VietMed-VQA"
    },
    "co_5.jpg": {
        "source_type": "slake_zip",
        "zip_member": "imgs/xmlab423/source.jpg",
        "body_part": "Cổ / Vùng cổ (Neck)",
        "finding_category": "Cắt lớp vi tính vùng cổ (Neck CT)",
        "view_position": "Axial CT",
        "clinical_notes_en": "Neck CT scan cut at upper thoracic inlet demonstrating apical lungs and trachea entry.",
        "finding_description": "Ảnh CT cổ lát cắt chuyển tiếp ngực cổ: Đoạn khí quản dưới và đỉnh phổi hai bên bình thường.",
        "ground_truth_caption_vi": "Ảnh CT cổ lát cắt thấp vùng nền cổ giáp ranh đỉnh lồng ngực.",
        "source": "SLAKE / VietMed-VQA"
    },

    # 6. TAY (Arm / Hand / Wrist)
    "tay_1.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/no-fracture.jpg",
        "body_part": "Xương tay (Arm)",
        "finding_category": "Tay bình thường - KHÔNG GÃY (Normal)",
        "view_position": "AP",
        "clinical_notes_en": "Normal arm radiograph. Continuous cortical margin, normal bone density, no fracture line or dislocation.",
        "finding_description": "Xương tay bình thường: Vỏ xương liên tục, mật độ xương đều, không phát hiện đường nứt rạn hay gãy xương.",
        "ground_truth_caption_vi": "Ảnh X-quang xương tay bình thường, trục xương thẳng và vỏ xương liên tục không có dấu hiệu gãy.",
        "source": "Roboflow Bone Dataset"
    },
    "tay_2.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture.jpg",
        "body_part": "Xương tay (Arm)",
        "finding_category": "Gãy xương cẳng tay (Forearm Fracture)",
        "view_position": "AP/Lateral",
        "clinical_notes_en": "Complete fracture of radius and ulna shafts with cortical discontinuity and mild displacement.",
        "finding_description": "Gãy xương cẳng tay: Đường gãy ngang mất liên tục vỏ xương rõ rệt ở thân xương quay và xương trụ, di lệch nhẹ.",
        "ground_truth_caption_vi": "Ảnh X-quang cẳng tay cho thấy hình ảnh gãy hoàn toàn thân xương quay và xương trụ có di lệch nhẹ.",
        "source": "Roboflow Bone Dataset"
    },
    "tay_3.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture%20(2).jpg",
        "body_part": "Xương tay (Arm)",
        "finding_category": "Gãy đầu dưới xương quay cổ tay (Colles Fracture)",
        "view_position": "AP/Lateral",
        "clinical_notes_en": "Distal radius fracture at the wrist joint with cortical break at the metaphysis.",
        "finding_description": "Gãy xương cổ tay: Đường gãy ngang đầu dưới xương quay sát khớp cổ tay (gãy Colles do ngã chống tay).",
        "ground_truth_caption_vi": "Ảnh X-quang cổ tay phát hiện đường gãy đầu dưới xương quay vùng khớp cổ tay.",
        "source": "Roboflow Bone Dataset"
    },
    "tay_4.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/manhph2211/Bone-Fracture-Detection/main/imgrm/hand_patient10101_image3.png",
        "body_part": "Xương tay (Arm)",
        "finding_category": "Gãy xương bàn / ngón tay (Hand / Finger Fracture)",
        "view_position": "PA",
        "clinical_notes_en": "Hand radiograph showing cortical disruption and fracture of the metacarpal/phalangeal bone.",
        "finding_description": "Gãy xương bàn tay: Vết rạn nứt gãy vỏ xương đốt ngón/đốt bàn tay, kèm phù nề mô mềm vùng chấn thương.",
        "ground_truth_caption_vi": "Ảnh X-quang bàn tay cho thấy tổn thương gãy nứt vỏ xương đốt bàn ngón tay.",
        "source": "Stanford MURA Dataset"
    },
    "tay_5.jpg": {
        "source_type": "archive_zip",
        "zip_member": "fracture/training/0.jpg",
        "body_part": "Xương tay (Arm)",
        "finding_category": "Gãy xương cánh tay (Humerus Fracture)",
        "view_position": "AP",
        "clinical_notes_en": "Displaced fracture of the humerus shaft with angular deformity.",
        "finding_description": "Gãy xương cánh tay: Gãy thân xương cánh tay có di lệch gập góc, thấy rõ đường gãy gián đoạn thân xương.",
        "ground_truth_caption_vi": "Ảnh X-quang cánh tay cho thấy gãy thân xương cánh tay có di lệch gập góc.",
        "source": "Automated Fracture Dataset"
    },

    # 7. CHÂN (Leg / Knee / Ankle)
    "chan_1.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/3.jpg",
        "body_part": "Xương chân (Leg)",
        "finding_category": "Chân bình thường - KHÔNG GÃY (Normal)",
        "view_position": "AP",
        "clinical_notes_en": "Normal leg radiograph. Intact cortex, normal alignment of tibia and fibula, no fracture.",
        "finding_description": "Xương cẳng chân bình thường: Xương chày và xương mác liên tục, không gãy nứt, diện khớp gối và cổ chân bình thường.",
        "ground_truth_caption_vi": "Ảnh X-quang chân bình thường, xương chày và xương mác nguyên vẹn không có đường gãy.",
        "source": "Leg Fracture Dataset"
    },
    "chan_2.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/4.jpg",
        "body_part": "Xương chân (Leg)",
        "finding_category": "Gãy cả 2 xương cẳng chân (Tibia & Fibula Fracture)",
        "view_position": "AP",
        "clinical_notes_en": "Complete transverse/oblique fracture of mid-shaft tibia and fibula with angular displacement.",
        "finding_description": "Gãy cả hai xương cẳng chân: Gãy hoàn toàn 1/3 giữa xương chày và xương mác, di lệch góc và di lệch sang bên rõ.",
        "ground_truth_caption_vi": "Ảnh X-quang cẳng chân cho thấy gãy hoàn toàn 1/3 giữa xương chày và xương mác kèm di lệch góc.",
        "source": "Leg Fracture Dataset"
    },
    "chan_3.jpg": {
        "source_type": "url",
        "source_url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/1.jpg",
        "body_part": "Xương chân (Leg)",
        "finding_category": "Gãy mâm chày / khớp gối (Tibial Plateau Fracture)",
        "view_position": "AP",
        "clinical_notes_en": "Tibial plateau fracture at the knee joint with articular surface depression.",
        "finding_description": "Gãy mâm chày khớp gối: Đường nứt gãy lún mặt khớp mâm chày, khe khớp gối hẹp không đều.",
        "ground_truth_caption_vi": "Ảnh X-quang khớp gối phát hiện đường gãy lún mâm chày xương cẳng chân.",
        "source": "Leg Fracture Dataset"
    },
    "chan_4.jpg": {
        "source_type": "archive_zip",
        "zip_member": "fracture/training/10.jpg",
        "body_part": "Xương chân (Leg)",
        "finding_category": "Gãy mắt cá chân / cổ chân (Ankle Malleolus Fracture)",
        "view_position": "AP/Mortise",
        "clinical_notes_en": "Bimalleolar ankle fracture with dislocation of the talocrural joint.",
        "finding_description": "Gãy cổ chân: Gãy mắt cá ngoài xương mác kèm tổn thương mộng chày mác vùng khớp cổ chân.",
        "ground_truth_caption_vi": "Ảnh X-quang cổ chân cho thấy đường gãy xương vùng mắt cá có di lệch nhẹ.",
        "source": "Automated Fracture Dataset"
    },
    "chan_5.jpg": {
        "source_type": "archive_zip",
        "zip_member": "fracture/training/15.jpg",
        "body_part": "Xương chân (Leg)",
        "finding_category": "Gãy thân xương đùi (Femoral Shaft Fracture)",
        "view_position": "AP",
        "clinical_notes_en": "Comminuted fracture of the mid femoral shaft with overriding bone fragments.",
        "finding_description": "Gãy xương đùi: Gãy nhiều mảnh thân xương đùi có di lệch chồng ngắn và di lệch góc điển hình.",
        "ground_truth_caption_vi": "Ảnh X-quang xương đùi ghi nhận gãy thân xương đùi phức tạp có di lệch mảnh gãy.",
        "source": "Automated Fracture Dataset"
    }
}

# Tải archive.zip cho các ca gãy xương nếu cần
print("2. Đang chuẩn bị kho ảnh gãy xương bổ sung (archive.zip)...")
archive_url = "https://raw.githubusercontent.com/arjun2703-ai/Automated-Fracture-Detection/main/archive.zip"
req_archive = urllib.request.Request(archive_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_archive, timeout=25) as resp:
    archive_zip = zipfile.ZipFile(io.BytesIO(resp.read()))

# 3. Trích xuất và tiền xử lý toàn bộ 35 ảnh
print("3. Đang xử lý và lưu 35 ảnh y tế vào thư mục data...")

metadata_rows = []
total_saved = 0

for idx, (filename, cfg) in enumerate(BODY_PARTS_CONFIG.items(), 1):
    raw_img = None
    if cfg["source_type"] == "slake_zip":
        member_name = cfg["zip_member"]
        if member_name in zip_namelist:
            data = slake_zip.read(member_name)
            raw_img = Image.open(io.BytesIO(data))
    elif cfg["source_type"] == "archive_zip":
        member_name = cfg["zip_member"]
        data = archive_zip.read(member_name)
        raw_img = Image.open(io.BytesIO(data))
    elif cfg["source_type"] == "url":
        url = cfg["source_url"]
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw_img = Image.open(io.BytesIO(resp.read()))

    if raw_img is None:
        raise RuntimeError(f"Không thể tải ảnh cho {filename}")

    # Chuyển đổi chuẩn hóa RGB & giữ tỷ lệ
    rgb_img = convert_to_rgb(raw_img)
    processed_img = resize_preserving_aspect_ratio(rgb_img, target_max_dim=2048)
    res_str = f"{processed_img.width}x{processed_img.height}"

    # Lưu vào cả Course_work/data/evaluation/images/ và thư mục E:/Course_work/data/
    path_cw = DEST_CW_IMAGES / filename
    path_root = DEST_ROOT_DATA / filename
    processed_img.save(path_cw, "JPEG", quality=95)
    processed_img.save(path_root, "JPEG", quality=95)

    meta_row = {
        "image_id": f"{filename.split('.')[0].upper()}",
        "filename": filename,
        "body_part": cfg["body_part"],
        "view_position": cfg["view_position"],
        "finding_category": cfg["finding_category"],
        "clinical_notes_en": cfg["clinical_notes_en"],
        "finding_description": cfg["finding_description"],
        "ground_truth_caption_vi": cfg["ground_truth_caption_vi"],
        "source": cfg["source"],
        "resolution": res_str
    }
    metadata_rows.append(meta_row)
    total_saved += 1
    print(f"  [{total_saved}/35] Đã lưu {filename} ({res_str}) - {cfg['body_part']}")

# Tạo lại các file lỗi kiểm thử trong DEST_CW_IMAGES
err_corrupt = DEST_CW_IMAGES / "ERR_001_corrupted.jpg"
with open(err_corrupt, "wb") as f:
    f.write(b"CORRUPTED_NOT_A_VALID_JPEG_HEADER_DATA_123456789")

err_zero = DEST_CW_IMAGES / "ERR_002_zero_byte.jpg"
err_zero.touch(exist_ok=True)

# 4. Trích xuất câu hỏi VQA tiếng Việt từ VietMed-VQA
print("4. Đang ánh xạ và tạo câu hỏi VQA tiếng Việt...")

# Map SLAKE questions
questions_rows = []
q_counter = 1

# Map VietMed-VQA questions cho 25 ảnh SLAKE (phổi, não, bụng, khung chậu, cổ)
slake_to_filename = {
    cfg["zip_member"]: fname for fname, cfg in BODY_PARTS_CONFIG.items() if cfg["source_type"] == "slake_zip"
}

for item in viet_test:
    img_path = item.get("image_path", "")
    qid_str = img_path.split("/")[-1].replace(".jpg", "")
    if qid_str.isdigit():
        qid = int(qid_str)
        if qid in all_slake:
            slake_img = "imgs/" + all_slake[qid].get("img_name", "")
            if slake_img in slake_to_filename:
                target_fname = slake_to_filename[slake_img]
                cfg = BODY_PARTS_CONFIG[target_fname]
                questions_rows.append({
                    "question_id": f"Q_{q_counter:04d}",
                    "image_name": target_fname,
                    "body_part": cfg["body_part"],
                    "question_vi": item.get("input", "").strip(),
                    "answer_vi": item.get("output", "").strip(),
                    "question_en": item.get("original_input", "").strip(),
                    "answer_en": item.get("original_output", "").strip(),
                    "modality": cfg["view_position"],
                    "source": "v1olet/VietMed-VQA"
                })
                q_counter += 1

# Bổ sung câu hỏi VQA lâm sàng tiếng Việt cho 10 ảnh tay & chân
EXTREMITIES_QA = [
    # TAY_1 (Bình thường)
    ("tay_1.jpg", "Xương tay (Arm)", "Hình ảnh này chụp bộ phận nào trên cơ thể?", "Xương tay", "Which part of the body does this image show?", "Arm", "AP"),
    ("tay_1.jpg", "Xương tay (Arm)", "Có phát hiện dấu hiệu gãy xương trong hình ảnh không?", "Không", "Is there any sign of bone fracture in the image?", "No", "AP"),
    ("tay_1.jpg", "Xương tay (Arm)", "Vỏ xương có liên tục không?", "Có", "Is the bone cortex continuous?", "Yes", "AP"),

    # TAY_2 (Gãy cẳng tay)
    ("tay_2.jpg", "Xương tay (Arm)", "Hình ảnh này thể hiện bệnh lý gì?", "Gãy xương cẳng tay", "What pathology does this image show?", "Forearm fracture", "AP/Lateral"),
    ("tay_2.jpg", "Xương tay (Arm)", "Có bao nhiêu xương bị gãy trong ảnh?", "2 xương (xương quay và xương trụ)", "How many bones are fractured?", "2 bones (radius and ulna)", "AP/Lateral"),
    ("tay_2.jpg", "Xương tay (Arm)", "Đường gãy xương có di lệch không?", "Có", "Is the fracture line displaced?", "Yes", "AP/Lateral"),

    # TAY_3 (Gãy cổ tay)
    ("tay_3.jpg", "Xương tay (Arm)", "Vết gãy xương nằm ở vị trí nào?", "Đầu dưới xương quay cổ tay", "Where is the fracture located?", "Distal radius at wrist joint", "AP/Lateral"),
    ("tay_3.jpg", "Xương tay (Arm)", "Đây là dạng gãy xương gì thường gặp khi ngã chống tay?", "Gãy Colles", "What type of fracture commonly occurs after falling on outstretched hand?", "Colles fracture", "AP/Lateral"),

    # TAY_4 (Gãy bàn tay)
    ("tay_4.jpg", "Xương tay (Arm)", "Đây là ảnh chụp X-quang của bộ phận nào?", "Bàn tay", "Which part of the body is this X-ray taken of?", "Hand", "PA"),
    ("tay_4.jpg", "Xương tay (Arm)", "Có đường rạn nứt xương ở đốt bàn hoặc đốt ngón tay không?", "Có", "Is there a cortical crack in metacarpal or phalangeal bone?", "Yes", "PA"),

    # TAY_5 (Gãy cánh tay)
    ("tay_5.jpg", "Xương tay (Arm)", "Xương nào bị gãy trong hình ảnh này?", "Xương cánh tay", "Which bone is fractured in this image?", "Humerus", "AP"),
    ("tay_5.jpg", "Xương tay (Arm)", "Vết gãy ở thân xương cánh tay có di lệch gập góc không?", "Có", "Is there angular displacement in the humerus shaft fracture?", "Yes", "AP"),

    # CHAN_1 (Bình thường)
    ("chan_1.jpg", "Xương chân (Leg)", "Hình ảnh này chụp cấu trúc giải phẫu nào?", "Xương cẳng chân", "Which anatomical structure is shown in this image?", "Leg / Tibia and Fibula", "AP"),
    ("chan_1.jpg", "Xương chân (Leg)", "Xương chày và xương mác có bị gãy không?", "Không", "Are the tibia and fibula fractured?", "No", "AP"),
    ("chan_1.jpg", "Xương chân (Leg)", "Trục xương cẳng chân có bình thường không?", "Có", "Is the leg bone alignment normal?", "Yes", "AP"),

    # CHAN_2 (Gãy 2 xương cẳng chân)
    ("chan_2.jpg", "Xương chân (Leg)", "Hình ảnh này cho thấy tổn thương gì?", "Gãy xương chày và xương mác", "What injury is visible in this image?", "Tibia and fibula fracture", "AP"),
    ("chan_2.jpg", "Xương chân (Leg)", "Đoạn nào của xương cẳng chân bị gãy?", "1/3 giữa thân xương", "Which segment of the leg bone is fractured?", "Mid-shaft", "AP"),
    ("chan_2.jpg", "Xương chân (Leg)", "Có di lệch góc giữa các đoạn gãy không?", "Có", "Is there angular deformity between bone fragments?", "Yes", "AP"),

    # CHAN_3 (Gãy mâm chày)
    ("chan_3.jpg", "Xương chân (Leg)", "Vết gãy nằm ở gần khớp nào?", "Khớp gối", "Near which joint is the fracture located?", "Knee joint", "AP"),
    ("chan_3.jpg", "Xương chân (Leg)", "Có hình ảnh lún mặt khớp mâm chày không?", "Có", "Is there depression of the tibial plateau articular surface?", "Yes", "AP"),

    # CHAN_4 (Gãy cổ chân)
    ("chan_4.jpg", "Xương chân (Leg)", "Bộ phận nào đang được khảo sát trong hình?", "Khớp cổ chân", "Which body part is examined in this picture?", "Ankle joint", "AP/Mortise"),
    ("chan_4.jpg", "Xương chân (Leg)", "Có tổn thương gãy xương vùng mắt cá chân không?", "Có", "Is there a fracture injury in the malleolar region?", "Yes", "AP/Mortise"),

    # CHAN_5 (Gãy xương đùi)
    ("chan_5.jpg", "Xương chân (Leg)", "Xương lớn nào của chi dưới bị gãy trong ảnh?", "Xương đùi", "Which large lower limb bone is fractured?", "Femur", "AP"),
    ("chan_5.jpg", "Xương chân (Leg)", "Tổn thương gãy thân xương đùi có phức tạp không?", "Có, có mảnh rời và di lệch", "Is the femoral shaft fracture complex?", "Yes, comminuted with displacement", "AP"),
]

for fname, bpart, q_vi, a_vi, q_en, a_en, mod in EXTREMITIES_QA:
    questions_rows.append({
        "question_id": f"Q_{q_counter:04d}",
        "image_name": fname,
        "body_part": bpart,
        "question_vi": q_vi,
        "answer_vi": a_vi,
        "question_en": q_en,
        "answer_en": a_en,
        "modality": mod,
        "source": "Clinical Orthopedics VQA"
    })
    q_counter += 1

print(f"-> Tổng cộng đã tạo {len(questions_rows)} câu hỏi VQA tiếng Việt hoàn chỉnh!")

# 5. Ghi các file CSV vào các thư mục tương ứng
fieldnames_meta = [
    "image_id", "filename", "body_part", "view_position", "finding_category",
    "clinical_notes_en", "finding_description", "ground_truth_caption_vi",
    "source", "resolution"
]

for out_csv in [DEST_CW_EVAL / "metadata.csv", DEST_CW_IMAGES / "metadata.csv", DEST_ROOT_DATA / "metadata.csv"]:
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_meta)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Đã lưu: {out_csv}")

fieldnames_q = [
    "question_id", "image_name", "body_part", "question_vi", "answer_vi",
    "question_en", "answer_en", "modality", "source"
]

for out_csv in [DEST_CW_IMAGES / "questions.csv", DEST_CW_EVAL / "questions.csv", DEST_ROOT_DATA / "questions.csv"]:
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_q)
        writer.writeheader()
        writer.writerows(questions_rows)
    print(f"Đã lưu: {out_csv}")

print("\n=== HOÀN TẤT THÀNH CÔNG TẬP DỮ LIỆU ĐA PHƯƠNG THỨC 35 ẢNH ===")
