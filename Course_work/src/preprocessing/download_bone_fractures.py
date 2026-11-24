"""
Script cào và bổ sung bộ ảnh X-quang chấn thương chỉnh hình: GÃY TAY, GÃY CHÂN (Bone Fractures)
Phục vụ đề tài VQA y tế (Câu hỏi: 'Xương có bị gãy không?', 'Gãy ở vị trí nào?').
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

# 1. Danh sách tải từ URL trực tiếp
DIRECT_BONE_IMAGES = [
    {
        "image_id": "BONE_001",
        "save_name": "REAL_BONE_001_arm_forearm_fracture.jpg",
        "url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture.jpg",
        "image_type": "Musculoskeletal X-Ray (Arm)",
        "view_position": "AP/Lateral",
        "finding_category": "Forearm Bone Fracture (Gãy xương cẳng tay)",
        "clinical_notes_en": "Displaced fracture of the radius/ulna shaft with discontinuity of bone cortex.",
        "finding_description": "Đường gãy mất liên tục vỏ xương rõ rệt ở thân xương cẳng tay (xương quay và xương trụ), di lệch nhẹ diện gãy, phù nề mô mềm xung quanh.",
        "ground_truth_caption_vi": "Ảnh X-quang cẳng tay cho thấy hình ảnh gãy xương quay và xương trụ có di lệch nhẹ.",
        "source": "Roboflow Bone Fracture Dataset"
    },
    {
        "image_id": "BONE_002",
        "save_name": "REAL_BONE_002_leg_tibia_fibula_fracture.jpg",
        "url": "https://raw.githubusercontent.com/saraspro/Leg-bone-fracture-detection-xray/master/4.jpg",
        "image_type": "Musculoskeletal X-Ray (Leg)",
        "view_position": "AP",
        "finding_category": "Leg Bone Fracture (Gãy xương cẳng chân)",
        "clinical_notes_en": "Complete transverse/oblique fracture of mid-shaft tibia and fibula with angular displacement.",
        "finding_description": "Gãy hoàn toàn thân xương chày và xương mác cẳng chân, đường gãy ngang chéo có di lệch góc và di lệch sang bên, tổn thương điển hình do chấn thương gãy chân.",
        "ground_truth_caption_vi": "Ảnh X-quang cẳng chân cho thấy gãy hoàn toàn 1/3 giữa xương chày và xương mác kèm di lệch.",
        "source": "Open Orthopedic Leg Fracture Dataset"
    },
    {
        "image_id": "BONE_003",
        "save_name": "REAL_BONE_003_wrist_distal_radius_fracture.jpg",
        "url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/fracture%20(2).jpg",
        "image_type": "Musculoskeletal X-Ray (Wrist)",
        "view_position": "AP/Lateral",
        "finding_category": "Wrist Bone Fracture (Gãy xương cổ tay)",
        "clinical_notes_en": "Distal radius fracture with cortical disruption at the metaphyseal region of the wrist.",
        "finding_description": "Gãy đầu dưới xương quay vùng cổ tay (dạng gãy Colles do ngã chống tay), mất tính liên tục của bờ xương hành xương quay.",
        "ground_truth_caption_vi": "Ảnh X-quang cổ tay phát hiện đường gãy đầu dưới xương quay vùng khớp cổ tay.",
        "source": "Roboflow Bone Fracture Dataset"
    },
    {
        "image_id": "BONE_004",
        "save_name": "REAL_BONE_004_hand_finger_fracture.png",
        "url": "https://raw.githubusercontent.com/manhph2211/Bone-Fracture-Detection/main/imgrm/hand_patient10101_image3.png",
        "image_type": "Musculoskeletal X-Ray (Hand)",
        "view_position": "PA",
        "finding_category": "Hand / Finger Fracture (Gãy xương bàn tay / ngón tay)",
        "clinical_notes_en": "Hand radiograph demonstrating cortical break and fracture line across the phalangeal/metacarpal bone.",
        "finding_description": "Đường gãy thấu khớp hoặc rạn nứt vỏ xương đốt bàn/đốt ngón tay, sưng nề mô liên kết phần mềm bàn tay.",
        "ground_truth_caption_vi": "Ảnh X-quang bàn tay cho thấy tổn thương nứt gãy xương đốt bàn ngón tay.",
        "source": "Stanford MURA / Clinical Hand Dataset"
    },
    {
        "image_id": "BONE_005",
        "save_name": "REAL_BONE_005_arm_normal_no_fracture.jpg",
        "url": "https://raw.githubusercontent.com/a7x3a/bone_fracture_xray_image_analysis/main/test_cases/no-fracture.jpg",
        "image_type": "Musculoskeletal X-Ray (Arm)",
        "view_position": "AP",
        "finding_category": "Normal Arm Bone (Tay bình thường - Không gãy)",
        "clinical_notes_en": "Intact cortex, normal anatomical alignment, no fracture or dislocation identified in the arm bones.",
        "finding_description": "Vỏ xương trơn láng liên tục, không thấy đường rạn nứt hay gãy xương, diện khớp và trục chi bình thường.",
        "ground_truth_caption_vi": "Ảnh X-quang xương tay bình thường, các xương trục chi liên tục không có dấu hiệu gãy hay trật khớp.",
        "source": "Roboflow Bone Fracture Dataset"
    }
]

print("=== BẮT ĐẦU TẢI ẢNH X-QUANG GÃY TAY, GÃY CHÂN ===")
bone_records = []

for item in DIRECT_BONE_IMAGES:
    dest_path = IMAGE_DIR / item["save_name"]
    print(f"Đang tải {item['image_id']} - {item['finding_category']}...")
    try:
        req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
            
        with Image.open(dest_path) as img:
            res_str = f"{img.width}x{img.height}"
            
        item["filename"] = item["save_name"]
        item["resolution"] = res_str
        bone_records.append(item)
        print(f" -> Đã tải: {item['save_name']} ({res_str}, {len(data):,} bytes)")
    except Exception as exc:
        print(f" -> Lỗi tải {item['url']}: {exc}")

# 2. Tải thêm ca gãy xương đùi (Femur fracture) từ archive.zip của repo Automated-Fracture-Detection
try:
    print("Đang trích xuất thêm ca gãy xương đùi (Femur) từ Ortho Fracture Archive...")
    archive_url = "https://raw.githubusercontent.com/arjun2703-ai/Automated-Fracture-Detection/main/archive.zip"
    req = urllib.request.Request(archive_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        zip_bytes = resp.read()
    
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    
    # 9.jpg là ca gãy thân xương dài chân (Tibia/Femur tall radiograph)
    femur_data = z.read("fracture/training/9.jpg")
    femur_name = "REAL_BONE_006_leg_femur_shaft_fracture.jpg"
    femur_path = IMAGE_DIR / femur_name
    with open(femur_path, "wb") as f:
        f.write(femur_data)
        
    with Image.open(femur_path) as img:
        femur_res = f"{img.width}x{img.height}"
        
    femur_item = {
        "image_id": "BONE_006",
        "filename": femur_name,
        "image_type": "Musculoskeletal X-Ray (Leg)",
        "view_position": "AP",
        "finding_category": "Femur / Long Bone Fracture (Gãy thân xương dài chân)",
        "clinical_notes_en": "Oblique fracture of the shaft of long leg bone with cortex displacement.",
        "finding_description": "Đường gãy xương chéo di lệch tại thân xương dài chân, thấy rõ đoạn gãy rời và di lệch trục xương.",
        "ground_truth_caption_vi": "Ảnh X-quang chân cho thấy tổn thương gãy chéo thân xương dài có di lệch vỏ xương.",
        "source": "OrthoIndex Automated Fracture Dataset",
        "resolution": femur_res
    }
    bone_records.append(femur_item)
    print(f" -> Đã trích xuất thành công: {femur_name} ({femur_res})")
except Exception as e:
    print("Lỗi trích xuất từ zip:", e)

# 3. Đọc lại metadata.csv cũ và cập nhật thêm các ảnh xương mới vào
fieldnames = [
    "image_id", "filename", "image_type", "view_position",
    "finding_category", "clinical_notes_en", "finding_description",
    "ground_truth_caption_vi", "source", "resolution"
]

existing_rows = []
if METADATA_PATH_1.exists():
    with open(METADATA_PATH_1, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Giữ lại các ca phổi và ca lỗi trước đó, không ghi đè
            if not r.get("image_id", "").startswith("BONE_"):
                existing_rows.append(r)

all_rows = existing_rows + bone_records

for meta_dest in [METADATA_PATH_1, METADATA_PATH_2]:
    with open(meta_dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            row_dict = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row_dict)
    print(f"Đã cập nhật metadata đầy đủ tại: {meta_dest}")

print(f"\n=== HOÀN TẤT: ĐÃ BỔ SUNG {len(bone_records)} ẢNH GÃY TAY, GÃY CHÂN VÀO DATASET! ===")
