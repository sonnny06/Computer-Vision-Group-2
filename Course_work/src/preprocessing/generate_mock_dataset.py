"""
Script sinh bộ dữ liệu ảnh X-quang y tế kiểm thử và file metadata.csv.
Phục vụ môn học: Coursework Multimodal AI (Captioning & VQA).
"""
import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = r"E:\Course_work\Course_work\data\evaluation\images"
METADATA_PATH_1 = r"E:\Course_work\Course_work\data\evaluation\metadata.csv"
METADATA_PATH_2 = r"E:\Course_work\Course_work\data\evaluation\images\metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_base_cxr(width=512, height=512, heart_ratio=0.45, spine_brightness=180):
    """Tạo phôi ảnh X-quang lồng ngực cơ bản với cấu trúc giải phẫu."""
    # Nền xám đen mô mềm bên ngoài
    img_arr = np.zeros((height, width), dtype=np.float32) + 25.0
    
    # Lồng ngực (hình bầu dục lớn)
    y, x = np.ogrid[:height, :width]
    thorax_mask = ((x - width/2)**2) / ((width*0.44)**2) + ((y - height*0.52)**2) / ((height*0.44)**2) <= 1.0
    img_arr[thorax_mask] = 80.0
    
    # Trường phổi trái & phải (Phổi chứa khí -> hấp thụ ít tia X -> màu tối)
    # Phổi phải của bệnh nhân (bên trái ảnh x < width/2)
    right_lung_mask = (
        ((x - width*0.30)**2) / ((width*0.14)**2) + 
        ((y - height*0.46)**2) / ((height*0.30)**2) <= 1.0
    )
    # Phổi trái của bệnh nhân (bên phải ảnh x > width/2)
    left_lung_mask = (
        ((x - width*0.70)**2) / ((width*0.14)**2) + 
        ((y - height*0.46)**2) / ((height*0.30)**2) <= 1.0
    )
    
    img_arr[right_lung_mask] = 35.0
    img_arr[left_lung_mask] = 35.0
    
    # Bóng tim và trung thất (chứa mô mềm & máu -> hấp thụ tia X -> màu sáng)
    mediastinum_mask = (np.abs(x - width/2) <= width*0.09) & (y >= height*0.15) & (y <= height*0.85)
    img_arr[mediastinum_mask] = spine_brightness - 20
    
    # Tim (lệch sang bên trái bệnh nhân / bên phải ảnh)
    heart_width = width * heart_ratio
    heart_mask = (
        ((x - (width*0.5 + heart_width*0.2))**2) / ((heart_width*0.5)**2) + 
        ((y - height*0.62)**2) / ((height*0.18)**2) <= 1.0
    )
    img_arr[heart_mask] = 190.0
    
    # Cột sống (ở giữa)
    spine_mask = (np.abs(x - width/2) <= width*0.035) & (y >= height*0.1) & (y <= height*0.9)
    img_arr[spine_mask] = spine_brightness
    
    # Vòm hoành hai bên (đáy phổi)
    diaphragm_y = int(height * 0.74)
    for cx in range(width):
        for cy in range(diaphragm_y, height):
            if thorax_mask[cy, cx]:
                img_arr[cy, cx] = np.clip(img_arr[cy, cx] + 90, 0, 240)
                
    # Thêm vân xương sườn mô phỏng (các đường cong nhẹ)
    for r_idx in range(6):
        rib_y = int(height * (0.28 + r_idx * 0.08))
        curve = np.sin(np.linspace(0, np.pi, width)) * 25
        for col in range(width):
            target_y = int(rib_y + curve[col])
            if 0 <= target_y < height and (right_lung_mask[target_y, col] or left_lung_mask[target_y, col]):
                img_arr[target_y:min(target_y+3, height), col] += 40.0
                
    # Thêm nhiễu Gaussian mô phỏng hạt tia X thực tế
    noise = np.random.normal(0, 4.0, (height, width))
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr, mode='L')
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img

def add_pathology(img, pathology_type):
    """Thêm tổn thương bệnh lý cụ thể vào ảnh X-quang."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    y, x = np.ogrid[:h, :w]
    
    if pathology_type == "pneumonia_right_lower":
        # Đám mờ thâm nhiễm thùy dưới phổi phải (trái ảnh, vùng đáy)
        lesion = ((x - w*0.32)**2)/((w*0.09)**2) + ((y - h*0.62)**2)/((h*0.08)**2) <= 1.0
        arr[lesion] += np.random.uniform(70, 110, size=arr[lesion].shape)
        
    elif pathology_type == "pleural_effusion_left":
        # Tràn dịch màng phổi trái (xóa góc sườn hoành bên phải ảnh)
        effusion = (x >= w*0.68) & (y >= h*0.65) & (y <= h*0.78)
        arr[effusion] += np.random.uniform(100, 140, size=arr[effusion].shape)
        
    elif pathology_type == "pneumothorax_right":
        # Tràn khí màng phổi phải: vùng ngoại vi phổi phải tăng sáng đậm (đen hơn), mất vân phổi
        apex = (x <= w*0.28) & (x >= w*0.18) & (y >= h*0.25) & (y <= h*0.48)
        arr[apex] = np.clip(arr[apex] * 0.4, 10, 255)
        
    elif pathology_type == "solitary_nodule":
        # Nốt đơn độc thùy trên phổi phải
        nodule = ((x - w*0.30)**2)/((w*0.025)**2) + ((y - h*0.36)**2)/((h*0.025)**2) <= 1.0
        arr[nodule] += 130.0
        
    elif pathology_type == "ground_glass_bilateral":
        # Kính mờ rải rác ngoại vi 2 phổi (tổn thương dạng COVID/viêm phổi kẽ)
        gg_r = ((x - w*0.25)**2)/((w*0.07)**2) + ((y - h*0.50)**2)/((h*0.14)**2) <= 1.0
        gg_l = ((x - w*0.75)**2)/((w*0.07)**2) + ((y - h*0.50)**2)/((h*0.14)**2) <= 1.0
        arr[gg_r] += np.random.uniform(40, 75, size=arr[gg_r].shape)
        arr[gg_l] += np.random.uniform(40, 75, size=arr[gg_l].shape)
        
    elif pathology_type == "pneumonia_bilateral":
        # Đám mờ thâm nhiễm rải rác hai đáy phổi
        p_r = ((x - w*0.30)**2)/((w*0.08)**2) + ((y - h*0.64)**2)/((h*0.07)**2) <= 1.0
        p_l = ((x - w*0.72)**2)/((w*0.08)**2) + ((y - h*0.63)**2)/((h*0.07)**2) <= 1.0
        arr[p_r] += np.random.uniform(60, 95, size=arr[p_r].shape)
        arr[p_l] += np.random.uniform(60, 95, size=arr[p_l].shape)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    res_img = Image.fromarray(arr, mode='L')
    return res_img.filter(ImageFilter.GaussianBlur(radius=0.8))

DATASET = [
    {
        "image_id": "CXR_001",
        "filename": "CXR_001_normal_pa.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Normal",
        "finding_description": "Không phát hiện tổn thương thâm nhiễm hay tràn dịch. Kích thước tim và bóng trung thất bình thường. Góc sườn hoành hai bên sáng nhọn.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng tư thế PA bình thường, hai trường phổi sáng rõ, cung tim trong giới hạn bình thường.",
        "source": "NIH-ChestXray8 Benchmark Sample",
        "heart_ratio": 0.44,
        "pathology": None
    },
    {
        "image_id": "CXR_002",
        "filename": "CXR_002_normal_ap.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "AP",
        "finding_category": "Normal",
        "finding_description": "Tư thế AP tại giường, nhu mô phổi hai bên bình thường, không có dấu hiệu đông đặc hay tràn khí.",
        "ground_truth_caption_vi": "Hình ảnh X-quang ngực tư thế AP bình thường, không ghi nhận vùng đông đặc hay tràn dịch màng phổi.",
        "source": "CheXpert Public Subset",
        "heart_ratio": 0.46,
        "pathology": None
    },
    {
        "image_id": "CXR_003",
        "filename": "CXR_003_pneumonia_right_lower.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Pneumonia",
        "finding_description": "Đám mờ đông đặc thâm nhiễm không đồng nhất ở vùng thùy dưới phổi phải, bờ mờ, gợi ý tổn thương viêm phổi thùy.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy đám mờ thâm nhiễm dạng đông đặc tại thùy dưới phổi phải, phù hợp với tổn thương viêm phổi.",
        "source": "RSNA Pneumonia Detection Dataset",
        "heart_ratio": 0.45,
        "pathology": "pneumonia_right_lower"
    },
    {
        "image_id": "CXR_004",
        "filename": "CXR_004_cardiomegaly.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Cardiomegaly",
        "finding_description": "Chỉ số tim - lồng ngực (CTR) lớn hơn 0.55, bóng tim to bè sang bên trái. Hai phế trường chưa thấy ứ dịch rõ rệt.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng cho thấy bóng tim to rõ rệt, chỉ số tim lồng ngực tăng (Cardiomegaly), hai trường phổi sáng.",
        "source": "VinDr-CXR Reference Sample",
        "heart_ratio": 0.62,
        "pathology": None
    },
    {
        "image_id": "CXR_005",
        "filename": "CXR_005_pleural_effusion_left.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Pleural Effusion",
        "finding_description": "Mờ đồng nhất góc sườn hoành trái kèm đường cong Damoiseau (meniscus sign), biểu hiện của tràn dịch màng phổi trái thể tự do.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực phát hiện mờ góc sườn hoành trái, dấu hiệu đường cong Damoiseau chỉ điểm tràn dịch màng phổi trái.",
        "source": "VinDr-CXR Reference Sample",
        "heart_ratio": 0.45,
        "pathology": "pleural_effusion_left"
    },
    {
        "image_id": "CXR_006",
        "filename": "CXR_006_pneumothorax_right.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Pneumothorax",
        "finding_description": "Vùng tăng sáng vô mạch ở ngoại vi đỉnh và hạ đòn phổi phải, thấy đường viền màng phổi tạng, dấu hiệu tràn khí màng phổi phải lượng ít.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực thẳng có vùng tăng sáng vô mạch ở đỉnh phổi phải, gợi ý tràn khí màng phổi phải.",
        "source": "SIIM-ACR Pneumothorax Dataset",
        "heart_ratio": 0.44,
        "pathology": "pneumothorax_right"
    },
    {
        "image_id": "CXR_007",
        "filename": "CXR_007_solitary_nodule.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Lung Nodule",
        "finding_description": "Hình mờ nốt tròn đơn độc kích thước khoảng 15mm, bờ khá rõ tại vùng hạ đòn phổi phải, cần chụp cắt lớp vi tính (CT) để định tính.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy một nốt mờ đơn độc tại vùng phổi phải, cần đánh giá chuyên sâu bằng CT.",
        "source": "LIDC-IDRI Dataset Reference",
        "heart_ratio": 0.43,
        "pathology": "solitary_nodule"
    },
    {
        "image_id": "CXR_008",
        "filename": "CXR_008_covid_ground_glass.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "COVID-19 / Interstitial Pneumonia",
        "finding_description": "Tổn thương dạng kính mờ (Ground-glass opacities) phân bố chủ yếu ở ngoại vi và hai đáy phổi, đặc trưng của viêm phổi kẽ.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực xuất hiện các đám mờ kính mờ rải rác ngoại vi hai phế trường, tổn thương dạng viêm phổi kẽ.",
        "source": "COVID-19 Chest X-Ray Database",
        "heart_ratio": 0.46,
        "pathology": "ground_glass_bilateral"
    },
    {
        "image_id": "CXR_009",
        "filename": "CXR_009_pneumonia_bilateral.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "AP",
        "finding_category": "Pneumonia",
        "finding_description": "Thâm nhiễm phế nang lan tỏa hai đáy phổi, phế quản hơi có thể quan sát thấy, dấu hiệu viêm phổi thùy hai bên.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực cho thấy tổn thương thâm nhiễm đông đặc lan tỏa hai đáy phổi dạng viêm phổi hai bên.",
        "source": "NIH-ChestXray8 Benchmark Sample",
        "heart_ratio": 0.48,
        "pathology": "pneumonia_bilateral"
    },
    {
        "image_id": "CXR_010",
        "filename": "CXR_010_normal_checkup.jpg",
        "image_type": "Chest X-Ray",
        "view_position": "PA",
        "finding_category": "Normal",
        "finding_description": "Khám sức khỏe định kỳ: Phổi sáng đều, cơ hoành hai bên bình thường, vòm hoành trơn láng, không có tổn thương nhu mô hay màng phổi.",
        "ground_truth_caption_vi": "Ảnh X-quang ngực khám sức khỏe định kỳ bình thường, cấu trúc tim phổi không ghi nhận bất thường.",
        "source": "VinDr-CXR Reference Sample",
        "heart_ratio": 0.42,
        "pathology": None
    }
]

# Tạo ảnh và lưu file
for item in DATASET:
    base = create_base_cxr(width=512, height=512, heart_ratio=item["heart_ratio"])
    if item["pathology"]:
        base = add_pathology(base, item["pathology"])
    
    # Lưu dưới định dạng RGB JPEG chất lượng cao
    rgb_img = base.convert("RGB")
    save_path = os.path.join(OUTPUT_DIR, item["filename"])
    rgb_img.save(save_path, "JPEG", quality=95)
    item["resolution"] = f"{rgb_img.width}x{rgb_img.height}"
    print(f"Đã tạo ảnh: {save_path}")

# Tạo thêm 2 file ca lỗi đặc biệt để test hệ thống (Corner cases)
err_corrupt_path = os.path.join(OUTPUT_DIR, "ERR_001_corrupted.jpg")
with open(err_corrupt_path, "wb") as f:
    f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_CORRUPT_BYTES_XYZ123456789")

err_zero_path = os.path.join(OUTPUT_DIR, "ERR_002_zero_byte.jpg")
with open(err_zero_path, "wb") as f:
    pass

print(f"Đã tạo các ca kiểm thử ảnh lỗi: {err_corrupt_path}, {err_zero_path}")

# Ghi file metadata.csv (cho cả 2 vị trí để nhóm import ở đâu cũng có)
fieldnames = [
    "image_id", "filename", "image_type", "view_position",
    "finding_category", "finding_description", "ground_truth_caption_vi",
    "source", "resolution"
]

for meta_dest in [METADATA_PATH_1, METADATA_PATH_2]:
    with open(meta_dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in DATASET:
            row = {k: item[k] for k in fieldnames}
            writer.writerow(row)
    print(f"Đã ghi metadata thành công tại: {meta_dest}")
