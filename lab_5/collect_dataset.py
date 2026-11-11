
import cv2
import torch
import numpy as np
import csv
import os
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, 'data', 'collected')
CSV_PATH = os.path.join(DATASET_DIR, 'metadata.csv')

os.makedirs(DATASET_DIR, exist_ok=True)

conditions = {
    '1': 'normal',
    '2': 'low_light',
    '3': 'bright',
    '4': 'near',
    '5': 'far',
    '6': 'left',
    '7': 'right',
    '8': 'expression'
}

os.makedirs(DATASET_DIR, exist_ok=True)

for condition in conditions.values():
    os.makedirs(os.path.join(DATASET_DIR, condition), exist_ok=True)

# Tao file CSV neu chua ton tai
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename',
            'condition',
            'timestamp',
            'face_width',
            'face_height',
            'brightness',
            'confidence'
        ])

# Dem so anh da co
counts = {}

for condition in conditions.values():
    folder = os.path.join(DATASET_DIR, condition)
    counts[condition] = len([
        f for f in os.listdir(folder)
        if f.lower().endswith('.jpg')
    ])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Khong the mo webcam!")

current_condition = None

print("=== FACE DATASET COLLECTION ===")
print("Nhan phim 1-8 de chon dieu kien")
print("Nhan SPACE de chup va luu anh")
print("Nhan Q de thoat")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    if current_condition:
        cv2.putText(
            display,
            f"Condition: {current_condition}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Saved: {counts[current_condition]}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            display,
            "Select condition: Press 1-8",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.imshow('Dataset Collection', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Chon dieu kien
    elif chr(key) in conditions:
        current_condition = conditions[chr(key)]
        print("Selected:", current_condition)

    # Chup anh
    elif key == 32:

        if current_condition is None:
            print("Please select a condition first!")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        boxes, probs = mtcnn.detect(img)

        if boxes is None:
            print("No face detected!")
            continue

        # Chon khuon mat lon nhat
        valid_boxes = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            valid_boxes.append((area, i, box))

        _, best_idx, box = max(valid_boxes, key=lambda x: x[0])

        x1, y1, x2, y2 = map(int, box)

        # Them margin 10%
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame.shape[1], x2 + margin_x)
        y2 = min(frame.shape[0], y2 + margin_y)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            print("Invalid face crop!")
            continue

        # Tao ten file
        counts[current_condition] += 1

        filename = f"{current_condition}_{counts[current_condition]:04d}.jpg"

        filepath = os.path.join(
            DATASET_DIR,
            current_condition,
            filename
        )

                # Kiểm tra ảnh đã được lưu thành công chưa
        print("Đường dẫn lưu:", os.path.abspath(filepath))
        print("Thư mục tồn tại:", os.path.isdir(os.path.dirname(filepath)))
        print("Kích thước ảnh:", face.shape)
        print("Kiểu dữ liệu ảnh:", face.dtype)

        # Mã hóa ảnh thành định dạng JPG
        success, encoded_img = cv2.imencode(".jpg", face)

        # Sử dụng Python để ghi file, hỗ trợ đường dẫn tiếng Việt
        saved = False

        if success:
            try:
                with open(filepath, "wb") as f:
                    f.write(encoded_img.tobytes())
                saved = True
            except Exception as error:
                print("Chi tiết lỗi:", error)

        print("Kết quả lưu:", saved)

        if not saved:
            counts[current_condition] -= 1
            print("Lỗi: Không thể lưu ảnh!")
            continue

        # Tính brightness trung bình
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        confidence = float(probs[best_idx])

        # Lưu metadata
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow([
                filename,
                current_condition,
                datetime.now().isoformat(timespec='seconds'),
                face.shape[1],
                face.shape[0],
                round(brightness, 2),
                round(confidence, 4)
            ])

        print(f"Saved: {filepath}")

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed!")
