"""
Thu thap du lieu khuon mat theo TUNG NGUOI.

Cach dung:
  1. Chay file, nhap ten nguoi can chup (vd: son, le). Ten da co se duoc dung lai.
  2. Nhan phim 1-8 de chon dieu kien, SPACE de chup.
  3. Nhan R de luu mot anh THAM CHIEU cua nguoi do (data/reference/<ten>/).
  4. Nhan Q de thoat. Muon chup nguoi khac thi chay lai file va nhap ten khac.

Cau truc luu tru:
  data/collected/<ten>/<dieu_kien>/<dieu_kien>_0001.jpg
  data/collected/<ten>/metadata.csv
  data/reference/<ten>/reference_0001.jpg
"""
import cv2
import torch
import numpy as np
import csv
import os
import re
import unicodedata
from datetime import datetime
from PIL import Image
from facenet_pytorch import MTCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLLECTED_ROOT = os.path.join(BASE_DIR, 'data', 'collected')
REFERENCE_ROOT = os.path.join(BASE_DIR, 'data', 'reference')

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

CSV_HEADER = [
    'filename',
    'condition',
    'timestamp',
    'face_width',
    'face_height',
    'brightness',
    'confidence'
]


def slugify(name):
    """'Nguyễn Văn Sơn' -> 'nguyen_van_son' (ten thu muc khong dau, khong khoang trang)."""
    name = name.replace('đ', 'd').replace('Đ', 'D')
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def list_people():
    """Cac nguoi da co trong data/collected (bo qua thu muc kieu cu: normal, left, ...)."""
    if not os.path.isdir(COLLECTED_ROOT):
        return []
    return sorted(
        d for d in os.listdir(COLLECTED_ROOT)
        if os.path.isdir(os.path.join(COLLECTED_ROOT, d)) and d not in conditions.values()
    )


def ask_person():
    people = list_people()
    print("=== THU THAP DU LIEU KHUON MAT ===")
    if people:
        print("Nguoi da co:", ", ".join(people))
    while True:
        raw = input("Nhap ten nguoi can chup (vd: son): ").strip()
        person = slugify(raw)
        if not person:
            print("Ten khong hop le, nhap lai.")
            continue
        if person in conditions.values():
            print("Ten khong duoc trung ten dieu kien:", person)
            continue
        if person != raw:
            print(f"Ten thu muc se la: {person}")
        return person


def next_index(folder, prefix):
    """Chi so lon nhat da co + 1 (khong dem so file) de khong ghi de len anh cu."""
    indexes = []
    for f in os.listdir(folder):
        m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.jpg", f, flags=re.IGNORECASE)
        if m:
            indexes.append(int(m.group(1)))
    return max(indexes, default=0) + 1


def save_jpg(filepath, image):
    """Ghi anh bang Python (ho tro duong dan tieng Viet / OneDrive)."""
    success, encoded_img = cv2.imencode(".jpg", image)
    if not success:
        return False
    try:
        with open(filepath, "wb") as f:
            f.write(encoded_img.tobytes())
        return True
    except Exception as error:
        print("Chi tiet loi:", error)
        return False


def detect_largest_face(frame):
    """Tra ve (anh mat da cat, confidence) cua khuon mat lon nhat, hoac None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        print("No face detected!")
        return None

    # Chon khuon mat lon nhat
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, boxes[best_idx])

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
        return None

    return face, float(probs[best_idx])


# ----------------------------- Chon nguoi -----------------------------
person = ask_person()

DATASET_DIR = os.path.join(COLLECTED_ROOT, person)
REFERENCE_DIR = os.path.join(REFERENCE_ROOT, person)
CSV_PATH = os.path.join(DATASET_DIR, 'metadata.csv')

os.makedirs(REFERENCE_DIR, exist_ok=True)
for condition in conditions.values():
    os.makedirs(os.path.join(DATASET_DIR, condition), exist_ok=True)

# Tao file CSV neu chua ton tai
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(CSV_HEADER)

# So anh da co cua nguoi nay theo tung dieu kien
counts = {}
for condition in conditions.values():
    folder = os.path.join(DATASET_DIR, condition)
    counts[condition] = next_index(folder, condition) - 1

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Khong the mo webcam!")

current_condition = None

print(f"\nDang chup cho: {person}")
print("Nhan phim 1-8 de chon dieu kien")
print("Nhan SPACE de chup va luu anh")
print("Nhan R de luu anh tham chieu")
print("Nhan Q de thoat")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    cv2.putText(display, f"Person: {person}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    if current_condition:
        cv2.putText(display, f"Condition: {current_condition}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {counts[current_condition]}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Select condition: Press 1-8", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Dataset Collection', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Luu anh tham chieu
    elif key == ord('r'):
        result = detect_largest_face(frame)
        if result is None:
            continue
        face, _ = result
        filename = f"reference_{next_index(REFERENCE_DIR, 'reference'):04d}.jpg"
        filepath = os.path.join(REFERENCE_DIR, filename)
        print("Saved reference:" if save_jpg(filepath, face) else "Loi luu anh tham chieu:", filepath)

    # Chon dieu kien
    elif chr(key) in conditions:
        current_condition = conditions[chr(key)]
        print("Selected:", current_condition)

    # Chup anh
    elif key == 32:

        if current_condition is None:
            print("Please select a condition first!")
            continue

        result = detect_largest_face(frame)
        if result is None:
            continue
        face, confidence = result

        folder = os.path.join(DATASET_DIR, current_condition)
        index = next_index(folder, current_condition)
        filename = f"{current_condition}_{index:04d}.jpg"
        filepath = os.path.join(folder, filename)

        if not save_jpg(filepath, face):
            print("Loi: Khong the luu anh!")
            continue

        counts[current_condition] = index

        # Tinh brightness trung binh
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # Luu metadata
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
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
