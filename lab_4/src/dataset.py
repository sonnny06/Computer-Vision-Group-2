"""
Chuẩn bị dữ liệu cho bài toán image similarity.
Vị trí trong project: lab_4/src/dataset.py
Ảnh đầu vào: lab_4/data/input/
Đầu ra: lab_4/data/output/

Chạy bằng notebook 01_dataset_preparation.ipynb hoặc chạy file này.
Nhãn không được tự suy ra từ tên file: 1 = tương tự, 0 = không tương tự.
"""
from pathlib import Path
from itertools import combinations
import csv
import hashlib

try:
    from PIL import Image, ImageFile
except ImportError as exc:
    raise SystemExit("Thiếu Pillow. Cài bằng: pip install pillow") from exc

ImageFile.LOAD_TRUNCATED_IMAGES = False
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"
INVENTORY_CSV = OUTPUT_DIR / "image_inventory.csv"
PAIRS_CSV = OUTPUT_DIR / "pairs.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_images():
    """Kiểm tra ảnh, lưu metadata và liệt kê ảnh lỗi/trùng khớp hoàn toàn."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {INPUT_DIR}")

    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    valid, invalid = [], []
    hashes = {}

    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height, mode = image.width, image.height, image.mode
            checksum = sha256_file(path)
            record = {
                "filename": path.name,
                "path": path.relative_to(ROOT).as_posix(),
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": checksum,
            }
            valid.append(record)
            hashes.setdefault(checksum, []).append(path.name)
        except Exception as exc:
            invalid.append({"filename": path.name, "error": str(exc)})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["filename", "path", "width", "height", "mode", "sha256"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(valid)

    duplicates = [names for names in hashes.values() if len(names) > 1]
    print(f"Tệp ảnh có định dạng hỗ trợ: {len(image_paths)}")
    print(f"Ảnh đọc được: {len(valid)} | Ảnh lỗi: {len(invalid)}")
    print(f"Đã lưu danh sách ảnh: {INVENTORY_CSV.relative_to(ROOT)}")

    if duplicates:
        print("Các nhóm tệp có nội dung giống hệt nhau (SHA-256):")
        for names in duplicates:
            print(" - " + ", ".join(names))
    else:
        print("Không phát hiện tệp trùng khớp byte.")
    if invalid:
        print("Ảnh không đọc được:")
        for item in invalid:
            print(f" - {item['filename']}: {item['error']}")
    return valid, invalid


def create_pair_template(valid_images):
    """
    Tạo tất cả cặp không thứ tự.
    Giữ lại nhãn/ghi chú đã nhập trong pairs.csv khi chạy lại.
    Nhãn của cặp mới để trống để người đánh giá điền thủ công.
    """
    old_labels = {}
    if PAIRS_CSV.exists():
        with PAIRS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                key = (row.get("image1", ""), row.get("image2", ""))
                reverse_key = (key[1], key[0])
                old_labels[key] = (row.get("label", ""), row.get("notes", ""))
                old_labels[reverse_key] = (row.get("label", ""), row.get("notes", ""))

    rows = []
    for left, right in combinations(valid_images, 2):
        image1, image2 = left["path"], right["path"]
        label, notes = old_labels.get((image1, image2), ("", ""))
        rows.append({
            "image1": image1,
            "image2": image2,
            "label": label,
            "notes": notes,
        })

    with PAIRS_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["image1", "image2", "label", "notes"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã tạo/cập nhật {len(rows)} cặp: {PAIRS_CSV.relative_to(ROOT)}")
    print("Điền label=1 (tương tự) hoặc label=0 (không tương tự).")
    print("Nhãn và ghi chú cũ được giữ lại khi chạy lại chương trình.")
    return rows


def validate_labels():
    """Kiểm tra nhãn đã nhập trong pairs.csv."""
    if not PAIRS_CSV.exists():
        raise FileNotFoundError("Chưa có pairs.csv. Hãy chạy inspect_images() và create_pair_template() trước.")

    with PAIRS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    missing = [r for r in rows if (r.get("label") or "").strip() == ""]
    invalid = [r for r in rows if (r.get("label") or "").strip() not in {"", "0", "1"}]
    positives = sum((r.get("label") or "").strip() == "1" for r in rows)
    negatives = sum((r.get("label") or "").strip() == "0" for r in rows)

    print(f"Tổng số cặp: {len(rows)}")
    print(f"Nhãn 1: {positives} | Nhãn 0: {negatives}")
    print(f"Cặp còn thiếu nhãn: {len(missing)} | Nhãn không hợp lệ: {len(invalid)}")
    if missing or invalid:
        print("CHƯA SẴN SÀNG đánh giá. Hoàn tất nhãn 0/1 cho mọi cặp trước.")
    elif not positives or not negatives:
        print("Cảnh báo: cần kiểm tra để bảo đảm tập nhãn có cả lớp 0 và lớp 1.")
    else:
        print("Nhãn đã hợp lệ về định dạng; tiếp theo cần kiểm tra tính nhất quán của nhãn.")
    return {"total": len(rows), "positive": positives, "negative": negatives,
            "missing": len(missing), "invalid": len(invalid)}


def prepare_dataset():
    valid, invalid = inspect_images()
    if len(valid) < 2:
        raise ValueError("Cần ít nhất 2 ảnh hợp lệ trong data/input/ để tạo cặp.")
    create_pair_template(valid)
    return {"valid_images": len(valid), "invalid_images": len(invalid)}


if __name__ == "__main__":
    prepare_dataset()
