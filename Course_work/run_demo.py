r"""Chạy thử pipeline từ dòng lệnh.

Ví dụ:
    python run_demo.py --mock
    python run_demo.py --mock --dataset
    python run_demo.py --mock --dataset -n 5
    python run_demo.py data\evaluation\images\phoi_1.jpg -q "Bức ảnh có chứa phổi không?" --mock
"""
import argparse
import csv
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent  # = thư mục Course_work
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMG_DIRS = [
    BASE / "data" / "evaluation" / "images",
    BASE / "src" / "evaluation" / "images",
]
EXTS = {".jpg", ".jpeg", ".png"}


def find_default_image():
    files = sorted(
        p
        for d in IMG_DIRS
        if d.exists()
        for p in d.glob("*")
        if p.suffix.lower() in EXTS and not p.name.startswith("ERR_")
    )
    real = [p for p in files if p.name.startswith("REAL_")]  # ưu tiên ảnh thật
    pick = real or files
    return str(pick[0]) if pick else None


def show(title, result):
    print(f"\n=== {title} ===")
    if result.success:
        print("Trạng thái : THÀNH CÔNG")
        print("Kết quả    :", result.answer)
        print("Thời gian  :", result.meta.get("time_s"), "giây")
    else:
        print("Trạng thái : LỖI")
        print("Mã lỗi     :", result.error_code)
        print("Thông báo  :", result.error)


def run_from_dataset(num_samples: int = 5):
    """Lấy mẫu câu hỏi bàn giao và metadata.csv để chạy thử."""
    from src.inference.captioning import generate_caption
    from src.inference.vqa import answer_question

    q_file = BASE / "data" / "evaluation" / "questions_handover_235.csv"
    meta_file = BASE / "data" / "evaluation" / "metadata.csv"
    img_dir = BASE / "data" / "evaluation" / "images"

    if not q_file.exists() or not img_dir.exists():
        sys.exit(f"Không tìm thấy tập dữ liệu tại: {q_file}")

    # Đọc metadata làm đối chiếu
    metadata_map = {}
    if meta_file.exists():
        with meta_file.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                metadata_map[r.get("filename")] = r

    # Đọc câu hỏi
    questions = []
    with q_file.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            questions.append(r)

    # Chọn mẫu đa dạng theo các bộ phận cơ thể khác nhau
    sampled = []
    seen_parts = set()
    for q in questions:
        bp = q.get("body_part")
        if bp not in seen_parts:
            sampled.append(q)
            seen_parts.add(bp)
        if len(sampled) >= num_samples:
            break
    if len(sampled) < num_samples:
        for q in questions:
            if q not in sampled:
                sampled.append(q)
            if len(sampled) >= num_samples:
                break

    print(f"\n=======================================================")
    print(f"  CHẠY THỬ {len(sampled)} MẪU TỪ BỘ DỮ LIỆU QUESTIONS.CSV")
    print(f"=======================================================")

    for idx, q in enumerate(sampled, start=1):
        img_name = q.get("image_name")
        img_path = img_dir / img_name
        bp = q.get("body_part", "Chung")
        q_vi = q.get("question_vi", "")
        a_vi = q.get("answer_vi", "")

        print(f"\n[{idx}/{len(sampled)}] Bộ phận: {bp} | Ảnh: {img_name}")
        print(f"  Câu hỏi (VQA)   : {q_vi}")
        print(f"  Đáp án chuẩn GT : {a_vi}")

        # Thử nghiệm VQA
        res_vqa = answer_question(str(img_path), q_vi)
        if res_vqa.success:
            print(f"  -> Model VQA    : {res_vqa.answer} ({res_vqa.meta.get('time_s', 0)}s)")
        else:
            print(f"  -> Model VQA LỖI: [{res_vqa.error_code}] {res_vqa.error}")

        # Thử nghiệm Captioning cho ảnh đó nếu có metadata
        meta = metadata_map.get(img_name)
        if meta and meta.get("ground_truth_caption_vi"):
            gt_cap = meta["ground_truth_caption_vi"]
            res_cap = generate_caption(str(img_path))
            print(f"  Mô tả chuẩn GT  : {gt_cap[:70]}...")
            if res_cap.success:
                print(f"  -> Model Caption: {res_cap.answer[:70]}... ({res_cap.meta.get('time_s', 0)}s)")


def main():
    parser = argparse.ArgumentParser(description="Chạy thử pipeline Medical VQA")
    parser.add_argument("image", nargs="?", help="Đường dẫn ảnh (bỏ trống để tự tìm)")
    parser.add_argument("-q", "--question",
                        default="Trong ảnh có tổn thương không? Nằm ở đâu?")
    parser.add_argument("--mock", action="store_true",
                        help="Dùng model giả (không cần GPU)")
    parser.add_argument("--dataset", action="store_true",
                        help="Chạy thử mẫu từ bộ 235 câu bàn giao và metadata.csv")
    parser.add_argument("-n", "--num-samples", type=int, default=5,
                        help="Số lượng mẫu chạy khi dùng --dataset (mặc định 5)")
    parser.add_argument("--no-error-tests", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["USE_MOCK"] = "1"

    if args.dataset:
        print("Chế độ:", "MOCK (model giả)" if args.mock else "MODEL THẬT")
        run_from_dataset(num_samples=args.num_samples)
        return

    from src.inference.captioning import generate_caption
    from src.inference.vqa import answer_question

    image = args.image or find_default_image()
    if not image:
        sys.exit("Chưa có ảnh trong: " + ", ".join(str(d) for d in IMG_DIRS))

    print("Ảnh  :", image)
    print("Chế độ:", "MOCK (model giả)" if args.mock else "MODEL THẬT")

    show("CAPTIONING", generate_caption(image))
    show(f"VQA: {args.question}", answer_question(image, args.question))

    if not args.no_error_tests:
        print("\n----- Kiểm tra xử lý lỗi (kỳ vọng đều báo LỖI) -----")
        show("Câu hỏi rỗng", answer_question(image, "   "))
        show("Ảnh không tồn tại", generate_caption(str(BASE / "khong_ton_tai.png")))
        show("Không có ảnh", generate_caption(None))


if __name__ == "__main__":
    main()
