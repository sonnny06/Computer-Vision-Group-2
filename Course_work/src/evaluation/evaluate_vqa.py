r"""Đánh giá mô hình trên tập dữ liệu Medical VQA (questions.csv).
Phụ trách: Evaluation & Benchmark

Cách dùng:
    python src/evaluation/evaluate_vqa.py --mock --limit 10
    python src/evaluation/evaluate_vqa.py --mock
    python src/evaluation/evaluate_vqa.py --output results/vqa_eval.csv
"""

import argparse
import csv
import os
import re
import string
import sys
import time
from pathlib import Path

# Đảm bảo import được src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.vqa import answer_question
from src.engine.pipeline import get_default_pipeline, build_pipeline


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản để so khớp đáp án."""
    if not text:
        return ""
    text = text.lower().strip()
    # Loại bỏ dấu câu
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    # Bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Tính F1 overlap giữa prediction và ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    recall = sum(1 for t in gt_tokens if t in common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def is_match(prediction: str, ground_truth: str) -> bool:
    """So khớp đáp án: exact match hoặc containment match."""
    norm_p = normalize_text(prediction)
    norm_g = normalize_text(ground_truth)
    if not norm_p or not norm_g:
        return False
    if norm_p == norm_g:
        return True
    # Containment cho câu trả lời ngắn (Có/Không hoặc từ khóa)
    if norm_g in norm_p.split() or norm_g in norm_p:
        return True
    return False


def run_vqa_evaluation(
    csv_path: Path,
    image_dir: Path,
    use_mock: bool = False,
    limit: int | None = None,
    output_path: Path | None = None,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    pipeline = build_pipeline(use_mock=use_mock)

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_available = len(rows)
    if limit and limit > 0:
        rows = rows[:limit]

    print(f"=== ĐÁNH GIÁ MEDICAL VQA ===")
    print(f"Tập dữ liệu   : {csv_path.name} (Chạy {len(rows)}/{total_available} câu)")
    print(f"Thư mục ảnh   : {image_dir}")
    print(f"Chế độ chạy   : {'MOCK (giả lập)' if use_mock else 'MODEL THẬT'}")
    print("-" * 65)

    results = []
    correct_count = 0
    total_time = 0.0
    success_count = 0

    body_part_stats = {}

    for idx, row in enumerate(rows, start=1):
        q_id = row.get("question_id", f"Q_{idx:04d}")
        img_name = row.get("image_name", "")
        body_part = row.get("body_part", "Chung")
        q_vi = row.get("question_vi", "")
        gt_vi = row.get("answer_vi", "")

        img_path = image_dir / img_name
        if not img_path.exists():
            print(f"[{idx}/{len(rows)}] ⚠️ Không tìm thấy ảnh: {img_name}")
            results.append({
                "question_id": q_id,
                "image_name": img_name,
                "body_part": body_part,
                "question_vi": q_vi,
                "ground_truth": gt_vi,
                "prediction": "",
                "success": False,
                "error": "Ảnh không tồn tại",
                "match": False,
                "f1": 0.0,
                "time_s": 0.0,
            })
            continue

        res = answer_question(str(img_path), q_vi, pipeline=pipeline)
        pred = res.answer if res.success else ""
        time_s = res.meta.get("time_s", 0.0) if res.success else 0.0
        total_time += time_s

        matched = is_match(pred, gt_vi) if res.success else False
        f1 = compute_f1_score(pred, gt_vi) if res.success else 0.0

        if res.success:
            success_count += 1
            if matched:
                correct_count += 1

        if body_part not in body_part_stats:
            body_part_stats[body_part] = {"total": 0, "correct": 0, "f1_sum": 0.0}
        body_part_stats[body_part]["total"] += 1
        if matched:
            body_part_stats[body_part]["correct"] += 1
        body_part_stats[body_part]["f1_sum"] += f1

        results.append({
            "question_id": q_id,
            "image_name": img_name,
            "body_part": body_part,
            "question_vi": q_vi,
            "ground_truth": gt_vi,
            "prediction": pred,
            "success": res.success,
            "error": res.error or "",
            "match": matched,
            "f1": round(f1, 4),
            "time_s": time_s,
        })

        status_mark = "✅" if matched else ("⚠️" if res.success else "❌")
        # In tóm tắt mỗi dòng
        pred_disp = (pred[:45] + "...") if len(pred) > 45 else pred
        print(f"[{idx:03d}/{len(rows):03d}] {status_mark} {q_id} | {img_name} | {q_vi[:35]}...")
        print(f"      GT: {gt_vi} | Pred: {pred_disp}")

    # Tổng kết
    avg_time = (total_time / len(rows)) if rows else 0.0
    accuracy = (correct_count / len(rows) * 100) if rows else 0.0
    overall_f1 = (sum(r["f1"] for r in results) / len(rows)) if rows else 0.0

    print("\n" + "=" * 65)
    print("=== KẾT QUẢ TỔNG HỢP VQA ===")
    print(f"Tổng số câu hỏi đánh giá : {len(rows)}")
    print(f"Thực thi thành công       : {success_count}/{len(rows)}")
    print(f"Độ chính xác (Match)     : {correct_count}/{len(rows)} ({accuracy:.2f}%)")
    print(f"Điểm F1 Token trung bình : {overall_f1:.4f}")
    print(f"Thời gian trung bình/câu : {avg_time:.3f} giây")

    print("\n--- Thống kê theo bộ phận cơ thể ---")
    print(f"{'Bộ phận':<30} | {'Số câu':<8} | {'Chính xác':<12} | {'F1 TB':<8}")
    print("-" * 65)
    for bp, stats in sorted(body_part_stats.items()):
        bp_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0
        bp_f1 = (stats["f1_sum"] / stats["total"]) if stats["total"] else 0.0
        print(f"{bp:<30} | {stats['total']:<8} | {bp_acc:>6.2f}% ({stats['correct']}/{stats['total']}) | {bp_f1:>6.4f}")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "question_id", "image_name", "body_part", "question_vi",
                "ground_truth", "prediction", "success", "error", "match",
                "f1", "time_s"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nĐã lưu kết quả chi tiết ra: {out_p.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Medical VQA trên questions.csv")
    parser.add_argument("--csv", default=str(_ROOT / "data" / "evaluation" / "questions.csv"),
                        help="Đường dẫn file questions.csv")
    parser.add_argument("--image-dir", default=str(_ROOT / "data" / "evaluation" / "images"),
                        help="Thư mục chứa ảnh đánh giá")
    parser.add_argument("--mock", action="store_true", help="Dùng model giả lập")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng câu hỏi cần chạy")
    parser.add_argument("--output", default=None, help="Đường dẫn lưu kết quả CSV")
    args = parser.parse_args()

    use_mock = args.mock or os.getenv("USE_MOCK") == "1"
    run_vqa_evaluation(
        csv_path=Path(args.csv),
        image_dir=Path(args.image_dir),
        use_mock=use_mock,
        limit=args.limit,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
