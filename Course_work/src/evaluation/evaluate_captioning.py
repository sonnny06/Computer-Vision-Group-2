r"""Đánh giá mô hình trên tập dữ liệu Medical Captioning (metadata.csv).
Phụ trách: Evaluation & Benchmark

Cách dùng:
    python src/evaluation/evaluate_captioning.py --mock --limit 10
    python src/evaluation/evaluate_captioning.py --mock
    python src/evaluation/evaluate_captioning.py --output results/caption_eval.csv
"""

import argparse
import csv
import os
import re
import string
import sys
from pathlib import Path

# Đảm bảo import được src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.captioning import generate_caption
from src.engine.pipeline import build_pipeline


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    for ch in string.punctuation:
        text = text.replace(ch, " ")
    return re.sub(r"\s+", " ", text).strip()


def compute_f1_score(prediction: str, ground_truth: str) -> float:
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


def compute_bleu1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in gt_tokens)
    return common / len(pred_tokens)


def run_caption_evaluation(
    csv_path: Path,
    image_dir: Path,
    use_mock: bool = False,
    limit: int | None = None,
    output_path: Path | None = None,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata: {csv_path}")
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

    print(f"=== ĐÁNH GIÁ MEDICAL IMAGE CAPTIONING ===")
    print(f"Tập dữ liệu   : {csv_path.name} (Chạy {len(rows)}/{total_available} ca bệnh)")
    print(f"Thư mục ảnh   : {image_dir}")
    print(f"Chế độ chạy   : {'MOCK (giả lập)' if use_mock else 'MODEL THẬT'}")
    print("-" * 70)

    results = []
    total_time = 0.0
    success_count = 0
    f1_sum = 0.0
    bleu_sum = 0.0

    body_part_stats = {}

    for idx, row in enumerate(rows, start=1):
        img_id = row.get("image_id", f"IMG_{idx:03d}")
        filename = row.get("filename", "")
        body_part = row.get("body_part", "Chung")
        gt_caption = row.get("ground_truth_caption_vi") or row.get("finding_description", "")

        img_path = image_dir / filename
        if not img_path.exists():
            print(f"[{idx}/{len(rows)}] ⚠️ Không tìm thấy ảnh: {filename}")
            results.append({
                "image_id": img_id,
                "filename": filename,
                "body_part": body_part,
                "ground_truth": gt_caption,
                "prediction": "",
                "success": False,
                "error": "Ảnh không tồn tại",
                "bleu1": 0.0,
                "f1": 0.0,
                "time_s": 0.0,
            })
            continue

        res = generate_caption(str(img_path), pipeline=pipeline)
        pred = res.answer if res.success else ""
        time_s = res.meta.get("time_s", 0.0) if res.success else 0.0
        total_time += time_s

        f1 = compute_f1_score(pred, gt_caption) if res.success else 0.0
        bleu1 = compute_bleu1(pred, gt_caption) if res.success else 0.0

        if res.success:
            success_count += 1
            f1_sum += f1
            bleu_sum += bleu1

        if body_part not in body_part_stats:
            body_part_stats[body_part] = {"total": 0, "f1_sum": 0.0, "bleu_sum": 0.0}
        body_part_stats[body_part]["total"] += 1
        body_part_stats[body_part]["f1_sum"] += f1
        body_part_stats[body_part]["bleu_sum"] += bleu1

        results.append({
            "image_id": img_id,
            "filename": filename,
            "body_part": body_part,
            "ground_truth": gt_caption,
            "prediction": pred,
            "success": res.success,
            "error": res.error or "",
            "bleu1": round(bleu1, 4),
            "f1": round(f1, 4),
            "time_s": time_s,
        })

        status_mark = "✅" if res.success else "❌"
        print(f"[{idx:02d}/{len(rows):02d}] {status_mark} {img_id} ({filename}) | {body_part}")
        print(f"      GT  : {gt_caption[:65]}...")
        print(f"      Pred: {pred[:65]}...")

    avg_time = (total_time / len(rows)) if rows else 0.0
    avg_f1 = (f1_sum / len(rows)) if rows else 0.0
    avg_bleu = (bleu_sum / len(rows)) if rows else 0.0

    print("\n" + "=" * 70)
    print("=== KẾT QUẢ TỔNG HỢP CAPTIONING ===")
    print(f"Tổng số ca bệnh đánh giá : {len(rows)}")
    print(f"Thực thi thành công      : {success_count}/{len(rows)}")
    print(f"Điểm BLEU-1 trung bình   : {avg_bleu:.4f}")
    print(f"Điểm F1 Token trung bình : {avg_f1:.4f}")
    print(f"Thời gian trung bình/ca  : {avg_time:.3f} giây")

    print("\n--- Thống kê theo bộ phận cơ thể ---")
    print(f"{'Bộ phận':<30} | {'Số ca':<6} | {'BLEU-1':<8} | {'F1 TB':<8}")
    print("-" * 70)
    for bp, stats in sorted(body_part_stats.items()):
        bp_bleu = (stats["bleu_sum"] / stats["total"]) if stats["total"] else 0.0
        bp_f1 = (stats["f1_sum"] / stats["total"]) if stats["total"] else 0.0
        print(f"{bp:<30} | {stats['total']:<6} | {bp_bleu:>6.4f} | {bp_f1:>6.4f}")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "image_id", "filename", "body_part", "ground_truth",
                "prediction", "success", "error", "bleu1", "f1", "time_s"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nĐã lưu kết quả chi tiết ra: {out_p.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Medical Image Captioning trên metadata.csv")
    parser.add_argument("--csv", default=str(_ROOT / "data" / "evaluation" / "metadata.csv"),
                        help="Đường dẫn file metadata.csv")
    parser.add_argument("--image-dir", default=str(_ROOT / "data" / "evaluation" / "images"),
                        help="Thư mục chứa ảnh đánh giá")
    parser.add_argument("--mock", action="store_true", help="Dùng model giả lập")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ca cần chạy")
    parser.add_argument("--output", default=None, help="Đường dẫn lưu kết quả CSV")
    args = parser.parse_args()

    use_mock = args.mock or os.getenv("USE_MOCK") == "1"
    run_caption_evaluation(
        csv_path=Path(args.csv),
        image_dir=Path(args.image_dir),
        use_mock=use_mock,
        limit=args.limit,
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
