r"""Đánh giá mô hình trên tập dữ liệu Medical Captioning (metadata.csv).
Phụ trách: Evaluation & Benchmark

Cách dùng:
    python src/evaluation/evaluate_captioning.py --mock --limit 10
    python src/evaluation/evaluate_captioning.py --mock
    python src/evaluation/evaluate_captioning.py --output results/caption_eval.csv
    python src/evaluation/evaluate_captioning.py --summarize-reviews refs.csv predictions.csv rubric.csv
"""

import argparse
import csv
import math
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

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


_REVIEW_GROUPS = ("modality", "region_group", "pilot_overlap")
_MATCH_FIELDS = ("modality_match", "body_region_match", "core_structure_1_match", "core_structure_2_match")


def _read_by_image_id(path: Path, required_columns: tuple[str, ...]):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected = {"image_id", *required_columns}
        missing_columns = expected - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path}: thiếu cột {', '.join(sorted(missing_columns))}")

        rows = {}
        for row in reader:
            image_id = (row["image_id"] or "").strip()
            if not image_id:
                raise ValueError(f"{path}:{reader.line_num}: image_id trống")
            if image_id in rows:
                raise ValueError(f"{path}:{reader.line_num}: trùng image_id {image_id}")
            if any(row[column] is None for column in expected):
                raise ValueError(f"{path}:{reader.line_num}: dòng {image_id} thiếu trường CSV")
            rows[image_id] = row
    return rows


def _review_group_summary(field, value, rows):
    successful = [(review, elapsed) for _, review, elapsed in rows if elapsed is not None]
    reviews = [review for review, _ in successful]
    times = [elapsed for _, elapsed in successful]
    coverage = Counter(review["main_coverage"] for review in reviews)
    language = Counter(review["vietnamese_clarity"] for review in reviews)
    return dict(
        group_by=field, group=value, total_images=len(rows), successful=len(successful),
        execution_error=len(rows) - len(successful), main_complete=coverage["complete"],
        main_partial=coverage["partial"], main_none=coverage["none"],
        **{key: sum(review[key] == "1" for review in reviews) for key in _MATCH_FIELDS},
        salient_feature_applicable=sum(review["required_feature_match"] in {"0", "1"} for review in reviews),
        salient_feature_match=sum(review["required_feature_match"] == "1" for review in reviews),
        contains_contradiction=sum(bool(review["contradicts_evidence"].strip()) for review in reviews),
        contains_unverified_claim=sum(bool(review["needs_verification"].strip()) for review in reviews),
        vietnamese_clear=language["clear"], vietnamese_needs_edit=language["needs_edit"],
        mean_time_s=round(mean(times), 3) if times else "",
        median_time_s=round(median(times), 3) if times else "",
        min_time_s=round(min(times), 3) if times else "",
        max_time_s=round(max(times), 3) if times else "",
    )


def summarize_caption_reviews(references_path: Path, predictions_path: Path, rubric_path: Path):
    references = _read_by_image_id(references_path, _REVIEW_GROUPS)
    predictions = _read_by_image_id(predictions_path, ("success", "raw_caption", "time_s"))
    rubric = _read_by_image_id(rubric_path, (*_MATCH_FIELDS, "required_feature_match", "main_coverage",
                                             "contradicts_evidence", "needs_verification", "vietnamese_clarity"))
    if not references:
        raise ValueError(f"{references_path}: không có image_id để tổng hợp")
    for path, rows in ((predictions_path, predictions), (rubric_path, rubric)):
        missing = ", ".join(sorted(references.keys() - rows.keys()))
        extra = ", ".join(sorted(rows.keys() - references.keys()))
        if missing or extra:
            details = [f"thiếu image_id {missing}" if missing else "", f"dư image_id {extra}" if extra else ""]
            raise ValueError(f"{path}: {' / '.join(filter(None, details))}")

    joined = []
    for image_id, reference in references.items():
        prediction = predictions[image_id]
        review = rubric[image_id]
        success = prediction["success"].strip().lower()
        if success not in {"true", "false"}:
            raise ValueError(f"{predictions_path}: {image_id} có giá trị success không hợp lệ")

        elapsed = None
        if success == "true":
            if not prediction["raw_caption"].strip():
                raise ValueError(f"{predictions_path}: {image_id} thành công nhưng raw_caption trống")
            try:
                elapsed = float(prediction["time_s"])
            except (ValueError, TypeError):
                raise ValueError(f"{predictions_path}: {image_id} có time_s không hợp lệ") from None
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"{predictions_path}: {image_id} có time_s không hợp lệ")
            if review["main_coverage"] not in {"complete", "partial", "none"}:
                raise ValueError(f"{rubric_path}: {image_id} có main_coverage không hợp lệ")
            if review["vietnamese_clarity"] not in {"clear", "needs_edit"}:
                raise ValueError(f"{rubric_path}: {image_id} có vietnamese_clarity không hợp lệ")

        joined.append((reference, review, elapsed))

    summary = [_review_group_summary("all", f"{len(joined)} ảnh", joined)]
    for field in _REVIEW_GROUPS:
        groups = {}
        for row in joined:
            groups.setdefault(row[0][field], []).append(row)
        summary.extend(_review_group_summary(field, value, rows) for value, rows in sorted(groups.items()))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Medical Image Captioning hoặc tổng hợp phiếu đã chấm")
    parser.add_argument("--csv", type=Path, help="Đường dẫn file metadata.csv")
    parser.add_argument("--image-dir", type=Path, help="Thư mục chứa ảnh đánh giá")
    parser.add_argument("--mock", action="store_true", help="Dùng model giả lập")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số ca cần chạy")
    parser.add_argument("--output", type=Path, help="Đường dẫn lưu kết quả CSV")
    parser.add_argument("--summarize-reviews", nargs=3, type=Path,
                        metavar=("REFERENCES", "PREDICTIONS", "RUBRIC"),
                        help="Tổng hợp phiếu chấm đã lưu theo image_id")
    args = parser.parse_args()

    if args.summarize_reviews:
        if args.csv or args.image_dir or args.mock or args.limit is not None:
            parser.error("--summarize-reviews dùng với --output hoặc tự in tổng hợp")
        try:
            summary = summarize_caption_reviews(*args.summarize_reviews)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        return

    use_mock = args.mock or os.getenv("USE_MOCK") == "1"
    data_dir = _ROOT / "data" / "evaluation"
    run_caption_evaluation(
        csv_path=args.csv or data_dir / "metadata.csv",
        image_dir=args.image_dir or data_dir / "images",
        use_mock=use_mock,
        limit=args.limit,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
