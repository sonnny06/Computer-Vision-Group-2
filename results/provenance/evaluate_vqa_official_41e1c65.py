r"""Đánh giá mô hình trên tập dữ liệu Medical VQA (questions.csv).
Phụ trách: Evaluation & Benchmark

Cách dùng:
    python src/evaluation/evaluate_vqa.py --mock --limit 10
    python src/evaluation/evaluate_vqa.py --mock
    python src/evaluation/evaluate_vqa.py --output results/vqa_eval.csv
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import string
import sys
import time
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path

# Đảm bảo import được src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.vqa import answer_question
from src.engine.pipeline import build_pipeline


CLOSED_LABELS = {"có": "Có", "không": "Không", "không đủ thông tin": "Không đủ thông tin"}
CHEST_PA_IMAGES = {"phoi_3.jpg", "phoi_4.jpg", "phoi_5.jpg"}
RESULT_FIELDS = (
    "question_id", "image_name", "body_part", "question_vi", "ground_truth",
    "prediction", "success", "error", "match", "f1", "time_s",
    "answer_type", "evaluation_role", "raw_answer", "answer", "ground_truth_label",
    "status", "score_note", "error_code",
    "is_chest", "is_cxr", "projection",
)

_EXPLICIT_LABEL = re.compile(r'(?:câu trả lời(?: là|:)\s+)?(có|không)(?:$|\s*[:,–—]\s*)')
_INSUFFICIENT = re.compile(
    r"(?:ảnh này |ảnh |hình ảnh này |hình ảnh )?"
    r"(?:không (?:có(?: đủ)?|đủ|chứa) thông tin|không đủ (?:rõ|dữ liệu|căn cứ)|"
    r"không thể (?:xác định|kết luận|trả lời|biết|suy ra)|"
    r"không xác định được|"
    r"không (?:cho biết|cung cấp thông tin)|chưa đủ (?:thông tin|dữ liệu|căn cứ))\b"
)
_NEGATIVE = re.compile(
    r"(?:đây |(?:hình )?ảnh này )?không (?:có|phải|thấy)\b"
    r"(?!\s+(?:vẻ|khả năng|rõ)\b)"
)


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


def normalize_closed_answer(raw_answer: str) -> str | None:
    """Lấy nhãn từ phần trả lời trực tiếp. Câu mơ hồ cần chấm tay."""
    text = " ".join(unicodedata.normalize("NFC", raw_answer).casefold().strip().split())
    if not text:
        return None
    # Qwen có thể in đậm nhãn ở đầu phần trả lời.
    text = re.sub(r"\*\*(có|không|không đủ thông tin)\*\*", r"\1", text)
    lead, *explanations = re.split(r"(?<=[.!?;])\s+|,?\s+nhưng\s+", text)
    if "?" in lead or re.search(
        r"\b(?:có|không)\s*(?:/|hoặc|hay|và)\s*(?:có|không)\b", lead
    ):
        return None
    lead = lead.rstrip(".!; ")
    explicit = _EXPLICIT_LABEL.match(lead)
    if _INSUFFICIENT.match(lead):
        label = CLOSED_LABELS["không đủ thông tin"]
    elif explicit:
        label = CLOSED_LABELS[explicit[1]]
        tail = lead[explicit.end():]
        if _INSUFFICIENT.match(tail) or re.match(r"(?:nếu|chưa chắc)\b", tail):
            return None
        repeated = _EXPLICIT_LABEL.match(tail)
        if repeated and CLOSED_LABELS[repeated[1]] != label:
            return None
        if label == CLOSED_LABELS["có"] and _NEGATIVE.fullmatch(tail):
            return None
    elif _NEGATIVE.match(lead):
        label = CLOSED_LABELS["không"]
    else:
        return None

    for clause in explanations:
        # Chỉ xét nhãn được nói rõ lần nữa. Các nhận định thêm cần đối chiếu với ảnh.
        if clause.endswith("?"):
            continue
        clause = clause.rstrip(".!; ")
        repeated = _EXPLICIT_LABEL.match(clause)
        if repeated and CLOSED_LABELS[repeated[1]] != label:
            return None
        if _INSUFFICIENT.match(clause) and label != CLOSED_LABELS["không đủ thông tin"]:
            return None
    return label


def _print_group(label, rows):
    counts = Counter(row["status"] for row in rows)
    scored = counts["correct"] + counts["incorrect"]
    score = f'{counts["correct"]}/{scored} đã chấm' if scored else "chưa có điểm"
    print(
        f"{label:<30} | {len(rows):>3} câu | {score} | "
        f'Chờ chấm tay: {counts["pending_manual"]} | Cần xem lại: {counts["needs_review"]} | '
        f'Lỗi: {counts["error"]}'
    )


def _run_identity(csv_path, image_dir, use_mock):
    return json.dumps({
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_dir": str(image_dir.resolve()),
        "use_mock": use_mock,
    }, sort_keys=True)


def _checkpoint_rows(db, identity):
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS run_info (id INTEGER PRIMARY KEY CHECK(id = 1), identity TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS completed (question_id TEXT PRIMARY KEY, result TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO run_info (id, identity) VALUES (1, ?)", (identity,))
        saved_identity, = db.execute("SELECT identity FROM run_info WHERE id = 1").fetchone()
        if saved_identity != identity:
            raise ValueError("Checkpoint không khớp CSV, mã evaluator, thư mục ảnh hoặc chế độ chạy")
    return {qid: json.loads(result) for qid, result in db.execute("SELECT question_id, result FROM completed")}


def _evaluate_rows(rows, image_dir, use_mock, saved, checkpoint=None):
    pipeline = None
    if any(qid not in saved for qid, _ in rows):
        init_start = time.perf_counter()
        pipeline = build_pipeline(use_mock=use_mock)
        print(f"Khởi tạo pipeline trước khi chạy các câu: {time.perf_counter() - init_start:.3f} giây")

    results = []
    for idx, (q_id, row) in enumerate(rows, start=1):
        if q_id in saved:
            results.append(saved[q_id])
            continue
        result = _evaluate_row(q_id, row, image_dir, pipeline)
        if checkpoint is not None:
            with checkpoint:
                checkpoint.execute("INSERT INTO completed VALUES (?, ?)", (
                    q_id, json.dumps(result, ensure_ascii=False),
                ))
        results.append(result)
        print(f'[{idx:03d}/{len(rows):03d}] {q_id} | {result["image_name"]} | '
              f'{result["evaluation_role"]} | {result["status"]}')
        print(f'      Tham chiếu: {result["ground_truth"]} | Đầu ra: {(result["raw_answer"] or "")[:80]}')
    return results


def run_vqa_evaluation(
    csv_path: Path,
    image_dir: Path,
    use_mock: bool = False,
    limit: int | None = None,
    output_path: Path | None = None,
    *,
    checkpoint_path: Path | None = None,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    total_available = len(rows)
    if limit and limit > 0:
        rows = rows[:limit]

    print("=== ĐÁNH GIÁ MEDICAL VQA ===")
    print(f"Tập dữ liệu   : {csv_path.name} (Chạy {len(rows)}/{total_available} câu)")
    print(f"Thư mục ảnh   : {image_dir}")
    print(f"Chế độ chạy   : {'MOCK (giả lập)' if use_mock else 'MODEL THẬT'}")
    print("-" * 65)

    indexed = [(row.get("question_id") or f"Q_{idx:04d}", row)
               for idx, row in enumerate(rows, start=1)]
    if checkpoint_path is None:
        results = _evaluate_rows(indexed, image_dir, use_mock, {})
    else:
        if any(not row.get("question_id") for row in rows) or len({qid for qid, _ in indexed}) != len(rows):
            raise ValueError("Checkpoint yêu cầu question_id riêng cho từng dòng")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(checkpoint_path)) as checkpoint:
            saved = _checkpoint_rows(checkpoint, _run_identity(csv_path, image_dir, use_mock))
            print(f"Số câu đã lưu trong checkpoint: {sum(qid in saved for qid, _ in indexed)}")
            results = _evaluate_rows(indexed, image_dir, use_mock, saved, checkpoint)

    main_rows = [r for r in results if r["evaluation_role"] != "probe"]
    closed_rows = [r for r in main_rows if (r["answer_type"] or "").strip().lower() == "closed"]
    open_rows = [r for r in main_rows if (r["answer_type"] or "").strip().lower() == "open"]
    success_times = [r["time_s"] for r in results if r["success"] and r["time_s"] is not None]
    print("\n" + "=" * 65)
    print("=== KẾT QUẢ TỔNG HỢP VQA ===")
    print(f"Thực thi thành công: {len(success_times)}/{len(results)}")
    if success_times:
        print(f"Thời gian phản hồi trung bình lượt thành công, sau khởi tạo: "
              f"{sum(success_times) / len(success_times):.3f} giây")
    _print_group("Tập chính", main_rows)
    _print_group("open", open_rows)
    _print_group("closed", closed_rows)
    for label in CLOSED_LABELS.values():
        _print_group(f"closed / tham chiếu {label}", [r for r in closed_rows if r["ground_truth_label"] == label])
    if any(r["is_chest"] for r in main_rows):
        _print_group("Ngực (mọi phương thức)", [r for r in main_rows if r["is_chest"] == "1"])
        cxr_rows = [r for r in main_rows if r["is_cxr"] == "1"]
        _print_group("X-quang ngực", cxr_rows)
        _print_group("X-quang ngực PA", [r for r in cxr_rows if r["projection"] == "PA"])
    else:
        _print_group("X-quang ngực PA", [r for r in main_rows if r["image_name"] in CHEST_PA_IMAGES])
    for body_part in sorted({r["body_part"] for r in main_rows}):
        _print_group(body_part, [r for r in main_rows if r["body_part"] == body_part])
    for row in results:
        if row["evaluation_role"] == "probe":
            print(f'Ca thăm dò {row["question_id"]}: {row["status"]}'
                  f' | Nhãn: {row["answer"] or "chưa xác định"}')
    print("Cột f1 là số đo trùng token tham khảo cho câu open. Câu open cần chấm theo ý nghĩa.")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nĐã lưu kết quả chi tiết ra: {out_p.resolve()}")

    return results


def _evaluate_row(question_id, row, image_dir, pipeline):
    img_name = row.get("image_name") or ""
    q_vi = row.get("question_vi", "")
    gt_vi = row.get("answer_vi", "")
    ref_key = unicodedata.normalize("NFC", gt_vi or "").strip().casefold()
    ref_label = CLOSED_LABELS.get(ref_key)
    # CSV mặc định cũ chưa có cột answer_type.
    answer_type = row.get("answer_type", "closed" if ref_label else "open")
    kind = (answer_type or "").strip().lower()
    raw_answer = None
    time_s = None
    img_path = image_dir / img_name
    if img_path.is_file():
        start = time.perf_counter()
        res = answer_question(str(img_path), q_vi, pipeline=pipeline)
        time_s = time.perf_counter() - start
        success, error, error_code = res.success, res.error, res.error_code
        raw_answer = res.answer
    else:
        success, error, error_code = False, "Ảnh không tồn tại", "IMAGE_NOT_FOUND"

    answer = matched = f1 = None
    score_note = ""
    if not success:
        status = "error"
    elif not raw_answer or not raw_answer.strip():
        status, score_note = "needs_review", "Đầu ra rỗng"
    elif kind == "open":
        answer = raw_answer
        status = "pending_manual"
        if gt_vi and gt_vi.strip():
            f1 = round(compute_f1_score(raw_answer, gt_vi), 4)
    elif kind == "closed":
        answer = normalize_closed_answer(raw_answer)
        if answer is None or ref_label is None:
            status = "needs_review"
            score_note = (
                "Đáp án tham chiếu không có nhãn rõ"
                if ref_label is None else "Đầu ra không có nhãn rõ"
            )
        else:
            matched = answer == ref_label
            status = "correct" if matched else "incorrect"
    else:
        status, score_note = "needs_review", "answer_type trống hoặc không hợp lệ"

    return {
        "question_id": question_id,
        "image_name": img_name,
        "body_part": row.get("body_part") or "Chung",
        "question_vi": q_vi,
        "ground_truth": gt_vi,
        "prediction": raw_answer or "",
        "success": success,
        "error": error or "",
        "match": matched,
        "f1": f1,
        "time_s": time_s,
        "answer_type": answer_type,
        "evaluation_role": (row.get("evaluation_role") or "score").strip().lower(),
        "raw_answer": raw_answer,
        "answer": answer,
        "ground_truth_label": ref_label if kind == "closed" else None,
        "status": status,
        "score_note": score_note,
        "error_code": error_code or "",
        "is_chest": row.get("is_chest", ""),
        "is_cxr": row.get("is_cxr", ""),
        "projection": row.get("projection", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Medical VQA trên questions.csv")
    parser.add_argument("--csv", default=str(_ROOT / "data" / "evaluation" / "questions.csv"),
                        help="Đường dẫn file questions.csv")
    parser.add_argument("--image-dir", default=str(_ROOT / "data" / "evaluation" / "images"),
                        help="Thư mục chứa ảnh đánh giá")
    parser.add_argument("--mock", action="store_true", help="Dùng model giả lập")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng câu hỏi cần chạy")
    parser.add_argument("--output", default=None, help="Đường dẫn lưu kết quả CSV")
    parser.add_argument("--checkpoint", type=Path, help="SQLite để lưu và tiếp tục các câu theo question_id")
    args = parser.parse_args()

    use_mock = args.mock or os.getenv("USE_MOCK") == "1"
    run_vqa_evaluation(
        csv_path=Path(args.csv),
        image_dir=Path(args.image_dir),
        use_mock=use_mock,
        limit=args.limit,
        output_path=Path(args.output) if args.output else None,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
