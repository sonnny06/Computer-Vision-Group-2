"""Kiểm thử phân loại câu trả lời và tổng hợp điểm VQA."""

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

COURSE_WORK_DIR = Path(__file__).resolve().parents[1]
if str(COURSE_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_WORK_DIR))

from src.engine.pipeline import MedicalPipeline
from src.engine.schemas import PipelineResult
from src.evaluation import evaluate_vqa


class TestVqaCli(unittest.TestCase):
    def test_official_csv_and_its_images_are_defaults(self):
        data_dir = COURSE_WORK_DIR / "data" / "evaluation"
        with patch.object(sys, "argv", ["evaluate_vqa.py", "--mock"]), patch.object(
            evaluate_vqa, "run_vqa_evaluation"
        ) as run:
            evaluate_vqa.main()
        self.assertEqual(run.call_args.kwargs["csv_path"], data_dir / "questions.csv")
        self.assertEqual(run.call_args.kwargs["image_dir"], data_dir / "vqa_images")

    def test_alternate_csv_uses_shared_images_unless_overridden(self):
        data_dir = COURSE_WORK_DIR / "data" / "evaluation"
        csv_path = data_dir / "questions_handover_235.csv"
        for extra, image_dir in (([], data_dir / "images"), (["--image-dir", "custom"], Path("custom"))):
            with self.subTest(extra=extra), patch.object(
                sys, "argv", ["evaluate_vqa.py", "--csv", str(csv_path), "--mock", *extra]
            ), patch.object(evaluate_vqa, "run_vqa_evaluation") as run:
                evaluate_vqa.main()
            self.assertEqual(run.call_args.kwargs["csv_path"], csv_path)
            self.assertEqual(run.call_args.kwargs["image_dir"], image_dir)


class TestClosedAnswer(unittest.TestCase):
    def test_clear_labels(self):
        cases = {
            "  **Có**, đây là ảnh X-quang.": "Có",
            "Không có dấu hiệu được hỏi.": "Không",
            "Không phải là mặt cắt ngang.": "Không",
            "Không đủ thông tin": "Không đủ thông tin",
            "Không có đủ thông tin để trả lời.": "Không đủ thông tin",
            "Ảnh không đủ rõ hoặc không chứa thông tin để trả lời câu hỏi này.": "Không đủ thông tin",
        }
        for raw, label in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(evaluate_vqa.normalize_closed_answer(raw), label)

    def test_ambiguous_or_conflicting_answers(self):
        for raw in (
            "Có thể có.",
            "Ảnh có thể cho thấy điều đó.",
            "Có hoặc không",
            "Có. Không.",
            "Có, không có.",
            "Có, nhưng không.",
            "Không, nếu chỉ xét một phía.",
            "Không, không thể biết từ ảnh này.",
            "Có?",
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(evaluate_vqa.normalize_closed_answer(raw))


class TestVQAEvaluation(unittest.TestCase):
    def test_checkpoint_resumes_saved_ids_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.csv"
            source.write_text(
                "question_id,image_name,question_vi,answer_vi,answer_type\n"
                "Q_A,one.jpg,Ảnh chụp vùng nào?,Ngực,open\n"
                "Q_B,two.jpg,Đây là ảnh X-quang?,Có,closed\n",
                encoding="utf-8",
            )
            (root / "one.jpg").touch()
            (root / "two.jpg").touch()
            checkpoint = root / "progress.sqlite"
            pipe = Mock(spec=MedicalPipeline)
            pipe.run.side_effect = [
                PipelineResult(True, "vqa", answer="Lồng ngực"),
                RuntimeError("Mất tiến trình"),
            ]
            with redirect_stdout(io.StringIO()), patch.object(evaluate_vqa, "build_pipeline", return_value=pipe):
                with self.assertRaisesRegex(RuntimeError, "Mất tiến trình"):
                    evaluate_vqa.run_vqa_evaluation(source, root, checkpoint_path=checkpoint)
                pipe.run.reset_mock(side_effect=True)
                pipe.run.return_value = PipelineResult(True, "vqa", answer="Có.")
                rows = evaluate_vqa.run_vqa_evaluation(source, root, checkpoint_path=checkpoint)
            pipe.run.assert_called_once_with(str(root / "two.jpg"), task="vqa", question="Đây là ảnh X-quang?")
            self.assertEqual([(r["question_id"], r["raw_answer"]) for r in rows], [
                ("Q_A", "Lồng ngực"), ("Q_B", "Có."),
            ])
            with redirect_stdout(io.StringIO()), patch.object(evaluate_vqa, "build_pipeline") as build:
                replay = evaluate_vqa.run_vqa_evaluation(source, root, checkpoint_path=checkpoint)
                self.assertEqual(replay, rows)
                with self.assertRaisesRegex(ValueError, "Checkpoint không khớp"):
                    evaluate_vqa.run_vqa_evaluation(source, root, use_mock=True, checkpoint_path=checkpoint)
                source.write_text(source.read_text(encoding="utf-8").replace("Ngực", "Bụng"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Checkpoint không khớp"):
                    evaluate_vqa.run_vqa_evaluation(source, root, checkpoint_path=checkpoint)
            build.assert_not_called()

    def test_bom_typed_rows_failures_and_probe(self):
        cases = [
            ("Q_0058", "phoi_3.jpg", "open", "X-quang", "score"),
            ("Q_negative", "phoi_4.jpg", "closed", "Có", "score"),
            ("Q_uncertain", "phoi_4.jpg", "closed", "Không", "score"),
            ("Q_conflict", "tay_1.jpg", "closed", "Có", "score"),
            ("Q_error", "tay_1.jpg", "open", "Tay", "score"),
            ("Q_missing", "missing.jpg", "closed", "Không", "score"),
            ("PILOT_02", "phoi_3.jpg", "closed", "Không đủ thông tin", "probe"),
        ]
        raw_open = " tia X.\n"
        pipeline = Mock(spec=MedicalPipeline)
        pipeline.run.side_effect = [
            PipelineResult(True, "vqa", answer=raw_open),
            PipelineResult(True, "vqa", answer="Không có."),
            PipelineResult(True, "vqa", answer="Không đủ thông tin"),
            PipelineResult(True, "vqa", answer="Có. Không."),
            PipelineResult(False, "vqa", error="Thử lại", error_code="INFERENCE_ERROR"),
            PipelineResult(True, "vqa", answer="Không đủ thông tin"),
        ]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.csv"
            output = root / "results.csv"
            for name in ("phoi_3.jpg", "phoi_4.jpg", "tay_1.jpg"):
                (root / name).touch()
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "question_id", "image_name", "answer_type", "answer_vi",
                    "evaluation_role", "question_vi", "modality",
                ])
                for case in cases:
                    writer.writerow([*case, "Câu hỏi tiếng Việt?", "PA"])

            log = io.StringIO()
            # Lỗi thực thi mất 5 giây. Mỗi lượt thành công mất 1 giây.
            ticks = [0, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 19, 19, 20]
            with (
                patch.object(evaluate_vqa, "build_pipeline", return_value=pipeline) as build,
                patch.object(evaluate_vqa.time, "perf_counter", side_effect=ticks),
                redirect_stdout(log),
            ):
                rows = evaluate_vqa.run_vqa_evaluation(source, root, output_path=output)

            build.assert_called_once_with(use_mock=False)
            pipeline.run.assert_any_call(str(root / "phoi_3.jpg"), task="vqa", question="Câu hỏi tiếng Việt?")
            self.assertEqual(pipeline.run.call_count, 6)
            self.assertEqual([row["question_id"] for row in rows], [case[0] for case in cases])
            self.assertEqual([row["status"] for row in rows], [
                "pending_manual", "incorrect", "incorrect", "needs_review",
                "error", "error", "correct",
            ])
            self.assertEqual([row["match"] for row in rows], [None, False, False, None, None, None, True])
            self.assertEqual(rows[0]["answer"], raw_open)
            self.assertEqual(rows[0]["f1"], 0.5)
            self.assertEqual(rows[1]["answer"], "Không")
            self.assertEqual(rows[2]["answer"], "Không đủ thông tin")
            self.assertEqual(rows[4]["error_code"], "INFERENCE_ERROR")
            self.assertEqual(rows[4]["time_s"], 5)
            self.assertEqual(rows[5]["error_code"], "IMAGE_NOT_FOUND")
            self.assertIsNone(rows[5]["time_s"])

            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["raw_answer"], raw_open)
            self.assertEqual(saved[0]["answer_type"], "open")
            self.assertEqual(saved[0]["match"], "")
            self.assertEqual(saved[4]["match"], "")

        summary = log.getvalue()
        self.assertIn("Khởi tạo pipeline trước khi chạy các câu: 10.000 giây", summary)
        self.assertIn("sau khởi tạo: 1.000 giây", summary)
        self.assertRegex(summary, r"Tập chính\s+\|\s+6 câu \| 0/2 đã chấm")
        self.assertRegex(summary, r"X-quang ngực PA\s+\|\s+3 câu")
        self.assertIn("Ca thăm dò PILOT_02: correct", summary)

    def test_legacy_csv_and_explicit_open_type(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.jpg").touch()
            source = root / "input.csv"
            pipe = Mock(spec=MedicalPipeline)
            pipe.run.return_value = PipelineResult(True, "vqa", answer="Có")
            for header, row, expected in (
                ("", "", ("closed", "correct")),
                (",answer_type", ",open", ("open", "pending_manual")),
                (",answer_type", ",", ("", "needs_review")),
            ):
                with self.subTest(answer_type=row):
                    source.write_text(
                        f"image_name,question_vi,answer_vi{header}\nimage.jpg,Đây là ảnh ngực?,Có{row}\n",
                        encoding="utf-8",
                    )
                    with patch.object(evaluate_vqa, "build_pipeline", return_value=pipe), redirect_stdout(io.StringIO()):
                        result, = evaluate_vqa.run_vqa_evaluation(source, root, use_mock=True)
                    self.assertEqual((result["answer_type"], result["status"]), expected)
                    self.assertEqual(result["question_id"], "Q_0001")


if __name__ == "__main__":
    unittest.main()
