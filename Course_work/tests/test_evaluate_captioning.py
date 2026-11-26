"""Regression tests for the saved-review and lexical Captioning modes."""

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

COURSE_WORK_DIR = Path(__file__).resolve().parents[1]
if str(COURSE_WORK_DIR) not in sys.path:
    sys.path.insert(0, str(COURSE_WORK_DIR))

from src.engine.schemas import PipelineResult
from src.evaluation import evaluate_captioning


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestCaptionReviews(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.references, self.predictions, self.rubric = [
            Path(temporary.name) / name for name in ("refs.csv", "predictions.csv", "rubric.csv")
        ]
        self.ref = dict(image_id="A", modality="CT", region_group="Head", pilot_overlap="no")
        self.pred = dict(image_id="A", success="True", raw_caption="Ảnh CT đầu", time_s="12.5")
        self.review = dict(
            image_id="A", modality_match="1", body_region_match="1", core_structure_1_match="0",
            core_structure_2_match="0", required_feature_match="-", main_coverage="partial",
            contradicts_evidence="Mâu thuẫn", needs_verification="Cần rà", vietnamese_clarity="clear",
            time_s="1234",
        )
        write_csv(self.references, [self.ref])
        write_csv(self.predictions, [self.pred])
        write_csv(self.rubric, [self.review])

    def test_cli_joins_by_id_and_uses_saved_run_time(self):
        write_csv(self.references, [self.ref, dict(self.ref, image_id="B")])
        write_csv(self.predictions, [dict(self.pred, image_id="B", success="False", time_s="0"), self.pred])
        write_csv(self.rubric, [self.review, dict(self.review, image_id="B", main_coverage="execution_error")])
        with redirect_stdout(io.StringIO()) as out, patch.object(
            sys, "argv", ["evaluate_captioning.py", "--summarize-reviews",
                          str(self.references), str(self.predictions), str(self.rubric)]
        ), patch.object(evaluate_captioning, "build_pipeline") as build:
            evaluate_captioning.main()
        build.assert_not_called()
        total = next(csv.DictReader(io.StringIO(out.getvalue())))
        self.assertEqual(total["successful"], "1")
        self.assertEqual(total["execution_error"], "1")
        self.assertEqual(total["main_partial"], "1")
        self.assertEqual(total["contains_contradiction"], "1")
        self.assertEqual(total["contains_unverified_claim"], "1")
        self.assertEqual(total["mean_time_s"], "12.5")

    def test_cli_rejects_missing_and_extra_ids(self):
        write_csv(self.predictions, [dict(self.pred, image_id="X")])
        with redirect_stderr(io.StringIO()) as err, patch.object(
            sys, "argv", ["evaluate_captioning.py", "--summarize-reviews",
                          str(self.references), str(self.predictions), str(self.rubric)]
        ), self.assertRaises(SystemExit) as failure:
            evaluate_captioning.main()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("predictions.csv: thiếu image_id A / dư image_id X", err.getvalue())

    def test_all_three_files_reject_duplicate_and_blank_ids(self):
        for path, row in ((self.references, self.ref), (self.predictions, self.pred), (self.rubric, self.review)):
            with self.subTest(path=path):
                write_csv(path, [row, row])
                with self.assertRaisesRegex(ValueError, f"{path.name}:3: trùng image_id A"):
                    evaluate_captioning.summarize_caption_reviews(self.references, self.predictions, self.rubric)
                write_csv(path, [dict(row, image_id=" ")])
                with self.assertRaisesRegex(ValueError, f"{path.name}:2: image_id trống"):
                    evaluate_captioning.summarize_caption_reviews(self.references, self.predictions, self.rubric)
                write_csv(path, [row])


class TestLegacyCaptionCli(unittest.TestCase):
    def test_mock_cli_keeps_bleu_and_f1_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, output = root / "metadata.csv", root / "out.csv"
            (root / "sample.jpg").touch()
            # The legacy reader uses UTF-8 without a BOM.
            metadata.write_text(
                "image_id,filename,ground_truth_caption_vi\nA,sample.jpg,Ảnh vùng ngực\n", encoding="utf-8"
            )
            pipeline = Mock()
            pipeline.run.return_value = PipelineResult(
                task="caption", success=True, answer="Ảnh ngực", meta={"time_s": 0.2}
            )
            with redirect_stdout(io.StringIO()) as out, patch.object(
                sys, "argv", ["evaluate_captioning.py", "--csv", str(metadata), "--image-dir",
                              str(root), "--mock", "--limit", "1", "--output", str(output)]
            ), patch.object(evaluate_captioning, "build_pipeline", return_value=pipeline) as build:
                evaluate_captioning.main()
            build.assert_called_once_with(use_mock=True)
            self.assertIn("Điểm BLEU-1", out.getvalue())
            self.assertIn("Điểm F1 Token", out.getvalue())
            with output.open(encoding="utf-8", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["image_id"], "A")
            self.assertEqual(row["bleu1"], "1.0")
            self.assertEqual(row["f1"], "0.8")
