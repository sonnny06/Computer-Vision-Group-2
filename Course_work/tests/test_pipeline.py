"""Unit test kiểm thử module MedicalPipeline và inference.
Phụ trách: Pipeline & Inference

Chạy kiểm thử:
    python -m unittest tests/test_pipeline.py
hoặc:
    pytest tests/test_pipeline.py
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

# Đảm bảo import được src khi chạy độc lập
COURSE_WORK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = COURSE_WORK_DIR.parent
for p in [str(COURSE_WORK_DIR), str(REPO_ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.engine.errors import (
    InferenceError,
    InferenceOutOfMemory,
    InferenceTimeout,
    InvalidImageError,
    InvalidQuestionError,
    PromptError,
)
from src.engine.pipeline import MedicalPipeline, build_pipeline
from src.engine.schemas import PipelineResult
from src.inference.captioning import generate_caption
from src.inference.vqa import answer_question


class TestMedicalPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_pipeline = build_pipeline(use_mock=True)
        self.sample_image = Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_caption_success(self):
        result = self.mock_pipeline.run(self.sample_image, task="caption")
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.success)
        self.assertEqual(result.task, "caption")
        self.assertIsNotNone(result.answer)
        self.assertIn("time_s", result.meta)

    def test_vqa_success(self):
        result = self.mock_pipeline.run(
            self.sample_image,
            task="vqa",
            question="Ảnh chụp bộ phận nào?",
        )
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.success)
        self.assertEqual(result.task, "vqa")
        self.assertIsNotNone(result.answer)

    def test_vqa_empty_question(self):
        result = self.mock_pipeline.run(self.sample_image, task="vqa", question="")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "INVALID_QUESTION")

        result_spaces = self.mock_pipeline.run(
            self.sample_image, task="vqa", question="   "
        )
        self.assertFalse(result_spaces.success)
        self.assertEqual(result_spaces.error_code, "INVALID_QUESTION")

    def test_vqa_question_too_long(self):
        long_q = "a" * (self.mock_pipeline.max_question_chars + 10)
        result = self.mock_pipeline.run(self.sample_image, task="vqa", question=long_q)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "INVALID_QUESTION")

    def test_missing_image(self):
        result = self.mock_pipeline.run(None, task="caption")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "INVALID_IMAGE")

    def test_invalid_image_type(self):
        result = self.mock_pipeline.run(12345, task="caption")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "INVALID_IMAGE")

    def test_inference_module_wrappers(self):
        caption_res = generate_caption(self.sample_image, pipeline=self.mock_pipeline)
        self.assertTrue(caption_res.success)
        self.assertEqual(caption_res.task, "caption")

        vqa_res = answer_question(
            self.sample_image,
            "Có tổn thương không?",
            pipeline=self.mock_pipeline,
        )
        self.assertTrue(vqa_res.success)
        self.assertEqual(vqa_res.task, "vqa")


if __name__ == "__main__":
    unittest.main()
