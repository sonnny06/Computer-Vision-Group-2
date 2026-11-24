import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Optional

# Thêm thư mục Course_work vào sys.path để luôn import được src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from src.engine.errors import (
    InferenceError, InferenceOutOfMemory, InferenceTimeout,
    InvalidImageError, InvalidQuestionError, PipelineError, PromptError,
)
from src.engine.schemas import PipelineResult

logger = logging.getLogger(__name__)

# Một luồng duy nhất vì model dùng chung 1 GPU (Thái đã có lock bên trong)
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")


class MedicalPipeline:
    def __init__(self, vlm, prompt_builder, preprocessor, *,
                 max_new_tokens=128, max_image_side=768, max_upload_mb=15,
                 max_question_chars=1500, timeout=120.0, oom_exceptions=()):
        self.vlm = vlm                        # Thái: generate(image, prompt, max_new_tokens)
        self.prompt_builder = prompt_builder  # Tôn: (task, question) -> str
        self.preprocessor = preprocessor      # Trí: (image, max_side, max_mb) -> PIL
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.max_upload_mb = max_upload_mb
        self.max_question_chars = max_question_chars
        self.timeout = timeout
        self.oom_exceptions = tuple(oom_exceptions)

    # ---------- các bước ----------
    def _check_question(self, task, question):
        if task != "vqa":
            return None
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuestionError("Câu hỏi không được để trống")
        question = question.strip()
        if len(question) > self.max_question_chars:
            raise InvalidQuestionError(
                f"Câu hỏi quá dài (tối đa {self.max_question_chars} ký tự)")
        return question

    def _prepare_image(self, image):
        if image is None:
            raise InvalidImageError("Chưa có ảnh đầu vào")
        try:
            return self.preprocessor(image, self.max_image_side, self.max_upload_mb)
        except PipelineError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Ảnh không hợp lệ: {e}")

    def _build_prompt(self, task, question):
        try:
            return self.prompt_builder(task, question)
        except PipelineError:
            raise
        except Exception as e:
            raise PromptError(f"Không tạo được prompt: {e}")

    def _infer(self, image, prompt) -> str:
        future = _EXECUTOR.submit(
            self.vlm.generate, image, prompt, self.max_new_tokens)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            raise InferenceTimeout(
                f"Model không phản hồi sau {self.timeout:.0f} giây")
        except self.oom_exceptions as e:
            raise InferenceOutOfMemory(str(e))
        except Exception as e:
            raise InferenceError(f"Model lỗi: {e}")

    # ---------- điểm vào chính ----------
    def run(self, image, task: str, question: Optional[str] = None) -> PipelineResult:
        start = time.time()
        try:
            question = self._check_question(task, question)
            img = self._prepare_image(image)
            prompt = self._build_prompt(task, question)
            answer = self._infer(img, prompt)
            return PipelineResult(True, task, answer=answer,
                                  meta={"time_s": round(time.time() - start, 2)})
        except PipelineError as e:
            logger.warning("Pipeline error [%s]: %s", e.code, e)
            return PipelineResult(False, task, error=str(e), error_code=e.code)
        except Exception:
            logger.exception("Lỗi không lường trước")
            return PipelineResult(False, task, error="Lỗi hệ thống, vui lòng thử lại",
                                  error_code="UNKNOWN")


# ---------- Mock (để test và để Sơn làm giao diện khi chưa có GPU) ----------
class _FakeVLM:
    def generate(self, image, prompt, max_new_tokens):
        return "Kết quả giả lập"


def _simple_prepare(image, max_side, max_mb):
    if isinstance(image, str):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        raise ValueError("Định dạng ảnh không được hỗ trợ")
    return image.convert("RGB")


def _default_prompt_builder(task, question=None):
    if task == "caption":
        return "Mô tả ảnh y tế này bằng tiếng Việt."
    return f"Trả lời bằng tiếng Việt, dựa trên ảnh. Câu hỏi: {question}"


# ---------- Khởi tạo ----------
def build_pipeline(use_mock: bool = False, config_path=None) -> MedicalPipeline:
    from src.utils.config import ROOT, load_config

    cfg = load_config(config_path or ROOT / "configs/model_config.yaml")
    inf = cfg["inference"]
    options = dict(
        max_new_tokens=inf["max_new_tokens"],
        max_image_side=inf["max_image_side"],
        max_upload_mb=inf["max_upload_mb"],
        max_question_chars=inf["max_question_chars"],
    )

    if use_mock:
        return MedicalPipeline(_FakeVLM(), _default_prompt_builder,
                               _simple_prepare, **options)

    from src.data.preprocessing import prepare_image                    # Trí
    from src.models.pretrained_vlm import VLMOutOfMemoryError, get_vlm  # Thái
    from src.utils.prompt import get_caption_prompt, get_vqa_prompt     # Tôn

    def prompt_builder(task, question=None):
        if task == "caption":
            return get_caption_prompt()
        return get_vqa_prompt(question)

    return MedicalPipeline(get_vlm(cfg), prompt_builder, prepare_image,
                           oom_exceptions=(VLMOutOfMemoryError,), **options)


_default_pipeline = None


def get_default_pipeline() -> MedicalPipeline:
    """Tạo một lần, dùng lại. Đặt biến môi trường USE_MOCK=1 để dùng model giả."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = build_pipeline(use_mock=os.getenv("USE_MOCK") == "1")
    return _default_pipeline


def get_pipeline_or_error(task: str):
    """Trả về (pipeline, None) hoặc (None, PipelineResult lỗi) nếu không nạp được model."""
    try:
        return get_default_pipeline(), None
    except Exception as e:
        logger.exception("Không khởi tạo được pipeline")
        return None, PipelineResult(
            False, task, error=f"Không khởi tạo được model: {e}",
            error_code="MODEL_UNAVAILABLE")


if __name__ == "__main__":
    print("Testing MedicalPipeline (mock mode)...")
    pipe = build_pipeline(use_mock=True)
    test_img = Image.new("RGB", (200, 200), color="blue")
    cap_res = pipe.run(test_img, task="caption")
    print("Caption status:", "OK" if cap_res.success else cap_res.error_code)
    print("Caption answer:", cap_res.answer)
    vqa_res = pipe.run(test_img, task="vqa", question="Bộ phận nào trong ảnh?")
    print("VQA status:", "OK" if vqa_res.success else vqa_res.error_code)
    print("VQA answer:", vqa_res.answer)
