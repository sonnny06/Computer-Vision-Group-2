import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.engine.pipeline import MedicalPipeline, get_pipeline_or_error
from src.engine.schemas import PipelineResult


def generate_caption(image, pipeline: Optional[MedicalPipeline] = None) -> PipelineResult:
    """Tạo mô tả cho ảnh. Thành công: result.answer; thất bại: result.error, result.error_code."""
    if pipeline is None:
        pipeline, err = get_pipeline_or_error("caption")
        if err:
            return err
    return pipeline.run(image, task="caption")