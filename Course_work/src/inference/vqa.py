import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.engine.pipeline import MedicalPipeline, get_pipeline_or_error
from src.engine.schemas import PipelineResult


def answer_question(image, question: str,
                    pipeline: Optional[MedicalPipeline] = None) -> PipelineResult:
    """Trả lời câu hỏi về ảnh. Kết quả giống generate_caption."""
    if pipeline is None:
        pipeline, err = get_pipeline_or_error("vqa")
        if err:
            return err
    return pipeline.run(image, task="vqa", question=question)