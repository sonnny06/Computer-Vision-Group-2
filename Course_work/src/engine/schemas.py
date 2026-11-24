from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineResult:
    success: bool
    task: str                       # "caption" | "vqa"
    answer: Optional[str] = None    # văn bản kết quả
    mask: Optional[Any] = None      # dự phòng cho kết quả phân vùng
    error: Optional[str] = None     # thông báo lỗi thân thiện
    error_code: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "task": self.task,
            "answer": self.answer,
            "error": self.error,
            "error_code": self.error_code,
            "meta": self.meta,
        }