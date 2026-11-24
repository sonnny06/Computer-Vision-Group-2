"""
Module quản lý lịch sử phiên làm việc — F05 (Phụ trách: Sơn & Bão).
Hỗ trợ lưu, lọc, tìm kiếm, xóa từng mục / toàn bộ và xuất CSV / JSON.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class HistoryItem:
    id: str
    timestamp: str
    task_type: str       # "Captioning" | "VQA"
    image_name: str
    question: str
    answer: str
    time_s: float
    mode: str            # "qwen3-vl" | "smart-demo" | "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionHistory:
    """Quản lý lịch sử hỏi đáp & mô tả trong phiên làm việc."""

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self._items: list[HistoryItem] = []
        if items:
            for item in items:
                self._items.append(HistoryItem(**item))

    def add(
        self,
        task_type: str,
        image_name: str,
        answer: str,
        time_s: float,
        mode: str = "smart-demo",
        question: str = "",
    ) -> HistoryItem:
        now = datetime.now()
        entry = HistoryItem(
            id=f"{task_type[:3].upper()}-{now.strftime('%H%M%S')}-{len(self._items) + 1:03d}",
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            task_type=task_type,
            image_name=image_name,
            question=question or ("(Sinh mô tả tự động)" if task_type == "Captioning" else ""),
            answer=answer,
            time_s=round(float(time_s), 3),
            mode=mode,
        )
        self._items.insert(0, entry)
        return entry

    def list_items(self, task_filter: str = "Tất cả", keyword: str = "") -> list[HistoryItem]:
        items = self._items
        if task_filter in ("Captioning", "VQA"):
            items = [x for x in items if x.task_type == task_filter]
        kw = (keyword or "").strip().lower()
        if kw:
            items = [
                x
                for x in items
                if kw in x.image_name.lower()
                or kw in x.question.lower()
                or kw in x.answer.lower()
            ]
        return items

    def delete_by_id(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [x for x in self._items if x.id != item_id]
        return len(self._items) < before

    def clear(self) -> None:
        self._items.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [x.to_dict() for x in self._items]

    def to_json(self) -> str:
        return json.dumps(self.to_dicts(), ensure_ascii=False, indent=2)

    def to_csv(self) -> str:
        buf = io.StringIO()
        fields = ["id", "timestamp", "task_type", "image_name", "question", "answer", "time_s", "mode"]
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for item in self._items:
            writer.writerow(item.to_dict())
        return buf.getvalue()

    def __len__(self) -> int:
        return len(self._items)
