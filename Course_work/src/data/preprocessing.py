"""Lớp nối: code của Thái gọi prepare_image, bên dưới dùng preprocess_image của Trí."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocessing.image_processing import (
    ImageProcessingError,
    preprocess_image,
    validate_image,
)

__all__ = ["prepare_image", "ImageProcessingError"]


def prepare_image(image, max_side: int = 768, max_mb: int = 15):
    """Kiểm tra dung lượng theo config rồi tiền xử lý ảnh bằng module của Trí.

    Trả về PIL.Image RGB (cạnh lớn nhất <= max_side).
    Ném ImageProcessingError (là ValueError) với thông báo tiếng Việt nếu ảnh không hợp lệ.
    """
    ok, message, _ = validate_image(image, max_file_size_mb=float(max_mb))
    if not ok:
        raise ImageProcessingError(message)
    return preprocess_image(image, target_max_dim=int(max_side))