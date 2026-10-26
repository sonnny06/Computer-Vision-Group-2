"""Module preprocessing.py - Tiền xử lý ảnh cho Lab 4 (TV3 - Văn Tôn).
Bao gồm: đọc ảnh, chuyển sang ảnh xám và chuẩn hóa kích thước.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """Kiểm tra tính hợp lệ của mảng ảnh đầu vào."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Ảnh phải là mảng numpy.ndarray, nhận được: {type(image)}")
    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng.")
    return image


def load_and_grayscale(path: Union[str, Path]) -> np.ndarray:
    """Đọc ảnh từ đường dẫn và chuyển đổi trực tiếp sang ảnh xám (Grayscale)."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {p}")

    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Không thể giải mã file ảnh: {p}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Chuẩn hóa kích thước ảnh về kích thước cố định (mặc định 256x256)."""
    validate_image(image)
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("target_size phải là tuple gồm (width, height).")
    
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_image(path: Union[str, Path], target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Hàm tổng hợp quy trình tiền xử lý: Đọc -> Chuyển xám -> Chuẩn hóa kích thước."""
    gray = load_and_grayscale(path)
    processed = resize_image(gray, target_size)
    return processed