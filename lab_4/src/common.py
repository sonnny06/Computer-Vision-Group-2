"""Module common.py - Các hàm tiện ích dùng chung cho Lab 4 (Wavelet Hash).

Được sử dụng bởi tất cả các thành viên trong nhóm (TV1 - TV6).
Bao gồm:
- Đọc/ghi ảnh an toàn với đường dẫn tiếng Việt Unicode trên hệ điều hành Windows.
- Chuyển đổi không gian màu (BGR, RGB, Grayscale).
- Kiểm tra tính hợp lệ của mảng ảnh (Validation).
- Các tiện ích định dạng chuỗi nhị phân, hex và đo đạc thời gian.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ==============================================================================
# 1. ĐỌC VÀ LƯU ẢNH AN TOÀN TRÊN WINDOWS
# ==============================================================================

def read_image(path: Union[str, Path], as_gray: bool = False) -> np.ndarray:
    """Đọc ảnh từ đường dẫn hỗ trợ an toàn ký tự Unicode (tiếng Việt có dấu).

    Tránh lỗi `cv2.imread()` trả về None khi đường dẫn chứa ký tự tiếng Việt.

    Args:
        path: Đường dẫn tới file ảnh (str hoặc Path).
        as_gray: Nếu True đọc ảnh thang xám, nếu False đọc ảnh màu (BGR).

    Returns:
        np.ndarray: Mảng ảnh uint8 (2D thang xám hoặc 3D BGR).

    Raises:
        FileNotFoundError: Nếu file ảnh không tồn tại.
        ValueError: Nếu file không thể giải mã thành ảnh hợp lệ.
    """
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path_obj}")

    flags = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    try:
        # Sử dụng numpy.fromfile + cv2.imdecode để an toàn tuyệt đối với Unicode trên Windows
        with open(path_obj, "rb") as f:
            bytes_data = f.read()
        image_array = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(image_array, flags)
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file ảnh {path_obj}: {e}") from e

    if img is None:
        raise ValueError(f"Không thể giải mã nội dung ảnh từ file: {path_obj}")

    return img


def save_image(path: Union[str, Path], image: np.ndarray) -> bool:
    """Lưu ảnh ra file hỗ trợ Unicode path trên Windows.

    Args:
        path: Đường dẫn lưu ảnh.
        image: Mảng ảnh numpy.

    Returns:
        bool: True nếu lưu thành công.
    """
    path_obj = Path(path).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    ext = path_obj.suffix.lower()
    if not ext:
        ext = ".png"
        path_obj = path_obj.with_suffix(ext)

    success, encoded_img = cv2.imencode(ext, image)
    if not success:
        return False

    with open(path_obj, "wb") as f:
        f.write(encoded_img.tobytes())
    return True


# ==============================================================================
# 2. XÁC THỰC VÀ CHUYỂN ĐỔI ẢNH (VALIDATION & CONVERSION)
# ==============================================================================

def validate_image(image: Any) -> np.ndarray:
    """Kiểm tra tính hợp lệ của mảng ảnh đầu vào."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Ảnh đầu vào phải là numpy.ndarray, nhận được: {type(image)}")
    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng (kích thước bằng 0).")
    if image.ndim not in (2, 3):
        raise ValueError(f"Ảnh phải có 2 chiều (grayscale) hoặc 3 chiều (color), nhận được {image.ndim} chiều.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Ảnh 3D phải có 1, 3 hoặc 4 kênh màu, nhận được {image.shape[2]} kênh.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh sang thang xám chuẩn uint8 2D."""
    validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy()
        elif channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Không thể chuyển đổi ảnh shape={image.shape} sang grayscale.")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR sang RGB an toàn để hiển thị với Matplotlib."""
    validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].copy()
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image
