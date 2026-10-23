"""Module wavelet_transform.py - Biến đổi Wavelet cho Lab 4 (TV3 - Văn Tôn).
Bao gồm:
- Thực hiện biến đổi Wavelet rời rạc 2D (2D DWT) sử dụng thư viện PyWavelets.
- Hỗ trợ các họ wavelet phổ biến như 'haar', 'db4' với các mức phân rã (level) khác nhau.
- Trích xuất các thành phần hệ số xấp xỉ (LL) và chi tiết (LH, HL, HH).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pywt

try:
    from .preprocessing import preprocess_image
except ImportError:
    from preprocessing import preprocess_image


def perform_wavelet_transform(
    image_input: Union[str, Path, np.ndarray],
    wavelet: str = 'db4',
    level: int = 2
) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Thực hiện biến đổi Wavelet đa mức (Multi-level 2D DWT) cho hình ảnh[cite: 46].

    Args:
        image_input: Đường dẫn file ảnh hoặc mảng numpy ảnh xám đã tiền xử lý.
        wavelet: Tên họ wavelet ('haar', 'db4', v.v.).
        level: Mức phân rã (decomposition level).

    Returns:
        Danh sách các hệ số wavelet trả về bởi pywt.wavedec2.
    """
    if isinstance(image_input, (str, Path)):
        img_gray = preprocess_image(image_input, target_size=(256, 256))
    elif isinstance(image_input, np.ndarray):
        img_gray = image_input.astype(np.float64)
    else:
        raise TypeError("Đầu vào phải là đường dẫn ảnh (str/Path) hoặc mảng numpy.ndarray.")

    # Thực hiện biến đổi wavelet 2D đa mức
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=level)
    return coeffs


def extract_first_level_components(
    image_input: Union[str, Path, np.ndarray],
    wavelet: str = 'haar'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thực hiện biến đổi 1 mức để trích xuất trực tiếp 4 thành phần hệ số phục vụ trực quan hóa:
    - LL (Approximation): Hệ số xấp xỉ
    - LH (Horizontal): Hệ số chi tiết ngang
    - HL (Vertical): Hệ số chi tiết dọc
    - HH (Diagonal): Hệ số chi tiết chéo
    """
    if isinstance(image_input, (str, Path)):
        img_gray = preprocess_image(image_input, target_size=(256, 256))
    elif isinstance(image_input, np.ndarray):
        img_gray = image_input.astype(np.float64)
    else:
        raise TypeError("Đầu vào phải là đường dẫn ảnh hoặc mảng numpy.ndarray.")

    # pywt.dwt2 trả về LL, và tuple (LH, HL, HH)
    LL, (LH, HL, HH) = pywt.dwt2(img_gray, wavelet=wavelet)
    return LL, LH, HL, HH