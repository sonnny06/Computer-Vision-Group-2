"""Module nonlinear_filters.py - Cài đặt các bộ lọc phi tuyến (Non-linear Spatial Filters).

Phần phụ trách của Thành viên 4 (TV4 - Trí):
- Cài đặt bộ lọc trung vị (Median Filter) - khắc tinh của nhiễu xung / nhiễu muối tiêu (Salt & Pepper Noise).
- Hỗ trợ cả tích chập trượt thủ công (Manual sliding-window median) và hàm tối ưu `cv2.medianBlur`.
- Cài đặt các bộ lọc phi tuyến xếp hạng khác:
  + Bộ lọc giá trị nhỏ nhất (Min Filter / Erosion).
  + Bộ lọc giá trị lớn nhất (Max Filter / Dilation).
  + Bộ lọc song phương (Bilateral Filter) làm mượt phẳng vùng đồng nhất nhưng bảo tồn sắc cạnh.
- Đánh giá khả năng khử nhiễu muối tiêu vượt trội của lọc trung vị so với các bộ lọc tuyến tính (Mean/Gaussian).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


# ==============================================================================
# 1. HÀM KIỂM TRA ĐẦU VÀO (INPUT VALIDATION)
# ==============================================================================

def validate_image(image: np.ndarray) -> np.ndarray:
    """Kiểm tra tính hợp lệ của ảnh đầu vào."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Ảnh đầu vào phải là numpy.ndarray, nhận được: {type(image)}")
    if image.dtype != np.uint8:
        raise TypeError(f"Ảnh đầu vào phải có dtype là uint8, nhận được: {image.dtype}")
    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng (kích thước bằng 0).")
    if image.ndim not in (2, 3):
        raise ValueError(f"Ảnh phải có 2 chiều (grayscale) hoặc 3 chiều (color), nhận được {image.ndim} chiều.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Ảnh màu phải có 1, 3 hoặc 4 kênh màu, nhận được {image.shape[2]} kênh.")
    return image


def validate_ksize(ksize: Union[int, Tuple[int, int]]) -> int:
    """Kiểm tra và chuẩn hóa kích thước kernel cho bộ lọc phi tuyến (phải là số nguyên lẻ dương)."""
    if isinstance(ksize, int):
        k = ksize
    elif isinstance(ksize, (tuple, list)) and len(ksize) == 2:
        if ksize[0] != ksize[1]:
            raise ValueError(f"Bộ lọc trung vị OpenCV yêu cầu kernel vuông (kw == kh), nhận được: {ksize}")
        k = ksize[0]
    else:
        raise TypeError(f"ksize phải là số nguyên int hoặc tuple 2 phần tử bằng nhau, nhận được: {ksize}")

    if k <= 0:
        raise ValueError(f"Kích thước kernel phải lớn hơn 0, nhận được: {k}")
    if k % 2 == 0:
        raise ValueError(f"Kích thước kernel của bộ lọc trung vị phải là số lẻ, nhận được: {k}")

    return k


# ==============================================================================
# 2. CÀI ĐẶT BỘ LỌC TRUNG VỊ (MEDIAN FILTER)
# ==============================================================================

def apply_median_filter(
    image: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = 3,
    use_opencv: bool = True,
) -> np.ndarray:
    """Áp dụng bộ lọc trung vị (Median Filter) lên ảnh.

    Bộ lọc phi tuyến thay thế giá trị điểm ảnh bằng giá trị trung vị của các điểm lân cận.
    Đặc tính quan trọng: Loại bỏ hoàn toàn giá trị đột biến (0 và 255 của nhiễu muối tiêu)
    mà không tạo ra giá trị nhòe trung gian như các bộ lọc tuyến tính.

    Args:
        image: Mảng ảnh uint8 (2D thang xám hoặc 3D BGR).
        ksize: Kích thước cửa sổ lọc lẻ (ví dụ 3 hoặc 5).
        use_opencv: Nếu True sử dụng `cv2.medianBlur()`.
                    Nếu False sử dụng thuật toán cửa sổ trượt thủ công.

    Returns:
        np.ndarray: Ảnh uint8 sau khi lọc trung vị.
    """
    validate_image(image)
    k = validate_ksize(ksize)

    if use_opencv:
        return cv2.medianBlur(image, k)
    else:
        return apply_manual_median_filter(image, ksize=k)


def apply_manual_median_filter(
    image: np.ndarray,
    ksize: int = 3,
    border_mode: str = "edge",
) -> np.ndarray:
    """Cài đặt thuật toán lọc trung vị thủ công bằng cửa sổ trượt (Sliding Window).

    Minh chứng bản chất phi tuyến:
        g(x, y) = median({f(x+i, y+j) | (i, j) in W})
    Không có phép nhân ma trận trọng số (không phải tích chập tuyến tính).

    Args:
        image: Mảng ảnh uint8.
        ksize: Khổ cửa sổ lẻ (3, 5,...).
        border_mode: Cách tạo viền ('reflect', 'edge', 'constant').

    Returns:
        np.ndarray: Ảnh uint8 sau khi lọc trung vị thủ công.
    """
    validate_image(image)
    k = validate_ksize(ksize)
    pad = k // 2

    is_color = (image.ndim == 3)
    channels = [image[:, :, c] for c in range(image.shape[2])] if is_color else [image]
    out_channels = []

    for ch in channels:
        h, w = ch.shape
        if border_mode == "reflect":
            padded = np.pad(ch, pad, mode="reflect")
        elif border_mode == "edge":
            padded = np.pad(ch, pad, mode="edge")
        else:
            padded = np.pad(ch, pad, mode="constant", constant_values=0)

        # Trích xuất tất cả các khối lân cận kích thước k x k
        # Bằng cách xếp chồng k*k mảng dịch chuyển để vectorized cực nhanh bằng NumPy
        neighbors = []
        for r in range(k):
            for c in range(k):
                neighbors.append(padded[r : r + h, c : c + w])

        # Xếp thành mảng 3D: (k*k, h, w)
        stack = np.array(neighbors, dtype=np.uint8)
        # Tính median theo trục đầu tiên
        med_ch = np.median(stack, axis=0).astype(np.uint8)
        out_channels.append(med_ch)

    if is_color:
        return np.dstack(out_channels)
    else:
        return out_channels[0]


# ==============================================================================
# 3. CÁC BỘ LỌC PHI TUYẾN XẾP HẠNG KHÁC (MIN, MAX, BILATERAL)
# ==============================================================================

def apply_min_filter(
    image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """Bộ lọc giá trị nhỏ nhất (Minimum Filter / Erosion).

    Thay thế pixel bằng giá trị nhỏ nhất trong lân cận, làm tối ảnh và mở rộng vùng tối.
    """
    validate_image(image)
    k = validate_ksize(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(image, kernel)


def apply_max_filter(
    image: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """Bộ lọc giá trị lớn nhất (Maximum Filter / Dilation).

    Thay thế pixel bằng giá trị lớn nhất trong lân cận, làm sáng ảnh và mở rộng vùng sáng.
    """
    validate_image(image)
    k = validate_ksize(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(image, kernel)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Bộ lọc song phương (Bilateral Filter) bảo tồn biên cạnh.

    Kết hợp cả khoảng cách không gian (Domain filter) và chênh lệch cường độ sáng (Range filter):
    - Các điểm ảnh gần nhau VÀ có cường độ tương đồng mới được làm mượt.
    - Tại các cạnh có chênh lệch cường độ lớn, trọng số giảm mạnh, giúp biên cạnh sắc nét hoàn hảo.

    Args:
        image: Ảnh uint8 (grayscale hoặc color BGR).
        d: Đường kính vùng lân cận điểm ảnh.
        sigma_color: Độ lệch chuẩn trong không gian màu/cường độ.
        sigma_space: Độ lệch chuẩn trong không gian tọa độ điểm ảnh.

    Returns:
        np.ndarray: Ảnh uint8 đã lọc song phương.
    """
    validate_image(image)
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
