"""Module sharpen_custom.py - Cài đặt các bộ lọc làm sắc nét ảnh và ma trận lọc tùy chỉnh.

Phần phụ trách của Thành viên 4 (TV4 - Trí):
- Cài đặt các toán tử làm sắc nét ảnh:
  + Bộ lọc Laplacian Sharpening (4 lân cận và 8 lân cận).
  + Kỹ thuật làm sắc nét Unsharp Masking (chuẩn công nghiệp với ngưỡng threshold).
  + Kỹ thuật lọc High-Boost Filtering (khuếch đại thành phần tần số cao với hệ số A > 1).
  + Ma trận lọc tùy chỉnh (Custom Kernels): Chạm nổi (Emboss), Sắc nét đường chéo (Diagonal), Tăng cường biên (Edge Enhance).
- Hỗ trợ cả ảnh thang xám (2D) và ảnh màu BGR (3D).
- Kiểm tra tính hợp lệ của ảnh đầu vào và xử lý ngoại lệ (Input validation).
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
    """Kiểm tra tính hợp lệ của ảnh đầu vào.

    Args:
        image: Mảng NumPy biểu diễn ảnh (2D thang xám hoặc 3D BGR).

    Returns:
        np.ndarray: Ảnh hợp lệ.

    Raises:
        TypeError: Nếu đầu vào không phải là numpy.ndarray hoặc dtype không phải uint8.
        ValueError: Nếu ảnh rỗng hoặc không phải định dạng 2D / 3D.
    """
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


def validate_ksize(ksize: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Kiểm tra và chuẩn hóa kích thước kernel (khổ chập)."""
    if isinstance(ksize, int):
        kw, kh = ksize, ksize
    elif isinstance(ksize, (tuple, list)) and len(ksize) == 2:
        kw, kh = ksize[0], ksize[1]
    else:
        raise TypeError(f"ksize phải là số nguyên int hoặc tuple 2 phần tử, nhận được: {ksize}")

    if kw <= 0 or kh <= 0:
        raise ValueError(f"Kích thước kernel phải lớn hơn 0, nhận được ({kw}, {kh}).")
    if kw % 2 == 0 or kh % 2 == 0:
        raise ValueError(f"Kích thước kernel phải là số lẻ, nhận được ({kw}, {kh}).")

    return (kw, kh)


# ==============================================================================
# 2. KHỞI TẠO MA TRẬN KERNEL LÀM SẮC NÉT VÀ TÙY BIẾN
# ==============================================================================

def create_laplacian_kernel(kernel_type: int = 1) -> np.ndarray:
    """Tạo ma trận mặt nạ làm sắc nét Laplacian 3x3.

    Args:
        kernel_type:
            - 1: Laplacian 4 lân cận (tâm 5, xung quanh -1):
                 [[ 0, -1,  0],
                  [-1,  5, -1],
                  [ 0, -1,  0]]
            - 2: Laplacian 8 lân cận (tâm 9, xung quanh -1, làm sắc nét mạnh hơn):
                 [[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]]
            - 3: Laplacian đạo hàm bậc hai thuần túy (tâm -4, tổng = 0):
                 [[ 0,  1,  0],
                  [ 1, -4,  1],
                  [ 0,  1,  0]]

    Returns:
        np.ndarray: Kernel 3x3 float32.
    """
    if kernel_type == 1:
        return np.array([
            [0.0, -1.0, 0.0],
            [-1.0, 5.0, -1.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float32)
    elif kernel_type == 2:
        return np.array([
            [-1.0, -1.0, -1.0],
            [-1.0,  9.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=np.float32)
    elif kernel_type == 3:
        return np.array([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=np.float32)
    else:
        raise ValueError(f"kernel_type không hợp lệ ({kernel_type}), phải là 1, 2 hoặc 3.")


def create_custom_kernel(name: str = "emboss") -> np.ndarray:
    """Tạo các ma trận kernel tùy biến đặc thù trong xử lý ảnh.

    Args:
        name: Tên loại kernel ('emboss', 'diagonal_sharpen', 'edge_enhance', 'ridge_detect').

    Returns:
        np.ndarray: Kernel 3x3 float32.
    """
    key = name.lower().strip()
    if key == "emboss":
        # Hiệu ứng chạm nổi nổi khối theo hướng chéo 45 độ
        return np.array([
            [-2.0, -1.0, 0.0],
            [-1.0,  1.0, 1.0],
            [ 0.0,  1.0, 2.0]
        ], dtype=np.float32)
    elif key == "diagonal_sharpen":
        # Làm sắc nét ưu tiên các cạnh chéo
        return np.array([
            [-1.0,  0.0, -1.0],
            [ 0.0,  5.0,  0.0],
            [-1.0,  0.0, -1.0]
        ], dtype=np.float32)
    elif key == "edge_enhance":
        # Tăng cường đường viền nhẹ nhàng
        return np.array([
            [ 0.0,  0.0,  0.0],
            [-1.0,  2.0,  0.0],
            [ 0.0,  0.0,  0.0]
        ], dtype=np.float32)
    elif key == "ridge_detect":
        # Phát hiện gờ rãnh chi tiết cao
        return np.array([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Tên kernel không hỗ trợ: '{name}'. Chọn 'emboss', 'diagonal_sharpen', 'edge_enhance', 'ridge_detect'.")


# ==============================================================================
# 3. CÁC HÀM LÀM SẮC NÉT VÀ LỌC TÙY CHỈNH
# ==============================================================================

def apply_laplacian_sharpen(
    image: np.ndarray,
    kernel_type: int = 1,
    alpha: float = 1.0,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Làm sắc nét ảnh bằng toán tử đạo hàm bậc hai Laplacian.

    Công thức:
        g(x, y) = f(x, y) + alpha * [kernel * f(x, y)]

    Args:
        image: Mảng ảnh uint8 (2D hoặc 3D).
        kernel_type: 1 (4-connectivity) hoặc 2 (8-connectivity).
        alpha: Hệ số cường độ làm sắc nét (mặc định 1.0).
        border_type: Kiểu xử lý biên OpenCV.

    Returns:
        np.ndarray: Ảnh uint8 sau khi làm sắc nét.
    """
    validate_image(image)
    kernel = create_laplacian_kernel(kernel_type)

    if alpha == 1.0 and kernel_type in (1, 2):
        # Trực tiếp tích chập kernel đã bao gồm ảnh gốc
        filtered = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=border_type)
        return filtered

    # Nếu alpha khác 1.0, tính toán đạo hàm Laplacian thuần túy rồi cộng có trọng số
    lap_kernel = create_laplacian_kernel(kernel_type=3)
    lap = cv2.filter2D(image.astype(np.float32), ddepth=-1, kernel=lap_kernel, borderType=border_type)
    # Vì tâm là -4 (âm), phép làm sắc nét là: f(x, y) - alpha * lap(x, y)
    sharp = image.astype(np.float32) - alpha * lap
    return np.clip(np.round(sharp), 0, 255).astype(np.uint8)


def apply_unsharp_mask(
    image: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: float = 0.0,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Kỹ thuật làm sắc nét Unsharp Masking chuẩn công nghiệp.

    Quy trình:
        1. Làm mờ ảnh bằng Gaussian blur để trích xuất thành phần tần số thấp: f_blur = Blur(f)
        2. Tạo mặt nạ chi tiết (Unsharp mask): mask = f - f_blur
        3. Áp dụng ngưỡng threshold để tránh khuếch đại nhiễu ở vùng phẳng:
           mask[abs(mask) < threshold] = 0
        4. Cộng mặt nạ chi tiết trở lại ảnh gốc: g = f + amount * mask

    Args:
        image: Mảng ảnh uint8 (2D hoặc 3D BGR).
        ksize: Kích thước kernel Gaussian lẻ (mặc định 5).
        sigma: Độ lệch chuẩn Gaussian (mặc định 1.0).
        amount: Hệ số khuếch đại chi tiết (mặc định 1.5).
        threshold: Ngưỡng loại trừ nhiễu (mặc định 0.0).
        border_type: Kiểu xử lý biên.

    Returns:
        np.ndarray: Ảnh uint8 đã làm sắc nét chi tiết cạnh.
    """
    validate_image(image)
    kw, kh = validate_ksize(ksize)

    blurred = cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma, sigmaY=sigma, borderType=border_type)
    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)

    mask = img_f - blur_f

    if threshold > 0.0:
        low_contrast_mask = np.abs(mask) < threshold
        mask[low_contrast_mask] = 0.0

    sharp = img_f + amount * mask
    return np.clip(np.round(sharp), 0, 255).astype(np.uint8)


def apply_highboost_filter(
    image: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = 5,
    sigma: float = 1.0,
    A: float = 1.5,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Bộ lọc High-Boost Filtering (khuếch đại ảnh gốc kết hợp làm sắc nét).

    Công thức:
        f_hb = A * f(x, y) - f_blur(x, y) = (A - 1) * f(x, y) + [f(x, y) - f_blur(x, y)]
        - Nếu A = 1.0: Trở thành Unsharp Masking thông thường.
        - Nếu A > 1.0: Một phần của ảnh gốc được giữ lại và khuếch đại, giúp ảnh sáng rõ nét.

    Args:
        image: Ảnh đầu vào uint8.
        ksize: Khổ kernel Gauss.
        sigma: Độ lệch chuẩn Gauss.
        A: Hệ số khuếch đại khuếch đại (A >= 1.0, mặc định 1.5).
        border_type: Kiểu xử lý viền.

    Returns:
        np.ndarray: Ảnh uint8 sau khi lọc High-Boost.
    """
    validate_image(image)
    if A < 1.0:
        raise ValueError(f"Hệ số High-Boost A phải >= 1.0, nhận được: {A}")

    kw, kh = validate_ksize(ksize)
    blurred = cv2.GaussianBlur(image, (kw, kh), sigmaX=sigma, sigmaY=sigma, borderType=border_type)

    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)

    highboost = A * img_f - blur_f
    return np.clip(np.round(highboost), 0, 255).astype(np.uint8)


def apply_custom_kernel(
    image: np.ndarray,
    kernel: np.ndarray,
    delta: float = 0.0,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Tích chập ảnh với ma trận kernel tùy biến tự định nghĩa.

    Args:
        image: Mảng ảnh uint8 (2D hoặc 3D).
        kernel: Mảng 2D float32.
        delta: Giá trị cộng thêm vào pixel (thường dùng 128 cho Emboss để tránh giá trị âm).
        border_type: Kiểu biên.

    Returns:
        np.ndarray: Ảnh uint8 sau khi lọc.
    """
    validate_image(image)
    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2:
        raise ValueError("Kernel phải là mảng numpy.ndarray 2 chiều.")

    filtered = cv2.filter2D(image.astype(np.float32), ddepth=-1, kernel=kernel, borderType=border_type)
    if delta != 0.0:
        filtered += delta

    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)
