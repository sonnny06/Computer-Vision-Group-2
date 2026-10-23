


# """Module negative_threshold.py - Biến đổi âm bản và cắt ngưỡng (TV3 - Văn Tôn).

# - Mục I.3: Đảo ngược âm bản (Image Negative): pixel_mới = 255 - pixel_cũ 
# - Mục I.4: Phân ngưỡng (Thresholding):
#     + Nếu pixel_cũ >= ngưỡng -> pixel_mới = 255 (màu trắng)
#     + Nếu pixel_cũ <  ngưỡng -> pixel_mới = 0   (màu đen)
# - Tích hợp tính Histogram mức xám bằng cv2.calcHist để phân tích ngưỡng.
# """

# from __future__ import annotations

# from typing import Dict, Sequence, Tuple

# import cv2
# import numpy as np



# def to_grayscale(image: np.ndarray) -> np.ndarray:
#     """Chuyển đổi không gian màu sang ảnh đa cấp xám 8-bit (256 mức xám: 0 -> 255).

#     Nếu ảnh đầu vào đã là ảnh xám (2D) thì trả về bản sao.
#     Nếu là ảnh màu BGR/BGRA thì chuyển về ảnh xám bằng cv2.cvtColor.
#     """
#     validate_image(image)
#     if image.ndim == 2:
#         return image.copy()
#     if image.shape[2] == 1:
#         return image[:, :, 0].copy()
#     if image.shape[2] == 3:
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.shape[2] == 4:
#         return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#     raise ValueError(f"Số kênh màu không hợp lệ: {image.shape}")


# def image_negative(image: np.ndarray) -> np.ndarray:
#     """Tạo ảnh âm bản theo đúng hàm mẫu trang 7 & công thức trang 6 trong bài giảng.

#     Công thức: pixel_mới = 255 - pixel_cũ
#     """
#     validate_image(image)
#     negative_image = 255 - image
#     return negative_image.astype(np.uint8)


# negative_transform = image_negative


# def validate_threshold_value(thresh: int | float, maxval: int = 255) -> Tuple[int, int]:
#     """Kiểm tra tính hợp lệ của giá trị ngưỡng trong khoảng mức xám 8-bit [0, 255]."""
#     if not isinstance(thresh, (int, float, np.number)) or isinstance(thresh, bool):
#         raise TypeError(f"Ngưỡng (thresh) phải là kiểu số, nhận được: {type(thresh)}")
#     if thresh < 0 or thresh > 255:
#         raise ValueError(f"Ngưỡng (thresh) phải nằm trong khoảng [0, 255], nhận được: {thresh}")
#     if not isinstance(maxval, (int, np.integer)) or not (0 <= int(maxval) <= 255):
#         raise ValueError(f"Giá trị maxval phải nằm trong [0, 255], nhận được: {maxval}")
#     return int(round(float(thresh))), int(maxval)


# def threshold_image(
#     image: np.ndarray,
#     thresh: int | float = 128,
#     maxval: int = 255,
#     invert: bool = False,
# ) -> np.ndarray:
#     """Phân ngưỡng chuyển ảnh xám thành ảnh nhị phân theo công thức Trang 6 bài giảng.

#     Công thức chuẩn (khi invert=False):
#     - Nếu pixel_cũ >= ngưỡng: pixel_mới = 255 (màu trắng)
#     - Nếu pixel_cũ <  ngưỡng: pixel_mới = 0   (màu đen)
#     """
#     t_val, max_val = validate_threshold_value(thresh, maxval)
#     gray = to_grayscale(image)
#     if not invert:
#         binary = np.where(gray >= t_val, max_val, 0)
#     else:
#         binary = np.where(gray >= t_val, 0, max_val)
#     return binary.astype(np.uint8)


# binary_threshold_manual = threshold_image


# def threshold_opencv(
#     image: np.ndarray,
#     thresh: int | float = 128,
#     maxval: int = 255,
#     invert: bool = False,
# ) -> np.ndarray:
#     """Phân ngưỡng bằng hàm cv2.threshold của OpenCV."""
#     t_val, max_val = validate_threshold_value(thresh, maxval)
#     gray = to_grayscale(image)
#     if t_val == 0:
#         return np.full_like(gray, 0 if invert else max_val, dtype=np.uint8)
#     flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
#     # cv2.threshold kiểm tra điều kiện (pixel > T), nên truyền (t_val - 1) để khớp với (pixel >= t_val)
#     _, binary = cv2.threshold(gray, t_val - 1, max_val, flag)
#     return binary


# def apply_multiple_thresholds(
#     image: np.ndarray,
#     thresholds: Sequence[int | float] = (80, 128, 180),
#     maxval: int = 255,
# ) -> Dict[int, np.ndarray]:
#     """Thử nghiệm nhiều giá trị ngưỡng khác nhau trên cùng một ảnh xám."""
#     if not thresholds:
#         raise ValueError("Danh sách ngưỡng (thresholds) không được rỗng.")
#     gray = to_grayscale(image)
#     results: Dict[int, np.ndarray] = {}
#     for t in thresholds:
#         t_val, _ = validate_threshold_value(t, maxval)
#         results[t_val] = threshold_image(gray, thresh=t_val, maxval=maxval)
#     return results


# def compute_grayscale_histogram(image: np.ndarray) -> np.ndarray:
#     """Tính Histogram mức xám bằng cv2.calcHist theo đúng hướng dẫn """
#     gray = to_grayscale(image)
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     return hist.ravel()


# def compute_white_black_stats(binary_image: np.ndarray) -> Dict[str, float]:
#     """Thống kê số lượng và tỷ lệ phần trăm điểm ảnh trắng (255) và đen (0)."""
#     validate_image(binary_image)
#     if binary_image.ndim != 2:
#         raise ValueError("Ảnh đầu vào phải là ảnh nhị phân 2 chiều.")
#     total = binary_image.size
#     white = int(np.count_nonzero(binary_image == 255))
#     black = int(total - white)
#     return {
#         "total_pixels": total,
#         "white_pixels": white,
#         "black_pixels": black,
#         "white_ratio": round(white * 100.0 / total, 2),
#         "black_ratio": round(black * 100.0 / total, 2),
#     }




"""Module negative_threshold.py - Phần 1: Chuyển xám và Ảnh âm bản (TV3 - Văn Tôn).
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .common import validate_image
except ImportError:
    from common import validate_image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh màu BGR hoặc BGRA sang ảnh đa cấp xám 8-bit (0 -> 255).
    Nếu ảnh đầu vào đã là ảnh xám 2 chiều, trả về bản sao của ảnh đó.
    """
    validate_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Số kênh màu không hợp lệ: {image.shape}")


def image_negative(image: np.ndarray) -> np.ndarray:
    """Tạo ảnh âm bản bằng cách đảo giá trị pixel
    Công thức: pixel_mới = 255 - pixel_cũ
    Hỗ trợ cả ảnh xám (2D) và ảnh màu (3D).
    """
    validate_image(image)
    # Trừ trực tiếp từng giá trị pixel cho 255
    negative_img = 255 - image
    return negative_img.astype(np.uint8)


def threshold_image(image: np.ndarray, thresh: int = 128, maxval: int = 255) -> np.ndarray:
    """Cắt ngưỡng nhị phân 
    - Bước 1: Chuyển ảnh màu sang ảnh xám trước khi threshold.
    - Bước 2: So sánh giá trị pixel với ngưỡng T:
      + Nếu pixel_cũ >= ngưỡng -> 255 (màu trắng)
      + Nếu pixel_cũ <  ngưỡng -> 0   (màu đen)
    """
    # Yêu cầu 1: Chuyển ảnh màu sang ảnh xám trước khi threshold
    gray = to_grayscale(image)
    
    # Thực hiện phân ngưỡng nhị phân
    binary = np.where(gray >= thresh, maxval, 0)
    return binary.astype(np.uint8)


def apply_multiple_thresholds(image: np.ndarray, thresholds: list[int] = [80, 128, 180]) -> dict[int, np.ndarray]:
    """Yêu cầu 3 & 4 của nhóm trưởng: Thử ít nhất 3 giá trị ngưỡng khác nhau 
    và quan sát sự thay đổi vùng đen/trắng.
    """
    gray = to_grayscale(image)
    results = {}
    for t in thresholds:
        results[t] = threshold_image(gray, thresh=t)
    return results


def compute_white_black_ratio(binary_image: np.ndarray) -> dict:
    """Thống kê số lượng và tỷ lệ phần trăm vùng trắng/đen để quan sát sự thay đổi."""
    validate_image(binary_image)
    total = binary_image.size
    white = int(np.count_nonzero(binary_image == 255))
    black = int(total - white)
    return {
        "white_pixels": white,
        "black_pixels": black,
        "white_percent": round(white * 100.0 / total, 2),
        "black_percent": round(black * 100.0 / total, 2),
    }