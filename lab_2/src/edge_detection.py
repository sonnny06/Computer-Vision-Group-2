"""Sobel và Prewitt 3x3: gradient có dấu và độ lớn trên ảnh xám."""

import cv2
import numpy as np

from .common import to_grayscale


def sobel_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (Gx, Gy, magnitude), mỗi mảng float32 có shape (H, W).

    Đầu vào uint8: ảnh xám 2D, một kênh, BGR hoặc BGRA; ảnh màu được
    chuyển sang xám trước khi lọc. Không sửa ảnh đầu vào. Gx/Gy giữ dấu,
    magnitude = hypot(Gx, Gy); không ép về uint8 trước khi tính độ lớn.
    Dùng kernel 3x3 và viền phản chiếu REFLECT_101.

    Khi so sánh cùng ngưỡng với Prewitt, có thể chia các kết quả Sobel
    cho 4 và Prewitt cho 3 (hệ số đáp ứng tại cạnh bậc thang). Hàm này
    trả về đáp ứng gốc, chưa chuẩn hóa.
    """
    gray = to_grayscale(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return gx, gy, np.hypot(gx, gy)


def prewitt_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (Gx, Gy, magnitude) float32 bằng kernel Prewitt tự khai báo.

    Cùng quy ước đầu vào, viền và dấu gradient như sobel_edges. Kernel
    Gx gồm ba hàng [-1, 0, 1]; Gy là chuyển vị của Gx. Trục x tăng sang
    phải, trục y tăng xuống dưới. Kết quả giữ nguyên thang đáp ứng gốc.
    """
    gray = to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_x.T, borderType=cv2.BORDER_REFLECT_101)
    return gx, gy, np.hypot(gx, gy)
