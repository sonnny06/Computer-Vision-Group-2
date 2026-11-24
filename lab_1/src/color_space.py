"""Hàm chuyển đổi không gian màu bằng OpenCV.

Phần Grayscale (TV3). Phần HSV/LAB (TV4) sẽ được thêm bên dưới.
"""
import cv2
import numpy as np


def bgr_to_gray(image):
    """Chuyển ảnh BGR (H, W, 3) sang grayscale (H, W) bằng cv2.cvtColor."""
    if image is None:
        raise ValueError("Ảnh rỗng (None)")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gray_manual(image):
    """Tự tính grayscale theo công thức trọng số: Y = 0.299 R + 0.587 G + 0.114 B."""
    b, g, r = [image[:, :, i].astype(np.float32) for i in range(3)]
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).round().astype(np.uint8)


def gray_mean(image):
    """Grayscale kiểu trung bình đơn giản 3 kênh (để so sánh, không phải cách chuẩn)."""
    return image.astype(np.float32).mean(axis=2).round().astype(np.uint8)


def intensity_stats(gray):
    """Thống kê cường độ pixel: min, max, mean."""
    return {"min": int(gray.min()), "max": int(gray.max()), "mean": float(gray.mean())}


# ---------------------------------------------------------------------------
# Phần HSV và LAB (TV4)
# ---------------------------------------------------------------------------
def bgr_to_hsv(image):
    """BGR -> HSV (ảnh 8-bit: H trong 0-179, S và V trong 0-255)."""
    if image is None:
        raise ValueError("Ảnh rỗng (None)")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image):
    """BGR -> LAB (ảnh 8-bit: L, a, b đều được co về 0-255, a và b lệch 128)."""
    if image is None:
        raise ValueError("Ảnh rỗng (None)")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def hsv_to_bgr(hsv):
    """HSV -> BGR để hiển thị hoặc lưu đúng màu."""
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def lab_to_bgr(lab):
    """LAB -> BGR để hiển thị hoặc lưu đúng màu."""
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def channel_ranges(img, names):
    """Trả về dict {tên kênh: (min, max)} cho ảnh nhiều kênh."""
    return {n: (int(img[:, :, i].min()), int(img[:, :, i].max())) for i, n in enumerate(names)}
