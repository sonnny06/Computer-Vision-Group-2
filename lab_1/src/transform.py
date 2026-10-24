"""Hàm crop và resize ảnh bằng OpenCV/NumPy (TV5).

Tọa độ ảnh: gốc (0, 0) ở góc trên-trái, x là cột (sang phải), y là hàng (xuống dưới).
Slicing NumPy viết theo thứ tự image[y1:y2, x1:x2] (hàng trước, cột sau).
"""
import cv2


def crop_image(image, x1, y1, x2, y2):
    """Cắt vùng chữ nhật [x1:x2] x [y1:y2]. Kiểm tra tọa độ hợp lệ.

    Lưu ý: slicing trả về VIEW của ảnh gốc; dùng .copy() để tách riêng bộ nhớ.
    """
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Tọa độ crop không hợp lệ cho ảnh {w}x{h}: x({x1},{x2}) y({y1},{y2})")
    return image[y1:y2, x1:x2].copy()


def resize_by_scale(image, fx, fy=None, interpolation=None):
    """Resize theo tỷ lệ. fy mặc định = fx để giữ nguyên tỷ lệ khung hình."""
    fy = fx if fy is None else fy
    if interpolation is None:
        interpolation = cv2.INTER_AREA if (fx < 1 and fy < 1) else cv2.INTER_LINEAR
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def resize_fixed(image, width, height, interpolation=cv2.INTER_LINEAR):
    """Resize về kích thước cố định (width, height). Có thể làm đổi tỷ lệ khung hình."""
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_keep_ratio(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """Resize theo MỘT cạnh (width hoặc height), cạnh còn lại tự tính để giữ tỷ lệ."""
    h, w = image.shape[:2]
    if width is not None and height is None:
        new_w, new_h = int(width), round(h * width / w)
    elif height is not None and width is None:
        new_w, new_h = round(w * height / h), int(height)
    else:
        raise ValueError("Chỉ truyền đúng một trong hai: width hoặc height")
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def aspect_ratio(image):
    """Tỷ lệ khung hình width / height."""
    h, w = image.shape[:2]
    return w / h
