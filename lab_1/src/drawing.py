"""Hàm tiện ích vẽ hình cơ bản và thêm văn bản lên ảnh bằng OpenCV.

Lưu ý: OpenCV dùng màu theo thứ tự BGR, gốc tọa độ (0, 0) ở góc TRÊN-TRÁI,
x tăng sang phải, y tăng xuống dưới.
"""
import cv2

# Màu theo BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def load_image(path):
    """Đọc ảnh; báo lỗi rõ ràng nếu không đọc được."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return image


def draw_line(image, pt1, pt2, color=BLUE, thickness=3):
    """Vẽ đường thẳng từ pt1 đến pt2 (x, y). Trả về ảnh mới, không sửa ảnh gốc."""
    out = image.copy()
    cv2.line(out, pt1, pt2, color, thickness)
    return out


def draw_rectangle(image, top_left, bottom_right, color=GREEN, thickness=3):
    """Vẽ hình chữ nhật. thickness = -1 để tô đặc."""
    out = image.copy()
    cv2.rectangle(out, top_left, bottom_right, color, thickness)
    return out


def draw_circle(image, center, radius, color=RED, thickness=3):
    """Vẽ hình tròn. thickness = -1 để tô đặc."""
    out = image.copy()
    cv2.circle(out, center, radius, color, thickness)
    return out


def draw_text(image, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1.0, color=WHITE, thickness=2):
    """Thêm văn bản; org là góc DƯỚI-TRÁI của dòng chữ. Không hỗ trợ dấu tiếng Việt."""
    out = image.copy()
    cv2.putText(out, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return out


def draw_all(image, text="Computer Vision"):
    """Ảnh tổng hợp: line + rectangle + circle + text (tọa độ co theo kích thước ảnh)."""
    h, w = image.shape[:2]
    out = draw_line(image, (int(0.06 * w), int(0.10 * h)), (int(0.40 * w), int(0.10 * h)), BLUE, 3)
    out = draw_rectangle(out, (int(0.12 * w), int(0.20 * h)), (int(0.40 * w), int(0.50 * h)), GREEN, 3)
    out = draw_circle(out, (int(0.65 * w), int(0.40 * h)), int(0.12 * w), RED, 3)
    out = draw_text(out, text, (int(0.06 * w), int(0.85 * h)), font_scale=max(0.6, w / 800), thickness=2)
    return out
