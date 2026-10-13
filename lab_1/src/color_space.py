from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def bgr_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a non-empty uint8 BGR image (H, W, 3) to grayscale (H, W).

    The input is preserved. Raises TypeError for a non-array or non-uint8
    input, and ValueError for an empty image or an invalid shape.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy.ndarray.")
    if image_bgr.dtype != np.uint8:
        raise TypeError("image_bgr must have dtype uint8.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H, W, 3).")
    if image_bgr.size == 0:
        raise ValueError("image_bgr must not be empty.")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh BGR sang RGB cho Matplotlib / Pillow."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("image_bgr phải là mảng NumPy 3 chiều.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh BGR sang không gian màu HSV."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("image_bgr phải là mảng NumPy 3 chiều.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh BGR sang không gian màu CIE L*a*b*."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("image_bgr phải là mảng NumPy 3 chiều.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)


def enhance_contrast_clahe_lab(
    image_bgr: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Cân bằng tương phản thích ứng (CLAHE) trên kênh L của không gian màu LAB."""
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def segment_color_hsv(
    image_bgr: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Phân đoạn màu sắc bằng ngưỡng HSV, trả về (mask_binary, segmented_bgr)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return mask, segmented


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Đọc ảnh BGR an toàn với đường dẫn tiếng Việt / Unicode trên Windows."""
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể giải mã ảnh: {p}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> bool:
    """Lưu ảnh an toàn với đường dẫn tiếng Việt / Unicode trên Windows."""
    from pathlib import Path
    p = Path(path)
    ext = p.suffix if p.suffix else ".png"
    p.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise OSError(f"Lỗi khi mã hóa ảnh: {p}")
    with open(p, "wb") as f:
        f.write(buffer)
    return True
