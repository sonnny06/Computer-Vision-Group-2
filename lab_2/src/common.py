"""Module common.py - Các hàm tiện ích dùng chung cho Bài Lab 2 (Chương 2).

Bao gồm:
- Đọc và lưu ảnh an toàn với đường dẫn tiếng Việt Unicode trên Windows.
- Chuyển đổi không gian màu BGR sang RGB để hiển thị bằng Matplotlib.
- Kiểm tra tính hợp lệ của mảng ảnh đầu vào.
- Tiện ích hiển thị nhiều ảnh cạnh nhau (side-by-side).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """Kiểm tra tính hợp lệ của mảng ảnh đầu vào."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Ảnh phải là mảng numpy.ndarray, nhận được: {type(image)}")
    if image.dtype != np.uint8:
        raise TypeError(f"Ảnh phải có kiểu dữ liệu uint8, nhận được: {image.dtype}")
    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Ảnh phải có 2 chiều (ảnh xám) hoặc 3 chiều (ảnh màu), nhận được: {image.ndim} chiều.")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Ảnh màu phải có 1, 3 hoặc 4 kênh màu, nhận được: {image.shape[2]} kênh.")
    return image


def read_image(path: Union[str, Path], as_gray: bool = False) -> np.ndarray:
    """Đọc ảnh từ đường dẫn hỗ trợ an toàn Unicode tiếng Việt trên Windows.

    Args:
        path: Đường dẫn tới file ảnh.
        as_gray: Nếu True đọc ảnh thang xám, nếu False đọc ảnh màu BGR.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {p}")

    data = np.fromfile(str(p), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)

    if img is None:
        raise ValueError(f"Không thể giải mã file ảnh: {p}")
    return img


def save_image(path: Union[str, Path], image: np.ndarray) -> bool:
    """Lưu ảnh vào đường dẫn hỗ trợ Unicode trên Windows."""
    validate_image(image)
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() if p.suffix else ".png"

    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise OSError(f"Không thể mã hóa ảnh sang định dạng: {ext}")

    with open(p, "wb") as f:
        f.write(buffer)
    return True


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển đổi ảnh BGR của OpenCV sang RGB để hiển thị bằng Matplotlib."""
    validate_image(image_bgr)
    if image_bgr.ndim == 2:
        return image_bgr
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
        return image_bgr[:, :, 0]
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


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


def show_images(
    images: List[np.ndarray],
    titles: List[str] | None = None,
    cols: int = 4,
    figsize: Tuple[int, int] | None = None,
    cmap: str = "gray",
    save_path: Union[str, Path] | None = None,
) -> None:
    """Hiển thị danh sách ảnh trên cùng một lưới bằng Matplotlib."""
    n = len(images)
    if n == 0:
        return

    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 4, rows * 4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for idx, img in enumerate(images):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3 and img.shape[2] == 1:
            ax.imshow(img[:, :, 0], cmap=cmap)
        else:
            ax.imshow(bgr_to_rgb(img))
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=11, fontweight="bold")
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        p = Path(save_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
    plt.show()
