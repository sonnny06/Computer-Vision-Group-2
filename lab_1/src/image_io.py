"""
image_io.py
-----------
Tiện ích đọc / ghi / chuyển đổi định dạng ảnh bằng Pillow và OpenCV.
Được sử dụng bởi notebook 02_image_io_pillow.ipynb.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

SUPPORTED_READ_FORMATS: List[str] = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".gif", ".ppm", ".pgm",
]

SUPPORTED_WRITE_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


# ---------------------------------------------------------------------------
# Đọc ảnh – Pillow
# ---------------------------------------------------------------------------

def pillow_open(path: str | Path) -> Image.Image:
    """Mở ảnh bằng Pillow và trả về đối tượng Image.

    Args:
        path: Đường dẫn đến file ảnh.

    Returns:
        PIL.Image.Image object.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        OSError: Nếu Pillow không thể mở file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return Image.open(path)


def get_image_info(img: Image.Image) -> Dict:
    """Trích xuất metadata của ảnh Pillow.

    Args:
        img: PIL.Image.Image đã mở.

    Returns:
        dict chứa: format, mode, size, width, height, channels, n_pixels.
    """
    width, height = img.size
    mode = img.mode

    # Xác định số kênh từ mode
    mode_channels = {
        "1": 1, "L": 1, "P": 1,
        "RGB": 3, "RGBA": 4,
        "CMYK": 4, "YCbCr": 3, "LAB": 3,
        "HSV": 3, "I": 1, "F": 1,
        "LA": 2, "RGBa": 4, "La": 2, "PA": 2,
    }
    channels = mode_channels.get(mode, len(img.getbands()))

    return {
        "format": img.format,          # "JPEG", "PNG", … (None nếu từ numpy)
        "mode": mode,                   # "RGB", "L", "RGBA", …
        "size": img.size,               # (width, height)
        "width": width,
        "height": height,
        "channels": channels,
        "n_pixels": width * height,
    }


def print_image_info(img: Image.Image, label: str = "") -> None:
    """In metadata của ảnh ra console theo định dạng đẹp.

    Args:
        img: PIL.Image.Image.
        label: Nhãn tiêu đề tuỳ chọn.
    """
    info = get_image_info(img)
    header = f"=== {label} ===" if label else "=== Image Info ==="
    print(header)
    print(f"  Format   : {info['format']}")
    print(f"  Mode     : {info['mode']}")
    print(f"  Size     : {info['size']}  ->  width={info['width']}px, height={info['height']}px")
    print(f"  Channels : {info['channels']}")
    print(f"  Pixels   : {info['n_pixels']:,}")
    print()


# ---------------------------------------------------------------------------
# Đọc ảnh – OpenCV
# ---------------------------------------------------------------------------

def cv2_open(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Đọc ảnh bằng OpenCV (BGR), hỗ trợ tốt đường dẫn Unicode trên Windows.

    Args:
        path: Đường dẫn file ảnh.
        flags: Cờ đọc ảnh của OpenCV (mặc định cv2.IMREAD_COLOR).

    Returns:
        numpy.ndarray (H, W, C) BGR.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu OpenCV không đọc được.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    # Trên Windows, cv2.imread() lỗi với đường dẫn Unicode/tiếng Việt.
    # Sử dụng np.fromfile + cv2.imdecode để tương thích 100%.
    img = None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception:
        pass

    if img is None:
        img = cv2.imread(str(path), flags)

    if img is None:
        raise ValueError(f"OpenCV không thể đọc: {path}")
    return img


def get_cv2_info(img: np.ndarray, path: Optional[str | Path] = None) -> Dict:
    """Trích xuất metadata từ mảng NumPy (OpenCV).

    Args:
        img: ndarray từ cv2.imread.
        path: (tuỳ chọn) đường dẫn gốc để lấy phần mở rộng.

    Returns:
        dict chứa: dtype, shape, height, width, channels, format.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    fmt = Path(path).suffix.upper().lstrip(".") if path else "N/A"
    return {
        "dtype": str(img.dtype),
        "shape": img.shape,
        "height": h,
        "width": w,
        "channels": channels,
        "format": fmt,
    }


# ---------------------------------------------------------------------------
# Chuyển đổi định dạng – Pillow
# ---------------------------------------------------------------------------

def convert_format_pillow(
    src: str | Path,
    dst: str | Path,
    quality: int = 95,
    optimize: bool = True,
) -> Path:
    """Chuyển đổi định dạng ảnh bằng Pillow.

    Tự động xử lý:
    - PNG (RGBA) → JPG: chuyển sang RGB trước.
    - Tạo thư mục đích nếu chưa tồn tại.

    Args:
        src: Đường dẫn file nguồn.
        dst: Đường dẫn file đích (phần mở rộng quyết định định dạng).
        quality: Chất lượng JPEG (1–95).
        optimize: Tối ưu file PNG/JPEG.

    Returns:
        Path của file đã lưu.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = pillow_open(src)

    dst_ext = dst.suffix.lower()

    # JPEG không hỗ trợ kênh alpha → chuyển sang RGB
    if dst_ext in (".jpg", ".jpeg") and img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif dst_ext in (".jpg", ".jpeg") and img.mode != "RGB":
        img = img.convert("RGB")

    save_kwargs: Dict = {}
    if dst_ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = optimize
    elif dst_ext == ".png":
        save_kwargs["optimize"] = optimize

    img.save(dst, **save_kwargs)
    return dst


def batch_convert_pillow(
    src_dir: str | Path,
    dst_dir: str | Path,
    src_ext: str = ".jpg",
    dst_ext: str = ".png",
    quality: int = 95,
) -> List[Path]:
    """Chuyển đổi hàng loạt ảnh trong một thư mục.

    Args:
        src_dir: Thư mục nguồn.
        dst_dir: Thư mục đích.
        src_ext: Phần mở rộng nguồn (vd: '.jpg').
        dst_ext: Phần mở rộng đích (vd: '.png').
        quality: Chất lượng JPEG.

    Returns:
        Danh sách Path của các file đã lưu.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for src_file in sorted(src_dir.glob(f"*{src_ext}")):
        dst_file = dst_dir / (src_file.stem + dst_ext)
        out = convert_format_pillow(src_file, dst_file, quality=quality)
        saved.append(out)

    return saved


# ---------------------------------------------------------------------------
# So sánh Pillow và OpenCV
# ---------------------------------------------------------------------------

def benchmark_read(
    path: str | Path,
    n_runs: int = 10,
) -> Dict:
    """Đo thời gian đọc ảnh của Pillow và OpenCV.

    Args:
        path: Đường dẫn file ảnh.
        n_runs: Số lần lặp để lấy trung bình.

    Returns:
        dict: {pillow_ms, opencv_ms, winner}.
    """
    path = str(path)

    # --- Pillow ---
    t0 = time.perf_counter()
    for _ in range(n_runs):
        img_pil = Image.open(path)
        img_pil.load()          # force decode
    pillow_ms = (time.perf_counter() - t0) / n_runs * 1000

    # --- OpenCV ---
    t0 = time.perf_counter()
    for _ in range(n_runs):
        data = np.fromfile(path, dtype=np.uint8)
        _ = cv2.imdecode(data, cv2.IMREAD_COLOR)
    opencv_ms = (time.perf_counter() - t0) / n_runs * 1000

    winner = "Pillow" if pillow_ms < opencv_ms else "OpenCV"

    return {
        "pillow_ms": round(pillow_ms, 3),
        "opencv_ms": round(opencv_ms, 3),
        "winner": winner,
    }


def compare_pillow_opencv(path: str | Path) -> Dict:
    """So sánh toàn diện Pillow vs OpenCV cho một ảnh.

    Trả về dict gồm:
    - Thông tin ảnh từ mỗi thư viện.
    - Benchmark đọc ảnh.
    - So sánh giá trị pixel (Pillow RGB vs OpenCV BGR→RGB).

    Args:
        path: Đường dẫn file ảnh.

    Returns:
        dict kết quả so sánh.
    """
    path = Path(path)

    # Pillow
    img_pil = pillow_open(path)
    pil_info = get_image_info(img_pil)

    # OpenCV
    img_cv = cv2_open(path)
    cv_info = get_cv2_info(img_cv, path)

    # So sánh pixel: chuyển OpenCV BGR → RGB
    arr_pil = np.array(img_pil.convert("RGB"))
    arr_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    pixel_match = np.array_equal(arr_pil, arr_cv)
    max_diff = int(np.max(np.abs(arr_pil.astype(int) - arr_cv.astype(int))))

    # Benchmark
    bench = benchmark_read(path)

    return {
        "pillow": pil_info,
        "opencv": cv_info,
        "pixel_match": pixel_match,
        "max_pixel_diff": max_diff,
        "benchmark": bench,
    }


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Chuyển PIL Image sang OpenCV ndarray (BGR / BGRA / Gray)."""
    arr = np.array(img)
    if img.mode == "RGB":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif img.mode == "RGBA":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif img.mode in ("L", "1"):
        return arr
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Chuyển OpenCV ndarray (BGR / BGRA / Gray) sang PIL Image."""
    if img.ndim == 2:
        return Image.fromarray(img, mode="L")
    elif img.ndim == 3:
        if img.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(img)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Tạo thư mục output nếu chưa tồn tại và trả về Path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
