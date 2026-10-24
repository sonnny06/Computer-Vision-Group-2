"""Hàm đọc, kiểm tra và hiển thị ảnh bằng OpenCV (TV1).

OpenCV đọc ảnh màu theo thứ tự kênh BGR, còn Matplotlib hiển thị theo RGB.
"""
import cv2
import matplotlib.pyplot as plt


def read_image(path):
    """Đọc ảnh bằng cv2.imread. Trả về None nếu không đọc được (sai đường dẫn, file hỏng)."""
    return cv2.imread(str(path))


def get_image_info(image):
    """Trả về dict thông tin ảnh: shape, height, width, channels, dtype."""
    if image is None:
        raise ValueError("Ảnh rỗng (None): không đọc được ảnh")
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {
        "shape": image.shape,
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": image.dtype,
    }


def bgr_to_rgb(image):
    """Đổi BGR -> RGB để Matplotlib hiển thị đúng màu."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(image, title="", is_bgr=True, figsize=(7, 4)):
    """Hiển thị ảnh bằng Matplotlib; tự đổi BGR -> RGB nếu is_bgr=True."""
    plt.figure(figsize=figsize)
    plt.imshow(bgr_to_rgb(image) if is_bgr else image)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Phần Pillow (TV2)
# ---------------------------------------------------------------------------
import os
from PIL import Image


def read_image_pillow(path):
    """Đọc ảnh bằng Pillow. Ném FileNotFoundError nếu file không tồn tại."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    return Image.open(path)


def get_pillow_info(img):
    """Thông tin ảnh Pillow: size (width, height), mode, format."""
    return {"size": img.size, "width": img.width, "height": img.height,
            "mode": img.mode, "format": img.format}


def convert_format(src_path, dst_path, **save_kwargs):
    """Đọc ảnh và lưu sang định dạng khác (định dạng suy ra từ đuôi file của dst_path)."""
    img = read_image_pillow(src_path)
    # JPEG không hỗ trợ kênh alpha nên cần đổi sang RGB trước khi lưu thành .jpg
    if str(dst_path).lower().endswith((".jpg", ".jpeg")) and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.save(dst_path, **save_kwargs)
    return dst_path


def file_size_kb(path):
    """Dung lượng file (KB)."""
    return os.path.getsize(path) / 1024


# ---------------------------------------------------------------------------
# Chọn ảnh đầu vào
# ---------------------------------------------------------------------------
from pathlib import Path


def find_input_image(preferred=None, folder="../data/input"):
    """Trả về đường dẫn ảnh đầu vào.

    Ưu tiên file `preferred` trong `folder`; nếu không có thì lấy ảnh đầu tiên
    (theo thứ tự tên) trong thư mục. Báo lỗi rõ ràng nếu thư mục không có ảnh.
    """
    folder = Path(folder)
    if preferred and (folder / preferred).exists():
        return folder / preferred
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"Không có ảnh nào trong {folder.resolve()}")
    if preferred:
        print(f"Không thấy '{preferred}', dùng ảnh: {files[0].name}")
    return files[0]
