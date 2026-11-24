"""
Module tiền xử lý và kiểm tra tính hợp lệ của ảnh y tế (Medical Image Preprocessing).
Phụ trách: Trí (Image Preprocessing & Data Preparation)

Module này đảm nhận:
1. Kiểm tra tính hợp lệ của ảnh (định dạng, kích thước, ảnh rỗng, ảnh hỏng).
2. Tải ảnh linh hoạt từ nhiều nguồn (đường dẫn file, bytes, BytesIO, Streamlit UploadedFile, PIL Image, numpy array).
3. Chuẩn hóa ảnh sang hệ màu RGB (3 kênh) tương thích chuẩn cho Qwen3-VL Processor.
4. Tự động xử lý xoay ảnh theo EXIF orientation.
5. Giữ nguyên tỷ lệ khung hình (Aspect Ratio), điều chỉnh kích thước an toàn nếu ảnh vượt quá giới hạn phân giải.
6. Trả về thông báo lỗi chi tiết, thân thiện bằng tiếng Việt.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

# Danh sách định dạng ảnh hợp lệ được hỗ trợ
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Cấu hình ngưỡng kiểm tra mặc định
DEFAULT_MAX_FILE_SIZE_MB = 25.0
DEFAULT_MIN_DIMENSION = 50
DEFAULT_MAX_DIMENSION = 8192
DEFAULT_TARGET_MAX_DIM = 2048


class ImageProcessingError(ValueError):
    """Lỗi xử lý hoặc kiểm tra ảnh không hợp lệ."""
    pass


def validate_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray, Any],
    max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    min_dimension: int = DEFAULT_MIN_DIMENSION,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Kiểm tra toàn diện tính hợp lệ của ảnh đầu vào.

    Args:
        image_input: Ảnh đầu vào (đường dẫn, bytes, buffer, đối tượng PIL, mảng numpy).
        max_file_size_mb: Dung lượng tối đa cho phép (MB).
        min_dimension: Chiều dài/rộng tối thiểu (pixel).
        max_dimension: Chiều dài/rộng tối đa cho phép (pixel).

    Returns:
        Tuple gồm (is_valid, message, metadata):
        - is_valid (bool): True nếu ảnh hợp lệ, False nếu không hợp lệ.
        - message (str): Thông báo bằng tiếng Việt (mô tả thành công hoặc nguyên nhân lỗi).
        - metadata (dict): Thông tin cơ bản của ảnh nếu đọc thành công.
    """
    metadata: Dict[str, Any] = {}

    if image_input is None:
        return False, "Dữ liệu ảnh không được để trống (None).", metadata

    # 1. Kiểm tra nếu đầu vào là đường dẫn file
    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            return False, f"Tệp tin ảnh không tồn tại: {file_path}", metadata
        if not file_path.is_file():
            return False, f"Đường dẫn không phải là tệp tin: {file_path}", metadata

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Định dạng tệp '{ext}' không được hỗ trợ. Chỉ chấp nhận các định dạng: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
                metadata,
            )

        file_size_bytes = file_path.stat().st_size
        metadata["file_path"] = str(file_path)
        metadata["file_size_bytes"] = file_size_bytes
        metadata["file_name"] = file_path.name

        if file_size_bytes == 0:
            return False, "Tệp ảnh rỗng (dung lượng 0 bytes).", metadata

        max_bytes = max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            return (
                False,
                f"Dung lượng tệp ({file_size_bytes / (1024 * 1024):.2f} MB) vượt quá giới hạn cho phép ({max_file_size_mb:.1f} MB).",
                metadata,
            )

    # 2. Kiểm tra nếu đầu vào là bytes
    elif isinstance(image_input, bytes):
        if len(image_input) == 0:
            return False, "Dữ liệu bytes của ảnh rỗng (0 bytes).", metadata
        metadata["file_size_bytes"] = len(image_input)
        if len(image_input) > max_file_size_mb * 1024 * 1024:
            return (
                False,
                f"Dung lượng dữ liệu ({len(image_input) / (1024 * 1024):.2f} MB) vượt quá giới hạn cho phép.",
                metadata,
            )

    # 3. Kiểm tra nếu là UploadedFile từ Streamlit (có name, size, getvalue)
    elif hasattr(image_input, "name") and hasattr(image_input, "size"):
        file_name = getattr(image_input, "name", "")
        file_size = getattr(image_input, "size", 0)
        metadata["file_name"] = file_name
        metadata["file_size_bytes"] = file_size

        ext = os.path.splitext(file_name)[1].lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Định dạng tệp '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
                metadata,
            )
        if file_size == 0:
            return False, "Tệp tải lên rỗng (0 bytes).", metadata
        if file_size > max_file_size_mb * 1024 * 1024:
            return (
                False,
                f"Dung lượng tệp ({file_size / (1024 * 1024):.2f} MB) vượt quá mức cho phép.",
                metadata,
            )

    # 4. Thử mở và kiểm tra tính toàn vẹn của cấu trúc ảnh
    try:
        raw_pil = _to_pil_image(image_input)
    except Exception as exc:
        return False, f"Tệp tin bị hỏng hoặc không phải là dữ liệu hình ảnh hợp lệ: {exc}", metadata

    width, height = raw_pil.size
    metadata.update(
        {
            "width": width,
            "height": height,
            "original_mode": raw_pil.mode,
            "original_format": raw_pil.format,
            "aspect_ratio": round(width / max(height, 1), 4),
        }
    )

    if width < min_dimension or height < min_dimension:
        return (
            False,
            f"Kích thước ảnh quá nhỏ ({width}x{height}). Yêu cầu tối thiểu {min_dimension}x{min_dimension} pixels.",
            metadata,
        )

    if width > max_dimension or height > max_dimension:
        return (
            False,
            f"Kích thước ảnh quá lớn ({width}x{height}). Giới hạn tối đa là {max_dimension}x{max_dimension} pixels.",
            metadata,
        )

    return True, "Ảnh hợp lệ.", metadata


def _to_pil_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray, Any]
) -> Image.Image:
    """Chuyển đổi các dạng đầu vào về đối tượng PIL Image."""
    if isinstance(image_input, Image.Image):
        # Trả về bản copy để tránh làm thay đổi ảnh gốc
        return image_input.copy()

    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
        img.load()  # Nạp toàn bộ dữ liệu vào bộ nhớ
        return img

    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
        img.load()
        return img

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        img = Image.open(image_input)
        img.load()
        return img

    # Streamlit UploadedFile hoặc đối tượng có hàm read/getvalue
    if hasattr(image_input, "getvalue"):
        data = image_input.getvalue()
        img = Image.open(io.BytesIO(data))
        img.load()
        return img

    if hasattr(image_input, "read"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        data = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        img = Image.open(io.BytesIO(data))
        img.load()
        return img

    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ImageProcessingError("Mảng numpy của ảnh rỗng.")
        if image_input.ndim == 2:
            return Image.fromarray(image_input)
        if image_input.ndim == 3:
            return Image.fromarray(image_input)
        raise ImageProcessingError(f"Số chiều của mảng numpy không hợp lệ: {image_input.ndim}")

    raise ImageProcessingError(f"Kiểu dữ liệu đầu vào không được hỗ trợ: {type(image_input)}")


def fix_orientation(image: Image.Image) -> Image.Image:
    """
    Tự động xoay ảnh về đúng chiều dựa trên thẻ EXIF Orientation (nếu có).
    Hữu ích với ảnh chụp X-quang hoặc quét scan từ máy quang học.
    """
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Chuẩn hóa ảnh về hệ màu RGB (3 kênh).
    - Ảnh Grayscale (L): chuyển thành 3 kênh bằng nhau để processor VLM đọc đúng cấu trúc.
    - Ảnh RGBA: hòa trộn nền trắng tự nhiên thay vì để kênh trong suốt gây lỗi tokenization.
    - Ảnh CMYK / P: chuyển về RGB chuẩn.
    """
    if image.mode == "RGB":
        return image

    if image.mode == "L":
        return image.convert("RGB")

    if image.mode == "RGBA":
        # Tạo ảnh nền trắng với kích thước tương ứng rồi dán ảnh RGBA lên
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Kênh Alpha làm mặt nạ
        return background

    # Các hệ màu khác (CMYK, P, 1, LAB,...)
    return image.convert("RGB")


def resize_preserving_aspect_ratio(
    image: Image.Image,
    target_max_dim: int = DEFAULT_TARGET_MAX_DIM,
) -> Image.Image:
    """
    Điều chỉnh kích thước ảnh an toàn khi vượt quá target_max_dim,
    đảm bảo TUYỆT ĐỐI KHÔNG làm méo tỷ lệ khung hình (Aspect Ratio) của ảnh y tế.
    Sử dụng thuật toán nội suy Resampling.LANCZOS chất lượng cao để bảo toàn các chi tiết mô giải phẫu.
    """
    width, height = image.size
    max_current = max(width, height)

    if max_current <= target_max_dim:
        return image

    scale = target_max_dim / float(max_current)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def preprocess_image(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray, Any],
    target_max_dim: int = DEFAULT_TARGET_MAX_DIM,
    return_metadata: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Dict[str, Any]]]:
    """
    Hàm tiền xử lý ảnh chính được bàn giao cho Bão (Backend Pipeline) và Sơn (App).

    Quy trình xử lý:
    1. Kiểm tra tính hợp lệ toàn diện (validation).
    2. Nạp ảnh thành đối tượng PIL Image.
    3. Hiệu chỉnh góc xoay theo EXIF.
    4. Chuyển đổi chuẩn hóa về hệ màu RGB.
    5. Resize an toàn giữ nguyên tỷ lệ nếu kích thước vượt ngưỡng target_max_dim.

    Args:
        image_input: Đầu vào ảnh đa dạng (Path, bytes, PIL Image, Streamlit UploadedFile, v.v.).
        target_max_dim: Cạnh lớn nhất tối đa cho phép trước khi resize bảo toàn tỷ lệ (mặc định 2048).
        return_metadata: Có kèm theo dictionary thông số sau xử lý hay không.

    Returns:
        - PIL.Image.Image (chế độ RGB chuẩn).
        - (Tùy chọn) Tuple[PIL.Image.Image, metadata_dict] nếu return_metadata=True.

    Raises:
        ImageProcessingError: Nếu ảnh không hợp lệ hoặc bị lỗi đọc.
    """
    is_valid, error_msg, meta = validate_image(image_input)
    if not is_valid:
        raise ImageProcessingError(f"Tiền xử lý thất bại: {error_msg}")

    # Nạp ảnh
    pil_img = _to_pil_image(image_input)

    # Hiệu chỉnh xoay theo EXIF
    pil_img = fix_orientation(pil_img)

    # Chuẩn hóa RGB
    pil_img = convert_to_rgb(pil_img)

    # Điều chỉnh kích thước bảo toàn tỷ lệ khung hình
    processed_img = resize_preserving_aspect_ratio(pil_img, target_max_dim=target_max_dim)

    meta["processed_width"] = processed_img.width
    meta["processed_height"] = processed_img.height
    meta["processed_mode"] = processed_img.mode

    if return_metadata:
        return processed_img, meta

    return processed_img


def get_image_metadata(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray, Any]
) -> Dict[str, Any]:
    """
    Trích xuất nhanh các thông số kỹ thuật của ảnh y tế (phục vụ hiển thị trên UI).
    """
    is_valid, msg, meta = validate_image(image_input)
    if not is_valid:
        meta["status"] = "invalid"
        meta["error_message"] = msg
        return meta

    meta["status"] = "valid"
    meta["error_message"] = ""
    return meta
