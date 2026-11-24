"""
Unit test kiểm thử module tiền xử lý và kiểm tra ảnh (Image Preprocessing & Data Validation).
Phụ trách: Trí (Image Preprocessing & Data Preparation)

Chạy kiểm thử:
    python -m unittest tests/test_preprocessing.py
hoặc từ thư mục Course_work:
    pytest tests/test_preprocessing.py
"""

import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Import trực tiếp từ module phụ trách
import sys

# Đảm bảo import được src khi chạy từ bất kỳ thư mục nào (repository root hoặc Course_work)
COURSE_WORK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = COURSE_WORK_DIR.parent
for p in [str(COURSE_WORK_DIR), str(REPO_ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.preprocessing.image_processing import (
        ImageProcessingError,
        SUPPORTED_EXTENSIONS,
        convert_to_rgb,
        fix_orientation,
        get_image_metadata,
        preprocess_image,
        resize_preserving_aspect_ratio,
        validate_image,
    )
except ImportError:
    from Course_work.src.preprocessing.image_processing import (
        ImageProcessingError,
        SUPPORTED_EXTENSIONS,
        convert_to_rgb,
        fix_orientation,
        get_image_metadata,
        preprocess_image,
        resize_preserving_aspect_ratio,
        validate_image,
    )


class TestImagePreprocessing(unittest.TestCase):
    """Tập hợp các ca kiểm thử cho module image_processing."""

    def setUp(self):
        """Khởi tạo các đối tượng và file ảnh mẫu dùng chung cho test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Ảnh RGB hợp lệ (200 x 200)
        self.valid_rgb_path = self.temp_path / "valid_rgb.jpg"
        img_rgb = Image.new("RGB", (200, 200), color=(120, 150, 180))
        img_rgb.save(self.valid_rgb_path, "JPEG")

        # 2. Ảnh Grayscale hợp lệ (300 x 250)
        self.valid_gray_path = self.temp_path / "valid_gray.png"
        img_gray = Image.new("L", (300, 250), color=100)
        img_gray.save(self.valid_gray_path, "PNG")

        # 3. Ảnh RGBA hợp lệ (có kênh alpha)
        self.valid_rgba_path = self.temp_path / "valid_rgba.png"
        img_rgba = Image.new("RGBA", (150, 150), color=(200, 50, 50, 180))
        img_rgba.save(self.valid_rgba_path, "PNG")

        # 4. File rỗng 0 bytes
        self.zero_byte_path = self.temp_path / "zero_byte.jpg"
        self.zero_byte_path.touch()

        # 5. File ảnh hỏng (corrupted bytes)
        self.corrupt_path = self.temp_path / "corrupt.jpg"
        with open(self.corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_NOT_A_VALID_HEADER_DATA_123456789")

        # 6. File không phải định dạng ảnh được hỗ trợ (.txt giả danh .jpg hoặc .bmp)
        self.text_file_path = self.temp_path / "test_doc.txt"
        with open(self.text_file_path, "w", encoding="utf-8") as f:
            f.write("Day la van ban khong phai anh.")

        # 7. Ảnh có kích thước quá nhỏ (< 50 x 50)
        self.small_img_path = self.temp_path / "too_small.png"
        img_small = Image.new("RGB", (30, 40), color=(0, 0, 0))
        img_small.save(self.small_img_path, "PNG")

    def tearDown(self):
        """Dọn dẹp thư mục tạm sau khi test xong."""
        self.temp_dir.cleanup()

    # =========================================================================
    # 1. KIỂM THỬ HÀM VALIDATE_IMAGE
    # =========================================================================

    def test_validate_valid_file_path(self):
        """Kiểm tra ảnh hợp lệ từ đường dẫn file."""
        is_valid, msg, meta = validate_image(self.valid_rgb_path)
        self.assertTrue(is_valid)
        self.assertEqual(meta["width"], 200)
        self.assertEqual(meta["height"], 200)
        self.assertEqual(meta["original_mode"], "RGB")

    def test_validate_valid_bytes(self):
        """Kiểm tra ảnh hợp lệ truyền vào dưới dạng bytes."""
        with open(self.valid_rgb_path, "rb") as f:
            raw_bytes = f.read()
        is_valid, msg, meta = validate_image(raw_bytes)
        self.assertTrue(is_valid)
        self.assertEqual(meta["width"], 200)

    def test_validate_valid_bytesio(self):
        """Kiểm tra ảnh truyền vào từ luồng io.BytesIO (tương tự Streamlit upload)."""
        with open(self.valid_rgb_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        is_valid, msg, meta = validate_image(buffer)
        self.assertTrue(is_valid)
        self.assertEqual(meta["height"], 200)

    def test_validate_pil_image_directly(self):
        """Kiểm tra truyền trực tiếp đối tượng PIL.Image."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        is_valid, msg, meta = validate_image(img)
        self.assertTrue(is_valid)
        self.assertEqual(meta["width"], 100)

    def test_validate_numpy_array(self):
        """Kiểm tra truyền mảng numpy 2D (grayscale) và 3D (RGB)."""
        arr_2d = np.ones((120, 120), dtype=np.uint8) * 128
        is_valid, _, meta_2d = validate_image(arr_2d)
        self.assertTrue(is_valid)
        self.assertEqual(meta_2d["width"], 120)

        arr_3d = np.zeros((150, 180, 3), dtype=np.uint8)
        is_valid, _, meta_3d = validate_image(arr_3d)
        self.assertTrue(is_valid)
        self.assertEqual(meta_3d["width"], 180)
        self.assertEqual(meta_3d["height"], 150)

    def test_validate_empty_and_corrupt_files(self):
        """Kiểm tra xử lý ảnh rỗng 0 bytes và ảnh hỏng cấu trúc."""
        # File 0 bytes
        is_valid, msg, _ = validate_image(self.zero_byte_path)
        self.assertFalse(is_valid)
        self.assertIn("0 bytes", msg)

        # File corrupt
        is_valid, msg, _ = validate_image(self.corrupt_path)
        self.assertFalse(is_valid)
        self.assertIn("hỏng", msg)

        # Dữ liệu None
        is_valid, msg, _ = validate_image(None)
        self.assertFalse(is_valid)

    def test_validate_unsupported_extension(self):
        """Kiểm tra từ chối định dạng tệp không được hỗ trợ."""
        is_valid, msg, _ = validate_image(self.text_file_path)
        self.assertFalse(is_valid)
        self.assertIn("không được hỗ trợ", msg)

    def test_validate_dimension_limits(self):
        """Kiểm tra ràng buộc kích thước tối thiểu và tối đa."""
        # Quá nhỏ (< 50)
        is_valid, msg, _ = validate_image(self.small_img_path, min_dimension=50)
        self.assertFalse(is_valid)
        self.assertIn("quá nhỏ", msg)

        # Quá lớn
        is_valid, msg, _ = validate_image(self.valid_rgb_path, max_dimension=150)
        self.assertFalse(is_valid)
        self.assertIn("quá lớn", msg)

    # =========================================================================
    # 2. KIỂM THỬ CHUẨN HÓA MÀU SẮC (RGB CONVERSION)
    # =========================================================================

    def test_convert_grayscale_to_rgb(self):
        """Ảnh 1 kênh Grayscale phải được chuyển thành 3 kênh RGB bằng nhau."""
        gray_img = Image.open(self.valid_gray_path)
        self.assertEqual(gray_img.mode, "L")

        rgb_img = convert_to_rgb(gray_img)
        self.assertEqual(rgb_img.mode, "RGB")
        self.assertEqual(rgb_img.size, gray_img.size)

        # Kiểm tra 3 kênh R, G, B có cùng giá trị (đặc thù chuyển đổi x-quang)
        arr = np.array(rgb_img)
        self.assertEqual(arr.shape[2], 3)
        np.testing.assert_array_equal(arr[:, :, 0], arr[:, :, 1])
        np.testing.assert_array_equal(arr[:, :, 1], arr[:, :, 2])

    def test_convert_rgba_to_rgb(self):
        """Ảnh RGBA có kênh trong suốt phải được hòa trộn nền trắng sang RGB."""
        rgba_img = Image.open(self.valid_rgba_path)
        self.assertEqual(rgba_img.mode, "RGBA")

        rgb_img = convert_to_rgb(rgba_img)
        self.assertEqual(rgb_img.mode, "RGB")
        self.assertEqual(rgb_img.size, rgba_img.size)

    # =========================================================================
    # 3. KIỂM THỬ BẢO TOÀN TỶ LỆ KHUNG HÌNH (ASPECT RATIO PRESERVATION)
    # =========================================================================

    def test_resize_preserving_aspect_ratio(self):
        """Ảnh lớn được thu nhỏ mà không làm méo tỷ lệ cạnh của cấu trúc y khoa."""
        # Tạo ảnh kích thước 3000 x 1500 (tỷ lệ 2:1)
        large_img = Image.new("RGB", (3000, 1500), color=(100, 100, 100))
        target_max = 2048

        resized = resize_preserving_aspect_ratio(large_img, target_max_dim=target_max)
        self.assertEqual(max(resized.size), target_max)
        self.assertEqual(resized.size[0], 2048)
        self.assertEqual(resized.size[1], 1024)  # 2:1 chính xác

        # Ảnh đã nhỏ hơn target_max thì giữ nguyên kích thước
        small_img = Image.new("RGB", (512, 512))
        untouched = resize_preserving_aspect_ratio(small_img, target_max_dim=1024)
        self.assertEqual(untouched.size, (512, 512))

    # =========================================================================
    # 4. KIỂM THỬ TOÀN DIỆN HÀM PREPROCESS_IMAGE (BÀN GIAO CHO BACKEND/APP)
    # =========================================================================

    def test_preprocess_image_success(self):
        """Quy trình tiền xử lý hoàn chỉnh thành công."""
        processed, meta = preprocess_image(self.valid_gray_path, return_metadata=True)
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(meta["processed_mode"], "RGB")
        self.assertEqual(meta["processed_width"], 300)
        self.assertEqual(meta["processed_height"], 250)

    def test_preprocess_image_raises_error_on_invalid(self):
        """Hàm preprocess_image phải raise ImageProcessingError khi ảnh không hợp lệ."""
        with self.assertRaises(ImageProcessingError):
            preprocess_image(self.zero_byte_path)

        with self.assertRaises(ImageProcessingError):
            preprocess_image(self.corrupt_path)

        with self.assertRaises(ImageProcessingError):
            preprocess_image(self.text_file_path)

    def test_get_image_metadata_helper(self):
        """Kiểm tra hàm trích xuất metadata hỗ trợ giao diện người dùng."""
        meta_valid = get_image_metadata(self.valid_rgb_path)
        self.assertEqual(meta_valid["status"], "valid")
        self.assertEqual(meta_valid["width"], 200)

        meta_invalid = get_image_metadata(self.zero_byte_path)
        self.assertEqual(meta_invalid["status"], "invalid")
        self.assertTrue(len(meta_invalid["error_message"]) > 0)


if __name__ == "__main__":
    unittest.main()
