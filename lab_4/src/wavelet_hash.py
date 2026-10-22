"""Module wavelet_hash.py - Cài đặt thuật toán băm ảnh tri giác Wavelet Hash (W-Hash).

Phần phân công nhiệm vụ của Thành viên 4 (TV4 - Trí):
- Xây dựng thuật toán băm ảnh dựa trên biến đổi Wavelet 2D (2D Discrete Wavelet Transform).
- Sử dụng các hệ số xấp xỉ tần số thấp (LL approximation coefficients) để tạo mã hash có độ dài cố định.
- Hỗ trợ các phương pháp lượng tử hóa ổn định: Median thresholding, Mean thresholding.
- Cài đặt lớp WaveletHash chuẩn mực với toán tử so sánh khoảng cách Hamming (`h1 - h2`).
- Đánh giá tính ổn định (Robustness) đối với các biến đổi nén, nhiễu, làm mờ, thay đổi kích thước.
- Đánh giá tính phân biệt (Discriminability) giữa các ảnh có nội dung hoàn toàn khác nhau.
- Thử nghiệm nâng cao: Khảo sát các họ wavelet (haar, db2, db4, bior2.2, sym4), các mức phân rã (level), và độ dài mã hash.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pywt

try:
    from common import read_image, to_grayscale, validate_image
except ImportError:
    try:
        from lab_4.src.common import read_image, to_grayscale, validate_image
    except ImportError:
        # Fallback to local common in same directory
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from common import read_image, to_grayscale, validate_image


# ==============================================================================
# 1. LỚP ĐẠI DIỆN MÃ HASH WAVELET (WAVELET HASH CLASS)
# ==============================================================================

class WaveletHash:
    """Lớp đối tượng lưu trữ và thao tác với mã băm Wavelet Hash.

    Cung cấp các toán tử tiện lợi:
    - `h1 - h2`: Tính khoảng cách Hamming (số bit khác nhau).
    - `h1 == h2`: So sánh xem hai mã hash có trùng khớp tuyệt đối hay không.
    - `str(h)`: Trả về chuỗi ký tự Hexadecimal biểu diễn mã hash.
    """

    def __init__(
        self,
        hash_bits: np.ndarray,
        hash_size: int = 8,
        wavelet: str = "haar",
        level: int = 1,
        mode: str = "median",
    ):
        """Khởi tạo đối tượng WaveletHash.

        Args:
            hash_bits: Mảng boolean 1D hoặc 2D biểu diễn các bit của mã hash.
            hash_size: Kích thước cạnh ma trận hash (mặc định 8, tương đương 64 bit).
            wavelet: Tên họ wavelet đã sử dụng.
            level: Mức phân rã wavelet.
            mode: Phương pháp tính ngưỡng lượng tử hóa ('median' hoặc 'mean').
        """
        bits = np.asarray(hash_bits, dtype=bool)
        if bits.ndim == 2:
            self.hash_matrix = bits
            self.hash_bits = bits.flatten()
        elif bits.ndim == 1:
            self.hash_bits = bits
            side = int(round(np.sqrt(bits.size)))
            if side * side == bits.size:
                self.hash_matrix = bits.reshape((side, side))
            else:
                self.hash_matrix = bits.reshape((hash_size, -1))
        else:
            raise ValueError(f"hash_bits phải là mảng 1D hoặc 2D, nhận được ndim={bits.ndim}")

        self.hash_size = hash_size
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.bit_length = self.hash_bits.size
        self._hash_hex = self._encode_hex()

    def _encode_hex(self) -> str:
        """Mã hóa mảng bit nhị phân thành chuỗi Hexadecimal cố định."""
        bits = self.hash_bits.astype(np.uint8)
        # Bổ sung padding 0 nếu số bit không chia hết cho 8
        pad_len = (8 - (len(bits) % 8)) % 8
        if pad_len > 0:
            bits = np.pad(bits, (0, pad_len), mode="constant", constant_values=0)
        byte_arr = np.packbits(bits)
        return bytes(byte_arr).hex()

    @property
    def hex(self) -> str:
        """Chuỗi hex đại diện của mã hash."""
        return self._hash_hex

    def to_binary_string(self) -> str:
        """Chuỗi ký tự '0' và '1' biểu diễn toàn bộ các bit."""
        return "".join("1" if b else "0" for b in self.hash_bits)

    def to_bits(self) -> np.ndarray:
        """Mảng boolean 1D chứa các bit."""
        return self.hash_bits.copy()

    def to_matrix(self) -> np.ndarray:
        """Ma trận boolean 2D kích thước (hash_size, hash_size)."""
        return self.hash_matrix.copy()

    def __sub__(self, other: Any) -> int:
        """Tính khoảng cách Hamming giữa hai đối tượng WaveletHash (hoặc chuỗi hex)."""
        return hamming_distance(self, other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (WaveletHash, str)):
            return False
        return (self - other) == 0

    def __len__(self) -> int:
        return self.bit_length

    def __str__(self) -> str:
        return self._hash_hex

    def __repr__(self) -> str:
        return f"WaveletHash(hex='{self._hash_hex}', bits={self.bit_length}, wavelet='{self.wavelet}', mode='{self.mode}')"

    @classmethod
    def from_hex(
        cls,
        hex_str: str,
        hash_size: int = 8,
        wavelet: str = "haar",
        level: int = 1,
        mode: str = "median",
    ) -> WaveletHash:
        """Tạo đối tượng WaveletHash từ một chuỗi Hexadecimal."""
        clean_hex = hex_str.strip().lower()
        byte_data = bytes.fromhex(clean_hex)
        raw_bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        target_len = hash_size * hash_size
        bits = raw_bits[:target_len].astype(bool)
        return cls(bits, hash_size=hash_size, wavelet=wavelet, level=level, mode=mode)


# ==============================================================================
# 2. HÀM TẠO MÃ BĂM WAVELET HASH CHÍNH (CORE ALGORITHM)
# ==============================================================================

def wavelet_hash(
    image_or_path: Union[str, Path, np.ndarray],
    hash_size: int = 8,
    wavelet: str = "haar",
    level: int = 1,
    mode: str = "median",
    image_scale: Optional[int] = None,
) -> WaveletHash:
    """Xây dựng mã băm Wavelet Hash (W-Hash) từ ảnh hoặc đường dẫn file ảnh.

    Quy trình thuật toán:
    1. Đọc và tiền xử lý: Chuyển ảnh về thang xám (Grayscale float32).
    2. Chuẩn hóa kích thước (Resizing): Tỷ lệ hóa ảnh về kích thước lũy thừa của 2
       phù hợp với mức phân rã wavelet: `scale = hash_size * (2 ** level)`.
    3. Biến đổi Wavelet 2D (2D DWT): Sử dụng thư viện `PyWavelets` (`pywt.wavedec2` hoặc `pywt.dwt2`).
       Trích xuất thành phần hệ số xấp xỉ tần số thấp (LL subband), nơi tập trung
       hầu hết cấu trúc hình học và năng lượng tri giác của ảnh.
    4. Lấy mẫu hệ số: Đảm bảo vùng hệ số LL có kích thước chuẩn `(hash_size, hash_size)`.
    5. Lượng tử hóa và ngưỡng hóa (Binarization):
       - Ngưỡng trung vị (`median`): `bits = LL > median(LL)` -> Đảm bảo số bit 0 và 1 cân bằng 50/50,
         tối đa hóa entropy thông tin và chống chịu xuất sắc với các điểm đột biến.
       - Ngưỡng trung bình (`mean`): `bits = LL > mean(LL)`.
    6. Đóng gói thành đối tượng `WaveletHash`.

    Args:
        image_or_path: Ảnh dưới dạng mảng numpy hoặc đường dẫn tới file ảnh.
        hash_size: Kích thước cạnh ma trận hash (mặc định 8, tạo ra mã hash 64-bit).
        wavelet: Tên họ wavelet ('haar', 'db2', 'db4', 'bior2.2', 'sym4'...).
        level: Mức phân rã wavelet (mặc định 1).
        mode: Phương pháp tính ngưỡng ('median' hoặc 'mean').
        image_scale: Kích thước ảnh chuẩn hóa trước khi phân rã. Nếu None, tự động tính.

    Returns:
        WaveletHash: Đối tượng mã hash chứa mảng bit và chuỗi hex.

    Raises:
        FileNotFoundError: Nếu file ảnh không tồn tại.
        ValueError: Nếu tham số không hợp lệ.
    """
    if hash_size <= 0:
        raise ValueError(f"hash_size phải là số nguyên dương, nhận được: {hash_size}")
    if level <= 0:
        raise ValueError(f"level phân rã phải >= 1, nhận được: {level}")
    if mode not in ("median", "mean", "zero"):
        raise ValueError(f"mode phải là 'median', 'mean' hoặc 'zero', nhận được: '{mode}'")

    # 1. Đọc ảnh và chuyển sang Grayscale
    if isinstance(image_or_path, (str, Path)):
        img = read_image(image_or_path, as_gray=True)
    elif isinstance(image_or_path, np.ndarray):
        validate_image(image_or_path)
        img = to_grayscale(image_or_path)
    else:
        raise TypeError(f"image_or_path phải là str, Path hoặc np.ndarray, nhận được: {type(image_or_path)}")

    # 2. Tính toán tỷ lệ kích thước chuẩn
    if image_scale is None:
        target_dim = hash_size * (2 ** level)
    else:
        target_dim = image_scale

    # Đảm bảo target_dim đủ lớn để phân rã đến level yêu cầu
    min_dim = hash_size * (2 ** level)
    if target_dim < min_dim:
        target_dim = min_dim

    # Resize ảnh về kích thước chuẩn hóa (dùng nội suy diện tích INTER_AREA để làm mượt chống aliasing)
    resized_img = cv2.resize(img, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    img_f32 = resized_img.astype(np.float32)

    # 3. Thực hiện biến đổi Wavelet 2D đa cấp
    coeffs = pywt.wavedec2(img_f32, wavelet=wavelet, level=level)
    # coeffs[0] chính là thành phần xấp xỉ LL ở mức phân rã sâu nhất
    ll_coeffs = coeffs[0]

    # 4. Đảm bảo kích thước ma trận LL đúng bằng (hash_size, hash_size)
    if ll_coeffs.shape != (hash_size, hash_size):
        ll_coeffs = cv2.resize(ll_coeffs, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # 5. Ngưỡng hóa (Quantization)
    if mode == "median":
        thresh = float(np.median(ll_coeffs))
    elif mode == "mean":
        thresh = float(np.mean(ll_coeffs))
    else: # zero
        thresh = 0.0

    hash_matrix = (ll_coeffs > thresh).astype(bool)

    # 6. Đóng gói đối tượng WaveletHash
    return WaveletHash(
        hash_bits=hash_matrix,
        hash_size=hash_size,
        wavelet=wavelet,
        level=level,
        mode=mode,
    )


# ==============================================================================
# 3. KHOẢNG CÁCH HAMMING VÀ ĐỘ TƯƠNG ĐỒNG (HAMMING DISTANCE & MATCHING)
# ==============================================================================

def hamming_distance(
    h1: Union[WaveletHash, str, np.ndarray],
    h2: Union[WaveletHash, str, np.ndarray],
) -> int:
    """Tính khoảng cách Hamming (số lượng bit khác biệt) giữa hai mã hash.

    Args:
        h1: Mã hash thứ nhất (WaveletHash, chuỗi hex hoặc mảng boolean/uint8).
        h2: Mã hash thứ hai (WaveletHash, chuỗi hex hoặc mảng boolean/uint8).

    Returns:
        int: Số lượng bit sai khác.
    """
    bits1 = _extract_bits(h1)
    bits2 = _extract_bits(h2)

    if len(bits1) != len(bits2):
        raise ValueError(
            f"Hai mã hash phải có cùng độ dài bit! (Nhận được {len(bits1)} bits và {len(bits2)} bits)"
        )

    # Phép XOR giữa hai mảng bit nhị phân
    diff_count = int(np.count_nonzero(bits1 != bits2))
    return diff_count


def normalized_hamming_distance(
    h1: Union[WaveletHash, str, np.ndarray],
    h2: Union[WaveletHash, str, np.ndarray],
) -> float:
    """Tính khoảng cách Hamming chuẩn hóa trong đoạn [0.0, 1.0].

    Khoảng cách 0.0 nghĩa là hai mã băm giống nhau tuyệt đối.
    Khoảng cách xấp xỉ 0.5 nghĩa là hai mã băm hoàn toàn ngẫu nhiên/khác nhau.
    """
    bits1 = _extract_bits(h1)
    raw_dist = hamming_distance(h1, h2)
    return float(raw_dist) / float(len(bits1))


def is_similar(
    h1: Union[WaveletHash, str, np.ndarray],
    h2: Union[WaveletHash, str, np.ndarray],
    threshold: int = 5,
) -> bool:
    """Dự đoán hai ảnh có tương đồng hay không dựa trên ngưỡng khoảng cách Hamming."""
    return hamming_distance(h1, h2) <= threshold


def _extract_bits(h: Union[WaveletHash, str, np.ndarray]) -> np.ndarray:
    """Chuyển đổi linh hoạt đầu vào về mảng 1D boolean bit."""
    if isinstance(h, WaveletHash):
        return h.hash_bits
    elif isinstance(h, np.ndarray):
        return h.flatten().astype(bool)
    elif isinstance(h, str):
        clean_hex = h.strip().lower()
        byte_data = bytes.fromhex(clean_hex)
        raw_bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        return raw_bits.astype(bool)
    else:
        raise TypeError(f"Không thể trích xuất bit từ kiểu dữ liệu: {type(h)}")


# ==============================================================================
# 4. HÀM TẠO BIẾN DỔI ẢNH ĐỂ ĐÁNH GIÁ TÍNH ỔN ĐỊNH (ROBUSTNESS TESTING)
# ==============================================================================

def perturb_image(
    image: np.ndarray,
    perturbation_type: str,
    severity: float = 1.0,
) -> np.ndarray:
    """Tạo các biến đổi ảnh có kiểm soát nhằm kiểm thử độ bền bỉ của Wavelet Hash.

    Args:
        image: Ảnh gốc uint8.
        perturbation_type:
            - 'jpeg': Nén JPEG với hệ số chất lượng (severity = 1.0 ứng với Q=50, 2.0 ứng với Q=20).
            - 'gaussian_noise': Thêm nhiễu Gauss (severity tương ứng độ lệch chuẩn sigma = 10 * severity).
            - 'salt_pepper': Thêm nhiễu muối tiêu (tỷ lệ noise = 0.02 * severity).
            - 'gaussian_blur': Làm mờ Gauss (ksize = 3, 5, 9 theo severity).
            - 'brightness': Tăng/giảm độ sáng (+/- 25 * severity).
            - 'contrast': Thay đổi độ tương phản (alpha = 1.0 + 0.3 * severity).
            - 'resize_down_up': Thu nhỏ 50% rồi phóng to lại (mất mát chi tiết).
            - 'rotation': Xoay góc nhỏ (1 độ * severity).
            - 'crop_pad': Cắt biên 5% và đệm lại.

    Returns:
        np.ndarray: Ảnh sau khi bị biến đổi.
    """
    validate_image(image)
    img = image.copy()
    p_type = perturbation_type.lower().strip()

    if p_type == "jpeg":
        # Severity: 0.5 (Q=80), 1.0 (Q=50), 2.0 (Q=20)
        quality = max(5, int(round(90 - severity * 35)))
        success, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            return cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
        return img

    elif p_type == "gaussian_noise":
        sigma = 10.0 * severity
        gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy

    elif p_type == "salt_pepper":
        prob = min(0.3, 0.02 * severity)
        noisy = img.copy()
        # Salt
        num_salt = int(prob * 0.5 * img.size / (img.shape[2] if img.ndim == 3 else 1))
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        noisy[tuple(coords)] = 255
        # Pepper
        num_pepper = int(prob * 0.5 * img.size / (img.shape[2] if img.ndim == 3 else 1))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
        noisy[tuple(coords)] = 0
        return noisy

    elif p_type == "gaussian_blur":
        k = 3 if severity <= 1.0 else (5 if severity <= 2.0 else 7)
        return cv2.GaussianBlur(img, (k, k), sigmaX=severity)

    elif p_type == "brightness":
        beta = int(25 * severity)
        return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

    elif p_type == "contrast":
        alpha = max(0.2, 1.0 + 0.3 * severity)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    elif p_type == "resize_down_up":
        scale = max(0.2, 1.0 - 0.25 * severity)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return restored

    elif p_type == "rotation":
        angle = 1.5 * severity
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    elif p_type == "crop_pad":
        h, w = img.shape[:2]
        crop_fraction = min(0.2, 0.04 * severity)
        dh, dw = int(h * crop_fraction), int(w * crop_fraction)
        cropped = img[dh : h - dh, dw : w - dw]
        padded = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return padded

    else:
        raise ValueError(f"Loại biến đổi không hỗ trợ: '{perturbation_type}'")


# ==============================================================================
# 5. HÀM THỰC NGHIỆM VÀ ĐÁNH GIÁ TỔNG QUÁT (EVALUATION HELPERS)
# ==============================================================================

def batch_compute_hashes(
    images_dict: Dict[str, Union[str, Path, np.ndarray]],
    hash_size: int = 8,
    wavelet: str = "haar",
    level: int = 1,
    mode: str = "median",
) -> Dict[str, WaveletHash]:
    """Tính mã Wavelet Hash hàng loạt cho một danh sách/từ điển ảnh."""
    hashes = {}
    for name, img_item in images_dict.items():
        hashes[name] = wavelet_hash(
            img_item,
            hash_size=hash_size,
            wavelet=wavelet,
            level=level,
            mode=mode,
        )
    return hashes
