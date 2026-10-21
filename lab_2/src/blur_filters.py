"""Module blur_filters.py - Cài đặt và đánh giá các bộ lọc làm mờ tuyến tính.

Phần phụ trách của Thành viên 4 (TV4 - Trí):
- Cài đặt Mean Filter (Lọc trung bình) và Gaussian Filter (Lọc Gauss).
- Cài đặt tạo ma trận trọng số (kernel) tùy chỉnh và đối sánh với thư viện OpenCV.
- Hỗ trợ cả tích chập thủ công (manual convolution) và hàm tối ưu của OpenCV.
- Kiểm tra tính hợp lệ và xử lý ngoại lệ cho đầu vào (Input validation).
- Các hàm đánh giá định lượng: PSNR, SSIM, Laplacian Variance (đo độ sắc nét), và Benchmark thời gian thực thi.
- Hỗ trợ thêm nhiễu Gaussian và nhiễu muối tiêu có kiểm soát để phục vụ thực nghiệm.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Tuple, Union

import cv2
import numpy as np


# ==============================================================================
# 1. HÀM KIỂM TRA ĐẦU VÀO (INPUT VALIDATION)
# ==============================================================================

def validate_image(image: np.ndarray) -> np.ndarray:
    """Kiểm tra tính hợp lệ của ảnh đầu vào.

    Args:
        image: Mảng NumPy biểu diễn ảnh (2D thang xám hoặc 3D BGR).

    Returns:
        np.ndarray: Ảnh hợp lệ.

    Raises:
        TypeError: Nếu đầu vào không phải là numpy.ndarray hoặc dtype không phải uint8.
        ValueError: Nếu ảnh rỗng hoặc không phải định dạng 2D / 3D 3 kênh.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Ảnh đầu vào phải là numpy.ndarray, nhận được: {type(image)}")
    if image.dtype != np.uint8:
        raise TypeError(f"Ảnh đầu vào phải có dtype là uint8, nhận được: {image.dtype}")
    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng (kích thước bằng 0).")
    if image.ndim not in (2, 3):
        raise ValueError(f"Ảnh phải có 2 chiều (grayscale) hoặc 3 chiều (color), nhận được {image.ndim} chiều.")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(f"Ảnh màu phải có 1 hoặc 3 kênh màu, nhận được {image.shape[2]} kênh.")
    return image


def validate_ksize(ksize: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Kiểm tra và chuẩn hóa kích thước kernel (khổ chập).

    Args:
        ksize: Kích thước kernel dạng số nguyên dương (k) hoặc tuple (kw, kh).

    Returns:
        Tuple[int, int]: Cặp kích thước (kw, kh) là các số nguyên dương lẻ.

    Raises:
        TypeError: Nếu ksize không phải int hoặc tuple/list các int.
        ValueError: Nếu ksize <= 0 hoặc là số chẵn.
    """
    if isinstance(ksize, int):
        kw, kh = ksize, ksize
    elif isinstance(ksize, (tuple, list)) and len(ksize) == 2:
        kw, kh = int(ksize[0]), int(ksize[1])
    else:
        raise TypeError(f"ksize phải là số nguyên int hoặc tuple 2 phần tử, nhận được: {ksize}")

    if kw <= 0 or kh <= 0:
        raise ValueError(f"Kích thước kernel phải lớn hơn 0, nhận được ({kw}, {kh}).")
    if kw % 2 == 0 or kh % 2 == 0:
        raise ValueError(f"Kích thước kernel của bộ lọc tuyến tính phải là số lẻ, nhận được ({kw}, {kh}).")

    return (kw, kh)


# ==============================================================================
# 2. HÀM ĐỌC VÀ LƯU ẢNH AN TOÀN TRÊN WINDOWS (UNICODE PATH SUPPORT)
# ==============================================================================

def read_image(path: Union[str, Path], as_gray: bool = False) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, hỗ trợ Unicode tiếng Việt an toàn trên Windows.

    Args:
        path: Đường dẫn tới file ảnh.
        as_gray: Nếu True đọc ảnh ở thang xám, ngược lại đọc ảnh màu BGR.

    Returns:
        np.ndarray: Ảnh uint8 đã đọc.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file bị lỗi hoặc không thể giải mã định dạng ảnh.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại đường dẫn: {p}")

    # Đọc raw binary bằng np.fromfile để tránh lỗi ký tự tiếng Việt trong cv2.imread trên Windows
    data = np.fromfile(str(p), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)

    if img is None:
        raise ValueError(f"Không thể giải mã nội dung ảnh từ file: {p}")
    return img


def save_image(path: Union[str, Path], image: np.ndarray) -> bool:
    """Lưu ảnh vào đường dẫn, hỗ trợ đường dẫn Unicode trên Windows.

    Args:
        path: Đường dẫn file lưu đầu ra.
        image: Mảng numpy ảnh uint8 cần lưu.

    Returns:
        bool: True nếu lưu thành công.

    Raises:
        OSError: Nếu không thể mã hóa hoặc ghi file.
    """
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


# ==============================================================================
# 3. KHỞI TẠO MA TRẬN TRỌNG SỐ (KERNEL GENERATORS)
# ==============================================================================

def create_mean_kernel(ksize: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
    """Tạo kernel lọc trung bình (Box/Mean filter) với trọng số đồng đều.

    Mọi phần tử trong kernel ksize x ksize đều bằng 1 / (kw * kh).

    Args:
        ksize: Kích thước kernel lẻ (int hoặc tuple 2 int). Mặc định là 3 (3x3).

    Returns:
        np.ndarray: Kernel ma trận 2D float32 với tổng trọng số bằng 1.0.
    """
    kw, kh = validate_ksize(ksize)
    kernel = np.ones((kh, kw), dtype=np.float32) / (kw * kh)
    return kernel


def create_gaussian_kernel_1d(ksize: int, sigma: float = 0.0) -> np.ndarray:
    """Tạo kernel Gauss 1 chiều theo công thức mật độ xác suất phân phối chuẩn.

    Công thức: G(x) = exp(-x^2 / (2 * sigma^2)) với x nằm trong khoảng [-(k-1)/2, (k-1)/2].
    Nếu sigma <= 0, tự động tính sigma theo công thức chuẩn của OpenCV:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    Args:
        ksize: Kích thước kernel 1 chiều (số lẻ dương).
        sigma: Độ lệch chuẩn của hàm Gauss.

    Returns:
        np.ndarray: Kernel 1 chiều float32 được chuẩn hóa có tổng bằng 1.0.
    """
    kw, _ = validate_ksize((ksize, ksize))
    if sigma <= 0:
        sigma = 0.3 * ((kw - 1) * 0.5 - 1.0) + 0.8

    radius = (kw - 1) // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(- (x ** 2) / (2.0 * (sigma ** 2)))
    kernel_1d /= np.sum(kernel_1d)
    return kernel_1d


def create_gaussian_kernel_2d(
    ksize: Union[int, Tuple[int, int]] = 3,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
) -> np.ndarray:
    """Tạo kernel Gauss 2 chiều hoàn chỉnh.

    Tận dụng tính chất tách được (separable) của hàm Gauss:
        G_2D(x, y) = G_1D(x, sigma_x) * G_1D(y, sigma_y)^T

    Args:
        ksize: Kích thước kernel lẻ (int hoặc tuple 2 int). Mặc định là 3 (3x3).
        sigma_x: Độ lệch chuẩn theo phương ngang (X). Mặc định 0.0 (tự động tính).
        sigma_y: Độ lệch chuẩn theo phương dọc (Y). Mặc định 0.0 (dùng bằng sigma_x).

    Returns:
        np.ndarray: Kernel 2D float32 với tổng các phần tử bằng 1.0.
    """
    kw, kh = validate_ksize(ksize)
    if sigma_x <= 0:
        sigma_x = 0.3 * ((kw - 1) * 0.5 - 1.0) + 0.8
    if sigma_y <= 0:
        sigma_y = sigma_x

    kx = create_gaussian_kernel_1d(kw, sigma_x).reshape(1, kw)
    ky = create_gaussian_kernel_1d(kh, sigma_y).reshape(kh, 1)

    kernel_2d = np.dot(ky, kx).astype(np.float32)
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d


# ==============================================================================
# 4. CÀI ĐẶT CÁC BỘ LỌC LÀM MỜ TUYẾN TÍNH (BLUR FILTERS)
# ==============================================================================

def apply_mean_filter(
    image: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = 3,
    use_opencv: bool = True,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Áp dụng bộ lọc trung bình (Mean / Box Filter) lên ảnh.

    Args:
        image: Ảnh đầu vào uint8 (2D hoặc 3D BGR).
        ksize: Kích thước kernel lẻ (int hoặc tuple). Ví dụ: 3 hoặc (3, 3), 5 hoặc (5, 5).
        use_opencv: Nếu True sử dụng hàm tối ưu cv2.blur().
                    Nếu False sử dụng tích chập ma trận cv2.filter2D với mean kernel tự tạo.
        border_type: Kiểu xử lý biên (cv2.BORDER_DEFAULT, cv2.BORDER_CONSTANT,...).

    Returns:
        np.ndarray: Ảnh uint8 sau khi lọc trung bình.
    """
    validate_image(image)
    kw, kh = validate_ksize(ksize)

    if use_opencv:
        return cv2.blur(image, (kw, kh), borderType=border_type)
    else:
        kernel = create_mean_kernel((kw, kh))
        return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=border_type)


def apply_gaussian_filter(
    image: np.ndarray,
    ksize: Union[int, Tuple[int, int]] = 3,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    use_opencv: bool = True,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian (Gaussian Blur) lên ảnh.

    Args:
        image: Ảnh đầu vào uint8 (2D hoặc 3D BGR).
        ksize: Kích thước kernel lẻ (int hoặc tuple). Mặc định 3 (3x3).
        sigma_x: Độ lệch chuẩn Gaussian theo trục X. 0 nghĩa là tự tính theo ksize.
        sigma_y: Độ lệch chuẩn Gaussian theo trục Y. 0 nghĩa là bằng sigma_x.
        use_opencv: Nếu True sử dụng hàm cv2.GaussianBlur() được tối ưu SIMD.
                    Nếu False sử dụng kernel Gauss 2D tự sinh kết hợp cv2.filter2D().
        border_type: Kiểu xử lý biên ảnh.

    Returns:
        np.ndarray: Ảnh uint8 sau khi làm mờ Gaussian.
    """
    validate_image(image)
    kw, kh = validate_ksize(ksize)

    if use_opencv:
        return cv2.GaussianBlur(
            image,
            ksize=(kw, kh),
            sigmaX=sigma_x,
            sigmaY=sigma_y,
            borderType=border_type,
        )
    else:
        kernel = create_gaussian_kernel_2d((kw, kh), sigma_x=sigma_x, sigma_y=sigma_y)
        return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=border_type)


def apply_manual_convolution(
    image: np.ndarray,
    kernel: np.ndarray,
    border_mode: str = "reflect",
) -> np.ndarray:
    """Cài đặt tích chập không gian 2D thủ công (Manual 2D Convolution).

    Mục đích: Minh chứng bản chất thuật toán tích chập theo đúng định nghĩa toán học:
        I'(y, x) = sum_{j} sum_{i} I(y+j, x+i) * K(j, i)
    Hỗ trợ cả ảnh grayscale (2D) và ảnh màu (3D, thực hiện tích chập từng kênh).

    Args:
        image: Mảng ảnh uint8 (2D hoặc 3D).
        kernel: Mảng kernel 2D float32 với kích thước lẻ.
        border_mode: Cách padding ('reflect', 'edge', 'constant').

    Returns:
        np.ndarray: Ảnh kết quả uint8 (được clip về khoảng [0, 255]).
    """
    validate_image(image)
    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2:
        raise ValueError("Kernel phải là mảng NumPy 2 chiều.")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel phải có kích thước các chiều là số lẻ.")

    pad_h = kh // 2
    pad_w = kw // 2

    # Flip kernel 180 độ theo đúng định nghĩa toán học của Convolution (thay vì Cross-correlation)
    # Tuy nhiên với kernel đối xứng (như Mean hay Gaussian), kernel lật bằng chính nó.
    k_flipped = np.flip(np.flip(kernel, 0), 1).astype(np.float32)

    is_color = (image.ndim == 3)
    channels = [image[:, :, c] for c in range(image.shape[2])] if is_color else [image]
    out_channels = []

    for ch in channels:
        h, w = ch.shape
        # Tạo viền padding tương ứng
        if border_mode == "reflect":
            padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect").astype(np.float32)
        elif border_mode == "edge":
            padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge").astype(np.float32)
        else:
            padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0).astype(np.float32)

        out_ch = np.zeros((h, w), dtype=np.float32)
        for r in range(kh):
            for c in range(kw):
                weight = k_flipped[r, c]
                out_ch += padded[r : r + h, c : c + w] * weight

        out_ch = np.clip(np.round(out_ch), 0, 255).astype(np.uint8)
        out_channels.append(out_ch)

    if is_color:
        return np.dstack(out_channels)
    else:
        return out_channels[0]


# ==============================================================================
# 5. CÁC HÀM TẠO NHIỄU THỰC NGHIỆM (NOISE GENERATION)
# ==============================================================================

def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 20.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Thêm nhiễu Gauss độc lập (Additive Gaussian Noise) vào ảnh.

    Args:
        image: Ảnh gốc uint8.
        mean: Giá trị kỳ vọng của nhiễu Gauss. Mặc định 0.0.
        sigma: Độ lệch chuẩn của nhiễu Gauss (cường độ nhiễu). Mặc định 20.0.
        seed: Seed sinh số ngẫu nhiên để tái lập kết quả. Mặc định 42.

    Returns:
        np.ndarray: Ảnh chứa nhiễu uint8 đã được clip vào khoảng [0, 255].
    """
    validate_image(image)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    salt_prob: float = 0.02,
    pepper_prob: float = 0.02,
    seed: int | None = 42,
) -> np.ndarray:
    """Thêm nhiễu muối tiêu (Impulse / Salt & Pepper Noise) vào ảnh.

    Args:
        image: Ảnh gốc uint8.
        salt_prob: Tỷ lệ pixel trắng (255 - hạt muối).
        pepper_prob: Tỷ lệ pixel đen (0 - hạt tiêu).
        seed: Seed sinh số ngẫu nhiên. Mặc định 42.

    Returns:
        np.ndarray: Ảnh chứa nhiễu muối tiêu uint8.
    """
    validate_image(image)
    rng = np.random.default_rng(seed)
    noisy = image.copy()

    # Sinh ma trận ngẫu nhiên theo kích thước chiều không gian (H, W)
    spatial_shape = image.shape[:2]
    rand_matrix = rng.uniform(0.0, 1.0, size=spatial_shape)

    # Đặt muối (255)
    salt_mask = rand_matrix < salt_prob
    # Đặt tiêu (0)
    pepper_mask = (rand_matrix >= salt_prob) & (rand_matrix < (salt_prob + pepper_prob))

    if image.ndim == 2:
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
    else:
        for c in range(image.shape[2]):
            noisy[:, :, c][salt_mask] = 255
            noisy[:, :, c][pepper_mask] = 0

    return noisy


# ==============================================================================
# 6. HÀM ĐÁNH GIÁ CHỈ SỐ ĐỊNH LƯỢNG (EVALUATION METRICS)
# ==============================================================================

def calculate_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """Tính toán tỷ số tín hiệu trên nhiễu đỉnh (Peak Signal-to-Noise Ratio - PSNR).

    Công thức: PSNR = 10 * log10(MAX_I^2 / MSE)
    Giá trị PSNR càng cao biểu thị ảnh sau xử lý càng tiệm cận chất lượng ảnh gốc.

    Args:
        original: Ảnh gốc chuẩn (uint8).
        filtered: Ảnh sau lọc / ảnh có nhiễu (uint8).

    Returns:
        float: Giá trị PSNR tính bằng decibel (dB). Trả về float('inf') nếu 2 ảnh giống hệt nhau.
    """
    validate_image(original)
    validate_image(filtered)
    if original.shape != filtered.shape:
        raise ValueError(f"Hai ảnh phải cùng kích thước: {original.shape} vs {filtered.shape}")

    mse = np.mean((original.astype(np.float64) - filtered.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
) -> float:
    """Tính chỉ số độ tương đồng cấu trúc (Structural Similarity Index - SSIM).

    Cài đặt thuật toán SSIM theo công thức chuẩn của Wang et al. (IEEE TIP 2004):
        SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) /
                     ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

    Args:
        img1: Ảnh thứ nhất (uint8).
        img2: Ảnh thứ hai (uint8).
        k1: Hằng số ổn định độ sáng C1 = (k1 * L)^2.
        k2: Hằng số ổn định độ tương phản C2 = (k2 * L)^2.
        win_size: Kích thước cửa sổ Gaussian trượt (mặc định 11x11).

    Returns:
        float: Chỉ số SSIM trung bình toàn ảnh trong khoảng [-1, 1] (thường từ 0 đến 1).
    """
    validate_image(img1)
    validate_image(img2)
    if img1.shape != img2.shape:
        raise ValueError(f"Kích thước hai ảnh phải khớp nhau: {img1.shape} vs {img2.shape}")

    # Chuyển ảnh xám nếu là ảnh màu để tính SSIM trên kênh cường độ sáng
    if img1.ndim == 3:
        i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        i2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)

    L = 255.0
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Cửa sổ trọng số Gaussian chuẩn cho SSIM (sigma = 1.5)
    kernel = cv2.getGaussianKernel(win_size, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(i1, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2]
    mu2 = cv2.filter2D(i2, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2]

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(i1 * i1, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu1_sq
    sigma2_sq = cv2.filter2D(i2 * i2, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu2_sq
    sigma12 = cv2.filter2D(i1 * i2, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def calculate_laplacian_variance(image: np.ndarray) -> float:
    """Tính phương sai toán tử Laplacian (Laplacian Variance) để đo độ sắc nét của ảnh.

    Khi ảnh bị làm mờ, các cạnh sắc nét và tần số cao bị triệt tiêu,
    khiến cho đáp ứng của đạo hàm bậc hai Laplacian giảm mạnh, dẫn đến phương sai giảm.
    - Giá trị càng cao: Ảnh càng sắc nét, nhiều chi tiết cạnh rõ.
    - Giá trị càng thấp: Ảnh càng bị mờ mịn hoặc mất chi tiết cạnh.

    Args:
        image: Ảnh đầu vào uint8 (grayscale hoặc color).

    Returns:
        float: Phương sai đáp ứng của bộ lọc Laplacian (Var(Laplacian)).
    """
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def benchmark_filter(
    filter_func: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    warmup: int = 3,
    iterations: int = 20,
) -> float:
    """Đo thời gian thực thi trung bình của bộ lọc trên ảnh.

    Args:
        filter_func: Hàm nhận một đối số image và trả về ảnh đã lọc.
        image: Ảnh dùng để benchmark.
        warmup: Số lần chạy khởi động (loại bỏ hiệu ứng cold start / cache miss).
        iterations: Số lần lặp đo thời gian.

    Returns:
        float: Thời gian thực thi trung bình cho 1 lần lọc tính bằng mili-giây (ms).
    """
    for _ in range(warmup):
        _ = filter_func(image)

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = filter_func(image)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / iterations) * 1000.0
    return float(avg_ms)
