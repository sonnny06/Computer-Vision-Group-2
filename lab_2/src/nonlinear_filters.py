import cv2
import numpy as np


def median_filter(image, ksize=3):
    """
    Reduce salt-and-pepper noise with a median filter.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or color image. OpenCV-compatible uint8 images
        are recommended for this lab.
    ksize : int
        Odd neighborhood size, typically 3 or 5.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as the input.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid NumPy image array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if not isinstance(ksize, int) or ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer greater than or equal to 3")

    if image.dtype != np.uint8:
        raise TypeError(
            "median_filter expects a uint8 image. Convert the image to uint8 "
            "before filtering, then convert it back if your pipeline requires it."
        )

    return cv2.medianBlur(image, ksize)


def add_salt_pepper_noise(image, amount=0.03, salt_vs_pepper=0.5, seed=42):
    """
    Add reproducible salt-and-pepper noise for experiments.

    amount: fraction of image pixels to corrupt, from 0 to 1.
    salt_vs_pepper: fraction of corrupted pixels set to white (salt).
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a valid, non-empty NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("add_salt_pepper_noise expects a uint8 image")

    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1")

    if not 0 <= salt_vs_pepper <= 1:
        raise ValueError("salt_vs_pepper must be between 0 and 1")

    noisy = image.copy()
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    total = int(height * width * amount)
    salt_count = int(total * salt_vs_pepper)
    pepper_count = total - salt_count

    salt_y = rng.integers(0, height, size=salt_count)
    salt_x = rng.integers(0, width, size=salt_count)
    pepper_y = rng.integers(0, height, size=pepper_count)
    pepper_x = rng.integers(0, width, size=pepper_count)

    noisy[salt_y, salt_x] = 255
    noisy[pepper_y, pepper_x] = 0

    return noisy
def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    if image is None:
        raise ValueError("Input image cannot be None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array")

    if image.size == 0:
        raise ValueError("Input image cannot be empty")

    if image.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or color")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Color image must have 3 channels")

    if d <= 0:
        raise ValueError("d must be positive")

    if sigma_color <= 0:
        raise ValueError("sigma_color must be positive")

    if sigma_space <= 0:
        raise ValueError("sigma_space must be positive")

    return cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )