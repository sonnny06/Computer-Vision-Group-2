import cv2
import numpy as np


def sharpen_image(image, strength=1.0):
    """
    Sharpen an image using a 3x3 sharpening kernel.

    Parameters
    ----------
    image : numpy.ndarray
        Input image, grayscale or color.
    strength : float
        Sharpening strength. Values around 1.0 are a good starting point.

    Returns
    -------
    numpy.ndarray
        Sharpened image with the same height, width, and channel count.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a valid NumPy image array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if strength < 0:
        raise ValueError("strength must be non-negative")

    kernel = np.array([
        [0, -1, 0],
        [-1, 4 * strength + 1, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def apply_custom_kernel(image, kernel, normalize=False):
    """
    Apply a user-designed 2D kernel to a grayscale or color image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    kernel : array-like
        A 2D kernel with odd height and odd width.
    normalize : bool
        If True, divide the kernel by its sum when that sum is non-zero.
        Keep False for sharpening kernels whose coefficients are designed
        to amplify local differences.

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a valid, non-empty NumPy array")

    kernel = np.asarray(kernel, dtype=np.float32)

    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2D matrix")

    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel height and width must both be odd")

    if normalize:
        kernel_sum = float(kernel.sum())
        if abs(kernel_sum) > 1e-8:
            kernel = kernel / kernel_sum

    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def custom_edge_sharpen(image):
    """
    A stronger custom sharpening kernel.

    The positive center coefficient emphasizes the current pixel.
    Negative neighboring coefficients subtract nearby pixel values,
    increasing local contrast around edges.
    """
    custom_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    return apply_custom_kernel(image, custom_kernel)


if __name__ == "__main__":
    image_path = "sample.jpg"
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read '{image_path}'. Put a test image beside this file "
            "or change image_path to the correct image location."
        )

    sharpened = sharpen_image(image, strength=1.0)
    custom_sharpened = custom_edge_sharpen(image)

    cv2.imwrite("sample_sharpened.jpg", sharpened)
    cv2.imwrite("sample_custom_sharpened.jpg", custom_sharpened)

    print("Saved: sample_sharpened.jpg")
    print("Saved: sample_custom_sharpened.jpg")
