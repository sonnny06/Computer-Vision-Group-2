"""Color-space conversions for Lab 1."""

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
