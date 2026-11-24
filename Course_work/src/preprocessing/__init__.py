"""
Package preprocessing
"""
try:
    from .image_processing import (
        ImageProcessingError,
        convert_to_rgb,
        fix_orientation,
        get_image_metadata,
        preprocess_image,
        resize_preserving_aspect_ratio,
        validate_image,
    )
except ImportError:
    from src.preprocessing.image_processing import (
        ImageProcessingError,
        convert_to_rgb,
        fix_orientation,
        get_image_metadata,
        preprocess_image,
        resize_preserving_aspect_ratio,
        validate_image,
    )

__all__ = [
    "preprocess_image",
    "validate_image",
    "get_image_metadata",
    "convert_to_rgb",
    "fix_orientation",
    "resize_preserving_aspect_ratio",
    "ImageProcessingError",
]
