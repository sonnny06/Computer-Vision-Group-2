import cv2


def crop_image(image, x1, y1, x2, y2):
    """
    Crop ảnh theo tọa độ:
    image[y1:y2, x1:x2]
    """
    if image is None:
        raise ValueError("image không được là None")

    height, width = image.shape[:2]

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Tọa độ crop không hợp lệ. "
            f"Ảnh có width={width}, height={height}; "
            f"nhận được ({x1}, {y1}, {x2}, {y2})."
        )

    return image[y1:y2, x1:x2]


def resize_by_ratio(image, ratio):
    """
    Resize ảnh theo một tỷ lệ.
    
    Ví dụ:
    ratio = 0.5 -> 50%
    ratio = 0.25 -> 25%
    """
    if image is None:
        raise ValueError("image không được là None")

    if ratio <= 0:
        raise ValueError("ratio phải lớn hơn 0")

    return cv2.resize(
        image,
        None,
        fx=ratio,
        fy=ratio,
        interpolation=cv2.INTER_AREA
    )


def resize_fixed(image, width, height):
    """
    Resize ảnh về kích thước cố định.
    
    Ví dụ:
    width = 800
    height = 600
    """
    if image is None:
        raise ValueError("image không được là None")

    if width <= 0 or height <= 0:
        raise ValueError("width và height phải lớn hơn 0")

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )