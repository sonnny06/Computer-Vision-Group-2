import cv2


def crop_image(image, x1, y1, x2, y2):
    """
    Crop ảnh theo tọa độ:
    image[y1:y2, x1:x2]

    Parameters:
        image: ảnh OpenCV
        x1, y1: tọa độ góc trên bên trái
        x2, y2: tọa độ góc dưới bên phải

    Returns:
        Ảnh đã crop
    """
    return image[y1:y2, x1:x2]


def resize_by_ratio(image, ratio):
    """
    Resize ảnh theo tỷ lệ phần trăm.
    
    """
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
    """
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )