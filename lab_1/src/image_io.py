import cv2


def read_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Không thể đọc ảnh tại: {image_path}"
        )

    return image