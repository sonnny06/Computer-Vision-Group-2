import cv2

def convert_to_hsv(image):
    """
    Chuyển ảnh từ định dạng BGR sang HSV
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convert_to_lab(image):
    """
    Chuyển ảnh từ định dạng BGR sang LAB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def split_hsv_channels(hsv):
    """
    Tách ảnh HSV thành ba kênh Hue, Saturation và Value
    """
    return cv2.split(hsv)


def split_lab_channels(lab):
    """
    Tách ảnh LAB thành ba kênh Lightness, A và B
    """
    return cv2.split(lab)