import base64
from pathlib import Path

import cv2


def encode_image(image_path):
    with Path.open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def take_image(image_location="", image_name="photo.png"):
    cam = cv2.VideoCapture(0)  # 0 is usually the default camera
    # discard the first photo since sometimes it fails to take
    _, _ = cam.read()
    result, image = cam.read()
    path_to_image = Path(image_location, image_name)
    cv2.imwrite(str(path_to_image), image)
    cam.release()
    return path_to_image
