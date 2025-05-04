import base64
from pathlib import Path


def encode_image(image_path):
    with Path.open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

