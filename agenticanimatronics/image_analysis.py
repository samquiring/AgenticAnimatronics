from pathlib import Path

import cv2

from agenticanimatronics.llm import LLMHandler


class ImageAnalysis:
    def __init__(self, prompt=""):
        self.llm = LLMHandler()
        self.prompt = prompt
        self.analysis = None

    def take_and_analyse_image(self, s, queue):
        image_location = self.take_image()
        self.analysis = self.llm.explain_image(image_location, self.prompt)
        queue.put(self.analysis)

    @staticmethod
    def take_image(image_location="", image_name="photo.png"):
        cam = cv2.VideoCapture(0)  # 0 is usually the default camera
        result, image = cam.read()
        path_to_image = Path(image_location, image_name)
        cv2.imwrite(str(path_to_image), image)
        return path_to_image