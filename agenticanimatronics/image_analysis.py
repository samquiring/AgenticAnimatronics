

from agenticanimatronics.image_creation import take_image
from agenticanimatronics.llm import LLMHandler


class ImageAnalysis:
    def __init__(self, prompt=""):
        self.llm = LLMHandler()
        self.prompt = prompt
        self.analysis = None

    def take_and_analyse_image(self, s, queue):
        image_location = take_image()
        self.analysis = self.llm.explain_image(image_location, self.prompt)
        queue.put(self.analysis)
