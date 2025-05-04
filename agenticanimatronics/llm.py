from litellm import completion

from agenticanimatronics.image_creation import encode_image


class LLMHandler:
    def __init__(self, model_name="gemini/gemini-2.5-flash-preview-04-17"):
        self.model = model_name

    def generate_response(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = completion(model=self.model, messages=messages)
        out = response.choices[0].message.content
        print(out)
        return out

    def explain_image(self, image_location, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_location)}"
                        },
                    },
                ],
            }
        ]

        response = completion(model=self.model, messages=messages)
        out = response.choices[0].message.content
        print(out)
        return out