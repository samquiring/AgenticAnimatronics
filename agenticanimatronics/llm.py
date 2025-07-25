from litellm import completion

from agenticanimatronics.image_creation import encode_image


class LLMHandler:
    def __init__(self, model_name="gemini/gemini-2.5-flash-lite"):
        self.model = model_name

    def generate_response(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = completion(model=self.model, messages=messages)
                out = response.choices[0].message.content
                print(out)
                return out
            except Exception as e:
                print(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Arr, me brain be foggy today, matey! Try again later."
                import time
                time.sleep(1)

    def explain_image(self, image_location, prompt):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
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
            except Exception as e:
                print(f"Image analysis error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "A mysterious stranger stands before me"
                import time
                time.sleep(1)