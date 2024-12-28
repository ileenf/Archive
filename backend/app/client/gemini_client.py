from google import genai

class GeminiClient:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate_response(self, prompt):
        return self.client.models.generate_content(model='gemini-1.5-pro-002', contents=prompt)