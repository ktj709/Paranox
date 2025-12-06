# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash"):
        # Load variables from .env file
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY not found! Please set it in a .env file or as an environment variable."
            )

        genai.configure(api_key=api_key)
        self.model_name = model

    def _get_model(self):
        """Create a fresh GenerativeModel instance for each request to avoid caching."""
        return genai.GenerativeModel(self.model_name)

    def summarize_text(self, text: str) -> str:
        model = self._get_model()
        response = model.generate_content(
            f"Summarize this text clearly and concisely:\n\n{text}"
        )
        return response.text

    def analyze_image(self, image_bytes: bytes) -> str:
        model = self._get_model()
        response = model.generate_content(
            [f"Describe this image in detail for a PDF report.", {"mime_type": "image/png", "data": image_bytes}]
        )
        return response.text

    def multimodal_summary(self, text: str, image_bytes: bytes) -> str:
        model = self._get_model()
        response = model.generate_content(
            [
                f"Summarize both the text and image together in context:\n\n{text}",
                {"mime_type": "image/png", "data": image_bytes},
            ]
        )
        return response.text