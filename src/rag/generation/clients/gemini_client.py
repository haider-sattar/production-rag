import os

from google import genai


class GeminiClient:
    """
    Thin adapter around the Gemini API.

    The rest of the RAG system interacts with this class through
    the LLMClient protocol defined in generator.py.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        resolved_api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        self.model_name = model_name
        self.client = genai.Client(
            api_key=resolved_api_key,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send the system instructions and grounded user prompt
        to Gemini and return the generated text.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.1,
            },
        )

        if not response.text:
            return ""

        return response.text