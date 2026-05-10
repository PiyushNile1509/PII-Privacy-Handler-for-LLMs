"""
Gemini LLM Handler with multi-key rotation for rate limit handling
"""

import google.generativeai as genai

API_KEYS = [
    "AIzaSyDtNUvXpp63Sjl2GHEmaLtw831zEe8Cuz8",  # Key 1
    "AIzaSyBnDG9B6sfPgrju3k_LT3-4uxqHWvlKqKk",  # Key 2 - add your keys here
    "AIzaSyCy9xvkM4dq77hJ2_C345cchctWZ2jbwoA",  # Key 3
]

class GeminiHandler:
    def __init__(self):
        self._key_index = 0
        self._configure_current_key()

    def _configure_current_key(self):
        genai.configure(api_key=API_KEYS[self._key_index])
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print(f"[GEMINI] Using API key index {self._key_index}")

    def _rotate_key(self):
        self._key_index = (self._key_index + 1) % len(API_KEYS)
        self._configure_current_key()

    def generate_response(self, masked_query: str, context: str = "General") -> str:
        """Generate response using Gemini API, rotating keys on rate limit errors"""
        prompt = f"You are a helpful AI assistant. The user's personal information has been masked for privacy. Respond naturally and helpfully to: {masked_query}"

        for attempt in range(len(API_KEYS)):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=200,
                        temperature=0.7
                    )
                )
                return response.text.strip()
            except Exception as e:
                error = str(e).lower()
                if any(x in error for x in ['quota', 'rate', '429', 'exhausted']):
                    print(f"[GEMINI] Rate limit on key {self._key_index}, rotating...")
                    self._rotate_key()
                else:
                    print(f"[ERROR] Gemini API error: {e}")
                    break

        return "I understand your message. How can I help you further?"
