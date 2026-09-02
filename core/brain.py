"""Brain module for TOM."""
from core.config import config


class TOMBrain:
    """Core brain responsible for processing inputs and generating responses."""

    def __init__(self) -> None:
        self.name = config.APP_NAME

    def respond(self, text: str) -> str:
        """Process user input text and return a response."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return "I am listening."

        return f"[{self.name}] Acknowledged: {cleaned_text}"
