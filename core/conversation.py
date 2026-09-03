"""Conversation context manager for TOM."""
from core.config import config


class ConversationManager:
    """Manages in-memory session conversation context."""

    def __init__(self, max_messages: int | None = None) -> None:
        """Initialize ConversationManager with a maximum message history limit."""
        if max_messages is None:
            max_messages = getattr(config, "MAX_CONVERSATION_MESSAGES", 20)
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: list[dict[str, str]] = []

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation context."""
        self._add_message("user", text)

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the conversation context."""
        self._add_message("assistant", text)

    def _add_message(self, role: str, content: str) -> None:
        """Append a message and trim to max_messages keeping only the newest."""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation messages."""
        return [dict(msg) for msg in self._messages]

    def clear(self) -> None:
        """Clear all messages from the conversation context."""
        self._messages.clear()
