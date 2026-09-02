"""Orchestrator module for TOM."""
from typing import Iterator
from core.brain import TOMBrain


class TOMOrchestrator:
    """Orchestrates input flow, routing user requests to the brain and returning output."""

    def __init__(self, brain: TOMBrain | None = None) -> None:
        self.brain = brain or TOMBrain()

    def process(self, user_text: str) -> str:
        """Receive user text, delegate to the brain, and return the response."""
        return self.brain.respond(user_text)

    def stream(self, user_text: str) -> Iterator[str]:
        """Receive user text, delegate streaming to the brain, and yield response chunks."""
        yield from self.brain.stream(user_text)
