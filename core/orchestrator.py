"""Orchestrator module for TOM."""
from typing import Iterator
from core.brain import TOMBrain
from core.conversation import ConversationManager
from memory.memory_manager import MemoryManager


class TOMOrchestrator:
    """Orchestrates input flow, routing user requests to memory or the brain."""

    def __init__(
        self,
        brain: TOMBrain | None = None,
        memory: MemoryManager | None = None,
        conversation: ConversationManager | None = None,
    ) -> None:
        self.brain = brain or TOMBrain()
        self.memory = memory or MemoryManager()
        self.conversation = conversation or ConversationManager()

    def _handle_memory_command(self, user_text: str) -> str | None:
        """Deterministically parse and handle explicit memory commands."""
        normalized = user_text.strip()
        if not normalized:
            return None

        cleaned = normalized.rstrip(".!?").strip()
        lower = cleaned.lower()

        # Recall commands
        if lower in ("what do you remember", "what do you remember about me"):
            all_memories = self.memory.get_all()
            if not all_memories:
                return "I don't remember anything yet."
            lines = ["Here is what I remember:"]
            for item in all_memories:
                if item["key"] == item["value"]:
                    lines.append(f"- {item['value']}")
                else:
                    lines.append(f"- {item['key']}: {item['value']}")
            return "\n".join(lines)

        # Remember commands: "remember that ..." or "remember ..."
        if lower.startswith("remember that "):
            content = normalized[len("remember that "):].strip().rstrip(".!?").strip()
            if content:
                self.memory.remember(content, content)
                return f"I will remember that: {content}"
            return None

        if lower.startswith("remember "):
            content = normalized[len("remember "):].strip().rstrip(".!?").strip()
            if content:
                self.memory.remember(content, content)
                return f"I will remember that: {content}"
            return None

        # Forget commands: "forget that ..." or "forget ..."
        if lower.startswith("forget that "):
            target = normalized[len("forget that "):].strip().rstrip(".!?").strip()
            if target:
                return self._forget_memory(target)
            return None

        if lower.startswith("forget "):
            target = normalized[len("forget "):].strip().rstrip(".!?").strip()
            if target:
                return self._forget_memory(target)
            return None

        return None

    def _forget_memory(self, target: str) -> str:
        """Remove matching memory by exact key, case-insensitive match, or substring match."""
        # 1. Direct exact match
        if self.memory.forget(target):
            return f"I have forgotten that: {target}"

        # 2. Case-insensitive exact or substring match in stored memories
        all_memories = self.memory.get_all()
        target_lower = target.lower()

        for item in all_memories:
            if item["key"].lower() == target_lower or item["value"].lower() == target_lower:
                self.memory.forget(item["key"])
                return f"I have forgotten that: {item['value']}"

        for item in all_memories:
            if target_lower in item["key"].lower() or target_lower in item["value"].lower():
                self.memory.forget(item["key"])
                return f"I have forgotten that: {item['value']}"

        return f"I couldn't find any memory matching '{target}'."

    def process(self, user_text: str) -> str:
        """Receive user text, check for memory commands, or delegate to brain with context."""
        memory_response = self._handle_memory_command(user_text)
        if memory_response is not None:
            return memory_response

        cleaned_text = user_text.strip()
        history = self.conversation.get_messages()
        response = self.brain.respond(cleaned_text, history=history)

        if cleaned_text and response:
            self.conversation.add_user_message(cleaned_text)
            self.conversation.add_assistant_message(response)

        return response

    def stream(self, user_text: str) -> Iterator[str]:
        """Receive user text, check for memory commands, or delegate streaming to brain with context."""
        memory_response = self._handle_memory_command(user_text)
        if memory_response is not None:
            yield memory_response
            return

        cleaned_text = user_text.strip()
        history = self.conversation.get_messages()
        collected_chunks = []
        for chunk in self.brain.stream(cleaned_text, history=history):
            collected_chunks.append(chunk)
            yield chunk

        full_response = "".join(collected_chunks)
        if cleaned_text and full_response:
            self.conversation.add_user_message(cleaned_text)
            self.conversation.add_assistant_message(full_response)
