"""Brain module for TOM."""
import json
import urllib.error
import urllib.request
from typing import Iterator
from core.config import config


class TOMBrain:
    """Core brain responsible for processing inputs and generating responses via Ollama."""

    def __init__(self) -> None:
        self.name = config.APP_NAME
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.OLLAMA_MODEL
        self.system_prompt = config.TOM_SYSTEM_PROMPT

    def _build_messages(
        self, text: str, history: list[dict[str, str]] | None = None
    ) -> list[dict[str, str]]:
        """Construct the message list for Ollama including system prompt, history, and current user text."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Ensure the current user message is included without duplication
        if not (
            len(messages) > 1
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == text
        ):
            messages.append({"role": "user", "content": text})

        return messages

    def respond(self, text: str, history: list[dict[str, str]] | None = None) -> str:
        """Process user input text with conversation history and return a response from Ollama."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return "I am listening."

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self._build_messages(cleaned_text, history),
            "stream": False,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                return response_data.get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            return f"Ollama HTTP error ({e.code}): {e.reason}"
        except urllib.error.URLError as e:
            return f"Ollama connection error: {e.reason}"
        except TimeoutError:
            return "Ollama request timed out."
        except Exception as e:
            return f"Unexpected error communicating with Ollama: {e}"

    def stream(self, text: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
        """Process user input text with conversation history and stream response chunks from Ollama."""
        cleaned_text = text.strip()
        if not cleaned_text:
            yield "I am listening."
            return

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": self._build_messages(cleaned_text, history),
            "stream": True,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                for raw_line in response:
                    if isinstance(raw_line, bytes):
                        line_str = raw_line.decode("utf-8").strip()
                    else:
                        line_str = str(raw_line).strip()
                    if not line_str:
                        continue
                    data = json.loads(line_str)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except urllib.error.HTTPError as e:
            yield f"Ollama HTTP error ({e.code}): {e.reason}"
        except urllib.error.URLError as e:
            yield f"Ollama connection error: {e.reason}"
        except TimeoutError:
            yield "Ollama request timed out."
        except Exception as e:
            yield f"Unexpected error communicating with Ollama: {e}"
