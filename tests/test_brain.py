"""Tests for TOMBrain and Ollama integration."""
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch
from core.brain import TOMBrain
from core.config import config
from core.orchestrator import TOMOrchestrator


class TestTOMBrain(unittest.TestCase):
    """Test suite for TOMBrain functionality."""

    def test_empty_input(self):
        """Empty input should return 'I am listening.' prompt."""
        brain = TOMBrain()
        self.assertEqual(brain.respond("   "), "I am listening.")

    @patch("urllib.request.urlopen")
    def test_respond_mock_success(self, mock_urlopen):
        """Mocked Ollama response test verifying system message is sent."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"message": {"role": "assistant", "content": "Hello from TOM!"}}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        brain = TOMBrain()
        response = brain.respond("Hello")
        self.assertEqual(response, "Hello from TOM!")

        # Verify request payload contains system message and user message
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["messages"][0], {"role": "system", "content": config.TOM_SYSTEM_PROMPT})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hello"})
        self.assertFalse(payload["stream"])

    @patch("urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        """Network error handling test."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        brain = TOMBrain()
        response = brain.respond("Hello")
        self.assertIn("Ollama connection error", response)

    def test_stream_empty_input(self):
        """Streaming empty input should yield 'I am listening.' prompt."""
        brain = TOMBrain()
        chunks = list(brain.stream("   "))
        self.assertEqual(chunks, ["I am listening."])

    @patch("urllib.request.urlopen")
    def test_stream_mock_success(self, mock_urlopen):
        """Mocked Ollama streaming chunks test verifying system message is sent."""
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [
            b'{"message": {"role": "assistant", "content": "Hello "}, "done": false}\n',
            b'{"message": {"role": "assistant", "content": "world"}, "done": false}\n',
            b'{"message": {"role": "assistant", "content": "!"}, "done": false}\n',
            b'{"message": {"role": "assistant", "content": ""}, "done": true}\n',
        ]
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        brain = TOMBrain()
        chunks = list(brain.stream("Hi"))
        self.assertEqual(chunks, ["Hello ", "world", "!"])
        self.assertEqual("".join(chunks), "Hello world!")

        # Verify request payload contains system message and user message
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["messages"][0], {"role": "system", "content": config.TOM_SYSTEM_PROMPT})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hi"})
        self.assertTrue(payload["stream"])

    @patch("urllib.request.urlopen")
    def test_stream_connection_error(self, mock_urlopen):
        """Streaming network error handling test."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        brain = TOMBrain()
        chunks = list(brain.stream("Hi"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("Ollama connection error", chunks[0])

    @patch("urllib.request.urlopen")
    def test_orchestrator_stream_mock(self, mock_urlopen):
        """Test orchestrator stream delegates to brain stream."""
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [
            b'{"message": {"role": "assistant", "content": "Chunk1"}, "done": false}\n',
            b'{"message": {"role": "assistant", "content": "Chunk2"}, "done": true}\n',
        ]
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        orchestrator = TOMOrchestrator()
        chunks = list(orchestrator.stream("Hello"))
        self.assertEqual(chunks, ["Chunk1", "Chunk2"])

    def test_live_qwen_integration(self):
        """Live test verifying non-streaming communication with local Ollama qwen2.5:3b model."""
        orchestrator = TOMOrchestrator()
        response = orchestrator.process("Respond with only the single word: PONG")
        self.assertTrue(len(response) > 0)
        self.assertNotIn("Ollama connection error", response)
        self.assertIn("PONG", response.upper())

    def test_live_qwen_stream_integration(self):
        """Live test verifying streaming communication with local Ollama qwen2.5:3b model."""
        orchestrator = TOMOrchestrator()
        chunks = list(orchestrator.stream("Respond with only the single word: PONG"))
        full_response = "".join(chunks)
        self.assertTrue(len(chunks) > 0)
        self.assertNotIn("Ollama connection error", full_response)
        self.assertIn("PONG", full_response.upper())

    def test_live_tom_identity(self):
        """Live test verifying TOM introduces itself as TOM and not as Qwen."""
        orchestrator = TOMOrchestrator()
        chunks = list(orchestrator.stream("What is your name? Answer in one short sentence."))
        full_response = "".join(chunks)
        self.assertIn("TOM", full_response)
        self.assertNotIn("Qwen", full_response)

    def test_build_messages_with_memories(self):
        """Verify long-term memory is labeled and injected as reference context."""
        brain = TOMBrain()
        memories = [
            {"key": "pet_name", "value": "Luna"},
            {"key": "diet", "value": "vegetarian"},
        ]
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages = brain._build_messages("What does my pet eat?", history=history, memories=memories)

        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], config.TOM_SYSTEM_PROMPT)

        # Long-term memory context message
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("Long-term memory:", messages[1]["content"])
        self.assertIn("Luna", messages[1]["content"])
        self.assertIn("vegetarian", messages[1]["content"])

        # History follows
        self.assertEqual(messages[2], {"role": "user", "content": "Hello"})
        self.assertEqual(messages[3], {"role": "assistant", "content": "Hi there!"})

        # Current user query
        self.assertEqual(messages[4], {"role": "user", "content": "What does my pet eat?"})

    def test_build_messages_without_memories(self):
        """Verify no memory message is injected when memories is empty or None."""
        brain = TOMBrain()
        messages_none = brain._build_messages("Hi", memories=None)
        self.assertEqual(len(messages_none), 2)
        self.assertEqual(messages_none[0]["role"], "system")
        self.assertEqual(messages_none[1]["role"], "user")

        messages_empty = brain._build_messages("Hi", memories=[])
        self.assertEqual(len(messages_empty), 2)
        self.assertEqual(messages_empty[0]["role"], "system")
        self.assertEqual(messages_empty[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
