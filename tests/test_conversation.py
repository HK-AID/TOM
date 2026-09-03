"""Tests for TOM conversation context management and routing."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.brain import TOMBrain
from core.config import config
from core.conversation import ConversationManager
from core.orchestrator import TOMOrchestrator
from memory.memory_manager import MemoryManager


class TestConversationManager(unittest.TestCase):
    """Unit tests for ConversationManager."""

    def test_message_ordering(self):
        """Verify messages are stored and retrieved in strict chronological order."""
        cm = ConversationManager(max_messages=10)
        cm.add_user_message("Hello")
        cm.add_assistant_message("Hi there!")
        cm.add_user_message("How are you?")
        cm.add_assistant_message("I am well.")

        messages = cm.get_messages()
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "user", "content": "Hello"})
        self.assertEqual(messages[1], {"role": "assistant", "content": "Hi there!"})
        self.assertEqual(messages[2], {"role": "user", "content": "How are you?"})
        self.assertEqual(messages[3], {"role": "assistant", "content": "I am well."})

    def test_maximum_history_size(self):
        """Verify oldest messages are dropped when the message limit is exceeded."""
        cm = ConversationManager(max_messages=4)
        cm.add_user_message("msg 1")
        cm.add_assistant_message("msg 2")
        cm.add_user_message("msg 3")
        cm.add_assistant_message("msg 4")
        cm.add_user_message("msg 5")

        messages = cm.get_messages()
        self.assertEqual(len(messages), 4)
        # Oldest message "msg 1" dropped
        self.assertEqual(messages[0]["content"], "msg 2")
        self.assertEqual(messages[1]["content"], "msg 3")
        self.assertEqual(messages[2]["content"], "msg 4")
        self.assertEqual(messages[3]["content"], "msg 5")

    def test_clear(self):
        """Verify clear() removes all conversation messages."""
        cm = ConversationManager(max_messages=10)
        cm.add_user_message("Hello")
        cm.add_assistant_message("Hi")
        self.assertEqual(len(cm.get_messages()), 2)

        cm.clear()
        self.assertEqual(cm.get_messages(), [])


class TestBrainConversationContext(unittest.TestCase):
    """Unit tests for TOMBrain conversation context handling."""

    def test_build_messages_structure(self):
        """Verify Ollama payload contains system message, history, and current message without duplication."""
        brain = TOMBrain()
        history = [
            {"role": "user", "content": "My name is John"},
            {"role": "assistant", "content": "Nice to meet you, John."},
        ]

        messages = brain._build_messages("What is my name?", history=history)

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0], {"role": "system", "content": config.TOM_SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "My name is John"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "Nice to meet you, John."})
        self.assertEqual(messages[3], {"role": "user", "content": "What is my name?"})

    def test_no_duplicate_user_message(self):
        """Verify the user message is not duplicated if history already ends with it."""
        brain = TOMBrain()
        history = [
            {"role": "user", "content": "What is my name?"}
        ]

        messages = brain._build_messages("What is my name?", history=history)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is my name?")

    @patch("urllib.request.urlopen")
    def test_respond_sends_history_in_payload(self, mock_urlopen):
        """Verify respond() sends full history in HTTP request."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"message": {"role": "assistant", "content": "Your name is John."}}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        brain = TOMBrain()
        history = [
            {"role": "user", "content": "My name is John"},
            {"role": "assistant", "content": "Hello John"},
        ]
        response = brain.respond("What is my name?", history=history)
        self.assertEqual(response, "Your name is John.")

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(len(payload["messages"]), 4)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["content"], "My name is John")
        self.assertEqual(payload["messages"][2]["content"], "Hello John")
        self.assertEqual(payload["messages"][3]["content"], "What is my name?")


class TestOrchestratorConversationContext(unittest.TestCase):
    """Integration tests for TOMOrchestrator multi-turn conversation and memory separation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.memory = MemoryManager(db_path=self.temp_db_path)
        self.conv = ConversationManager(max_messages=10)

        self.mock_brain = MagicMock(spec=TOMBrain)
        self.mock_brain.respond.return_value = "Answer 1"
        self.mock_brain.stream.side_effect = lambda text, history=None: iter(["Chunk ", "1"])

        self.orchestrator = TOMOrchestrator(
            brain=self.mock_brain,
            memory=self.memory,
            conversation=self.conv,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multiturn_context_accumulates_in_orchestrator(self):
        """Verify orchestrator tracks conversation history across multiple turns."""
        self.mock_brain.respond.side_effect = ["Nice to meet you.", "Your name is Bob."]

        # Turn 1
        resp1 = self.orchestrator.process("My name is Bob")
        self.assertEqual(resp1, "Nice to meet you.")
        self.mock_brain.respond.assert_called_with("My name is Bob", history=[], memories=[])

        # Turn 2
        resp2 = self.orchestrator.process("What is my name?")
        self.assertEqual(resp2, "Your name is Bob.")
        expected_history = [
            {"role": "user", "content": "My name is Bob"},
            {"role": "assistant", "content": "Nice to meet you."},
        ]
        self.mock_brain.respond.assert_called_with("What is my name?", history=expected_history, memories=[])

        # Verify conversation state
        messages = self.orchestrator.conversation.get_messages()
        self.assertEqual(len(messages), 4)

    def test_streaming_with_conversation_context(self):
        """Verify streaming properly yields tokens and appends conversation turns."""
        self.mock_brain.stream.side_effect = [
            iter(["Hello ", "there!"]),
            iter(["I am ", "good."]),
        ]

        # Turn 1 streaming
        chunks1 = list(self.orchestrator.stream("Hello"))
        self.assertEqual("".join(chunks1), "Hello there!")
        self.mock_brain.stream.assert_called_with("Hello", history=[], memories=[])

        # Turn 2 streaming receives Turn 1 history
        chunks2 = list(self.orchestrator.stream("How are you?"))
        self.assertEqual("".join(chunks2), "I am good.")
        expected_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        self.mock_brain.stream.assert_called_with("How are you?", history=expected_history, memories=[])

        # Verify conversation state
        messages = self.orchestrator.conversation.get_messages()
        self.assertEqual(len(messages), 4)

    def test_memory_commands_not_in_conversation_history(self):
        """Verify memory commands are executed deterministically and NOT recorded into conversation history."""
        # Run remember command
        rem_resp = self.orchestrator.process("remember that my key is secret123")
        self.assertIn("secret123", rem_resp)

        # Run recall command
        recall_resp = self.orchestrator.process("what do you remember")
        self.assertIn("secret123", recall_resp)

        # Run forget command
        forget_resp = self.orchestrator.process("forget that my key is secret123")
        self.assertIn("forgotten", forget_resp.lower())

        # Conversation history must remain empty
        self.assertEqual(self.orchestrator.conversation.get_messages(), [])
        self.mock_brain.respond.assert_not_called()
        self.mock_brain.stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
