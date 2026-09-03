"""Tests for TOMOrchestrator memory integration and routing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from core.brain import TOMBrain
from core.orchestrator import TOMOrchestrator
from memory.memory_manager import MemoryManager


class TestTOMOrchestrator(unittest.TestCase):
    """Test suite for TOMOrchestrator memory commands and routing."""

    def setUp(self):
        """Set up isolated temporary SQLite database and orchestrator for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.memory = MemoryManager(db_path=self.temp_db_path)

        # Mock brain to test routing without depending on Ollama
        self.mock_brain = MagicMock(spec=TOMBrain)
        self.mock_brain.respond.return_value = "Mock brain response"
        self.mock_brain.stream.return_value = iter(["Mock ", "brain ", "stream"])

        self.orchestrator = TOMOrchestrator(brain=self.mock_brain, memory=self.memory)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_remember_that_command(self):
        """Test 'remember that ...' stores the content and returns confirmation."""
        response = self.orchestrator.process("remember that my name is Alice")
        self.assertIn("Alice", response)

        # Verify stored in MemoryManager
        stored = self.memory.get_all()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["value"], "my name is Alice")

        # Mock brain was not called
        self.mock_brain.respond.assert_not_called()

    def test_remember_command(self):
        """Test 'remember ...' stores the content."""
        response = self.orchestrator.process("Remember my favorite color is blue.")
        self.assertIn("blue", response)

        stored = self.memory.get_all()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["value"], "my favorite color is blue")

    def test_recall_when_empty(self):
        """Test 'what do you remember' when no memories exist."""
        response = self.orchestrator.process("what do you remember")
        self.assertIn("don't remember anything", response.lower())

    def test_recall_with_memories(self):
        """Test 'what do you remember' and 'what do you remember about me' with stored items."""
        self.orchestrator.process("remember that I like coffee")
        self.orchestrator.process("remember that I live in New York")

        recall1 = self.orchestrator.process("what do you remember")
        self.assertIn("I like coffee", recall1)
        self.assertIn("I live in New York", recall1)

        recall2 = self.orchestrator.process("What do you remember about me?")
        self.assertIn("I like coffee", recall2)
        self.assertIn("I live in New York", recall2)

    def test_forget_that_command(self):
        """Test 'forget that ...' removes the memory."""
        self.orchestrator.process("remember that I like coffee")
        self.assertEqual(len(self.memory.get_all()), 1)

        forget_resp = self.orchestrator.process("forget that I like coffee")
        self.assertIn("forgotten", forget_resp.lower())
        self.assertEqual(len(self.memory.get_all()), 0)

    def test_forget_command_with_substring(self):
        """Test 'forget ...' removes matching memory even by substring."""
        self.orchestrator.process("remember that I like green tea")
        self.assertEqual(len(self.memory.get_all()), 1)

        forget_resp = self.orchestrator.process("forget green tea")
        self.assertIn("forgotten", forget_resp.lower())
        self.assertEqual(len(self.memory.get_all()), 0)

    def test_forget_nonexistent_command(self):
        """Test forgetting something that does not exist."""
        forget_resp = self.orchestrator.process("forget that dogs fly")
        self.assertIn("couldn't find", forget_resp.lower())

    def test_normal_message_routes_to_brain_stream(self):
        """Normal messages must be routed to TOMBrain.stream()."""
        chunks = list(self.orchestrator.stream("What is an array in Java?"))
        self.assertEqual("".join(chunks), "Mock brain stream")
        self.mock_brain.stream.assert_called_once_with("What is an array in Java?", history=[], memories=[])

    def test_normal_message_routes_to_brain_process(self):
        """Normal messages in process() must route to TOMBrain.respond()."""
        response = self.orchestrator.process("What is an array in Java?")
        self.assertEqual(response, "Mock brain response")
        self.mock_brain.respond.assert_called_once_with("What is an array in Java?", history=[], memories=[])

    def test_stream_handles_memory_commands(self):
        """Verify orchestrator.stream() yields memory responses for interactive loop."""
        chunks = list(self.orchestrator.stream("remember that I am a developer"))
        full_response = "".join(chunks)
        self.assertIn("developer", full_response)
        self.mock_brain.stream.assert_not_called()

        recall_chunks = list(self.orchestrator.stream("what do you remember?"))
        recall_response = "".join(recall_chunks)
        self.assertIn("developer", recall_response)
        self.mock_brain.stream.assert_not_called()

    def test_relevant_memory_injection(self):
        """Verify relevant memories are retrieved and passed to brain.respond."""
        self.memory.remember("favorite_framework", "FastAPI", category="preferences")
        self.memory.remember("unrelated_fact", "The sky is blue", category="general")

        self.orchestrator.process("What is my favorite framework?")

        # Check call arguments to mock_brain.respond
        call_kwargs = self.mock_brain.respond.call_args[1]
        injected = call_kwargs.get("memories", [])
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["key"], "favorite_framework")
        self.assertEqual(injected[0]["value"], "FastAPI")

    def test_no_match_memory_injection(self):
        """Verify unrelated memories are NOT injected when no memories match the query."""
        self.memory.remember("favorite_framework", "FastAPI", category="preferences")

        self.orchestrator.process("What is the speed of light?")

        call_kwargs = self.mock_brain.respond.call_args[1]
        injected = call_kwargs.get("memories", [])
        self.assertEqual(injected, [])

    def test_streaming_with_injected_memory(self):
        """Verify relevant memories are retrieved and passed to brain.stream."""
        self.memory.remember("home_city", "Berlin", category="profile")

        chunks = list(self.orchestrator.stream("Tell me about my home city"))
        self.assertEqual("".join(chunks), "Mock brain stream")

        call_kwargs = self.mock_brain.stream.call_args[1]
        injected = call_kwargs.get("memories", [])
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["key"], "home_city")
        self.assertEqual(injected[0]["value"], "Berlin")

    def test_memory_commands_remain_deterministic_without_llm(self):
        """Verify explicit memory commands do not invoke search injection or LLM calls."""
        resp = self.orchestrator.process("remember that dog_name is Buddy")
        self.assertIn("Buddy", resp)

        self.mock_brain.respond.assert_not_called()
        self.mock_brain.stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
