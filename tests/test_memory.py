"""Tests for TOM MemoryManager."""
import tempfile
import unittest
from pathlib import Path
from memory.memory_manager import MemoryManager, DEFAULT_DB_PATH


class TestMemoryManager(unittest.TestCase):
    """Test suite for MemoryManager using a temporary database."""

    def setUp(self):
        """Create a temporary directory and isolated SQLite database for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.memory = MemoryManager(db_path=self.temp_db_path)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def test_insert_and_recall(self):
        """Test inserting a memory and recalling it by key."""
        self.memory.remember("username", "Alice", category="profile")
        recalled = self.memory.recall("username")
        self.assertEqual(recalled, "Alice")

    def test_recall_nonexistent(self):
        """Test recalling a key that does not exist returns None."""
        self.assertIsNone(self.memory.recall("nonexistent_key"))

    def test_update_existing_key(self):
        """Test updating an existing key updates value/category without creating duplicates."""
        self.memory.remember("theme", "light", category="settings")
        all_before = self.memory.get_all()
        self.assertEqual(len(all_before), 1)
        created_at_original = all_before[0]["created_at"]

        # Update the key
        self.memory.remember("theme", "dark", category="appearance")
        all_after = self.memory.get_all()

        # No duplicate created
        self.assertEqual(len(all_after), 1)
        self.assertEqual(all_after[0]["key"], "theme")
        self.assertEqual(all_after[0]["value"], "dark")
        self.assertEqual(all_after[0]["category"], "appearance")
        self.assertEqual(all_after[0]["created_at"], created_at_original)
        self.assertEqual(self.memory.recall("theme"), "dark")

    def test_get_all(self):
        """Test get_all retrieves all memories and supports category filtering."""
        self.memory.remember("item1", "val1", category="work")
        self.memory.remember("item2", "val2", category="personal")
        self.memory.remember("item3", "val3", category="work")

        all_items = self.memory.get_all()
        self.assertEqual(len(all_items), 3)

        keys = [item["key"] for item in all_items]
        self.assertIn("item1", keys)
        self.assertIn("item2", keys)
        self.assertIn("item3", keys)

        # Test category filtering
        work_items = self.memory.get_all(category="work")
        self.assertEqual(len(work_items), 2)
        for item in work_items:
            self.assertEqual(item["category"], "work")

        personal_items = self.memory.get_all(category="personal")
        self.assertEqual(len(personal_items), 1)
        self.assertEqual(personal_items[0]["key"], "item2")

    def test_forget(self):
        """Test deleting a memory by key."""
        self.memory.remember("temp_note", "buy milk")
        self.assertEqual(self.memory.recall("temp_note"), "buy milk")

        # Forget existing key returns True
        result = self.memory.forget("temp_note")
        self.assertTrue(result)
        self.assertIsNone(self.memory.recall("temp_note"))

        # Forget nonexistent key returns False
        result_again = self.memory.forget("temp_note")
        self.assertFalse(result_again)

    def test_clear(self):
        """Test clearing all memories from the database."""
        self.memory.remember("k1", "v1")
        self.memory.remember("k2", "v2")
        self.memory.remember("k3", "v3")
        self.assertEqual(len(self.memory.get_all()), 3)

        self.memory.clear()
        self.assertEqual(self.memory.get_all(), [])
        self.assertIsNone(self.memory.recall("k1"))
        self.assertIsNone(self.memory.recall("k2"))
        self.assertIsNone(self.memory.recall("k3"))

    def test_parameterized_sql(self):
        """Test that keys and values with SQL injection characters are handled safely."""
        malicious_key = "user' OR '1'='1"
        malicious_val = "val'); DROP TABLE memories; --"
        self.memory.remember(malicious_key, malicious_val)

        self.assertEqual(self.memory.recall(malicious_key), malicious_val)

        # Ensure table was not dropped
        all_items = self.memory.get_all()
        self.assertEqual(len(all_items), 1)
        self.assertEqual(all_items[0]["key"], malicious_key)

    def test_default_db_path_is_relative_project_safe(self):
        """Verify DEFAULT_DB_PATH points to tom_memory.db inside the memory directory."""
        self.assertEqual(DEFAULT_DB_PATH.name, "tom_memory.db")
        self.assertEqual(DEFAULT_DB_PATH.parent.name, "memory")


if __name__ == "__main__":
    unittest.main()
