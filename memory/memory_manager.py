"""Memory manager module for TOM using SQLite."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "tom_memory.db"


class MemoryManager:
    """Manages persistent key-value memory for TOM."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize MemoryManager with a database path (defaults to relative memory/tom_memory.db)."""
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite connections ensuring clean closure."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the memories table schema if it does not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def remember(self, key: str, value: str, category: str = "general") -> None:
        """Store or update a memory key-value pair without creating duplicates."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM memories WHERE key = ?", (key,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE memories
                    SET value = ?, category = ?, updated_at = ?
                    WHERE key = ?
                    """,
                    (value, category, now, key),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO memories (key, value, category, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, value, category, now, now),
                )
            conn.commit()

    def recall(self, key: str) -> str | None:
        """Recall the stored value for a given key, or None if not found."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def get_all(self, category: str | None = None) -> list[dict]:
        """Retrieve all memories as dictionaries, optionally filtered by category."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if category is not None:
                cursor.execute(
                    """
                    SELECT id, key, value, category, created_at, updated_at
                    FROM memories
                    WHERE category = ?
                    ORDER BY id ASC
                    """,
                    (category,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, key, value, category, created_at, updated_at
                    FROM memories
                    ORDER BY id ASC
                    """
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def forget(self, key: str) -> bool:
        """Delete a memory by key. Returns True if removed, False otherwise."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all memories from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            conn.commit()
