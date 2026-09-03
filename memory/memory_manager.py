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

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search key, value, and category using SQLite parameterized LIKE queries.

        Returns matching memories ordered by relevance, then updated_at.
        """
        cleaned = query.strip()
        if not cleaned or limit < 1:
            return []

        import re

        normalized_phrase = cleaned.rstrip(".!?").strip().lower()
        stopwords = {
            "what", "is", "are", "was", "were", "the", "a", "an", "and", "or",
            "do", "does", "did", "you", "your", "my", "me", "i", "to", "in",
            "on", "at", "of", "for", "with", "about", "tell", "know", "can",
        }
        all_words = [w.lower() for w in re.findall(r"\b\w+\b", cleaned)]
        keywords = [w for w in all_words if len(w) >= 3 and w not in stopwords]

        terms = []
        if normalized_phrase:
            terms.append(normalized_phrase)
        for kw in keywords:
            if kw not in terms:
                terms.append(kw)

        if not terms:
            terms = [cleaned.lower()]

        where_clauses = []
        where_params = []

        relevance_cases = []
        relevance_params = []

        primary = terms[0]
        primary_pattern = f"%{primary}%"

        relevance_cases.append("WHEN LOWER(key) = ? THEN 100")
        relevance_params.append(primary)
        relevance_cases.append("WHEN LOWER(value) = ? THEN 90")
        relevance_params.append(primary)
        relevance_cases.append("WHEN LOWER(category) = ? THEN 80")
        relevance_params.append(primary)

        relevance_cases.append("WHEN LOWER(key) LIKE ? THEN 60")
        relevance_params.append(primary_pattern)
        relevance_cases.append("WHEN LOWER(value) LIKE ? THEN 50")
        relevance_params.append(primary_pattern)
        relevance_cases.append("WHEN LOWER(category) LIKE ? THEN 40")
        relevance_params.append(primary_pattern)

        for kw in terms[1:]:
            kw_pattern = f"%{kw}%"
            relevance_cases.append("WHEN LOWER(key) LIKE ? THEN 25")
            relevance_params.append(kw_pattern)
            relevance_cases.append("WHEN LOWER(value) LIKE ? THEN 20")
            relevance_params.append(kw_pattern)
            relevance_cases.append("WHEN LOWER(category) LIKE ? THEN 15")
            relevance_params.append(kw_pattern)

        for term in terms:
            term_pattern = f"%{term}%"
            where_clauses.append("(LOWER(key) LIKE ? OR LOWER(value) LIKE ? OR LOWER(category) LIKE ?)")
            where_params.extend([term_pattern, term_pattern, term_pattern])

        relevance_sql = "CASE " + " ".join(relevance_cases) + " ELSE 10 END"
        where_sql = " OR ".join(where_clauses)

        query_sql = f"""
            SELECT id, key, value, category, created_at, updated_at,
                   ({relevance_sql}) AS relevance
            FROM memories
            WHERE {where_sql}
            ORDER BY relevance DESC, updated_at DESC
            LIMIT ?
        """
        all_params = tuple(relevance_params + where_params + [limit])

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query_sql, all_params)
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "key": row["key"],
                    "value": row["value"],
                    "category": row["category"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]
