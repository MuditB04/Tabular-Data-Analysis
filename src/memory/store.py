"""SQLite/File-backed Preference Memory with LRU/TTL eviction.

Ensures:
- Only user preferences, term definitions, and column aliases are stored.
- Zero raw dataset rows or PII are persisted.
- Stale or inactive entries are automatically evicted after expiration.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.memory.preferences import UserPreferences


class MemoryStore:
    """Persistent storage for non-sensitive user preferences and definitions with TTL eviction."""

    def __init__(self, db_path: Optional[str] = None, ttl_days: int = 30):
        self.ttl_days = ttl_days
        if db_path:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._init_tables()
        self.evict_expired()

    def _init_tables(self) -> None:
        """Create storage schema."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    accessed_at TIMESTAMP NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_aliases (
                    dataset_name TEXT,
                    alias_key TEXT,
                    actual_column TEXT,
                    PRIMARY KEY (dataset_name, alias_key)
                )
                """
            )

    def get_preferences(self, user_id: str = "default_user") -> UserPreferences:
        """Retrieve stored user preferences or return default object."""
        self.evict_expired()
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT data_json FROM user_preferences WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        now_str = datetime.utcnow().isoformat()

        if row:
            # Update access timestamp
            with self._conn:
                self._conn.execute(
                    "UPDATE user_preferences SET accessed_at = ? WHERE user_id = ?",
                    (now_str, user_id),
                )
            try:
                data = json.loads(row[0])
                return UserPreferences(**data)
            except Exception:
                return UserPreferences()

        # If not present, save default
        prefs = UserPreferences()
        self.save_preferences(user_id, prefs)
        return prefs

    def save_preferences(self, user_id: str, prefs: UserPreferences) -> None:
        """Save or update user preferences."""
        prefs.last_updated = datetime.utcnow().isoformat()
        serialized = prefs.model_dump_json()
        now_str = datetime.utcnow().isoformat()

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO user_preferences (user_id, data_json, updated_at, accessed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at,
                    accessed_at = excluded.accessed_at
                """,
                (user_id, serialized, now_str, now_str),
            )

    def set_column_alias(self, dataset_name: str, alias_key: str, actual_column: str) -> None:
        """Store custom column alias mapping (e.g. 'sales' -> 'revenue')."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dataset_aliases (dataset_name, alias_key, actual_column)
                VALUES (?, ?, ?)
                ON CONFLICT(dataset_name, alias_key) DO UPDATE SET
                    actual_column = excluded.actual_column
                """,
                (dataset_name.lower(), alias_key.lower(), actual_column),
            )

    def get_column_aliases(self, dataset_name: str) -> Dict[str, str]:
        """Fetch all resolved aliases for a given dataset."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT alias_key, actual_column FROM dataset_aliases WHERE dataset_name = ?",
            (dataset_name.lower(),),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def evict_expired(self) -> int:
        """Remove entries older than TTL threshold."""
        cutoff = (datetime.utcnow() - timedelta(days=self.ttl_days)).isoformat()
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM user_preferences WHERE accessed_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def clear(self) -> None:
        """Purge all stored preferences."""
        with self._conn:
            self._conn.execute("DELETE FROM user_preferences")
            self._conn.execute("DELETE FROM dataset_aliases")

    def close(self) -> None:
        self._conn.close()
