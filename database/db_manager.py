"""Thread-safe and async-safe SQLite database manager with WAL mode."""

import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from config.settings import DB_PATH, SCHEMA_PATH


class DatabaseManager:
    """Thread-safe SQLite database manager.
    
    Uses a threading lock to serialize write access and configures
    WAL journal mode for concurrent read/write without blocking.
    """

    _instance: Optional['DatabaseManager'] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> 'DatabaseManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        self.db_path = db_path or DB_PATH
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._initialized = True

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection with WAL pragmas."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        return self._local.connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._get_connection()

    def initialize_schema(self, schema_path: Optional[Path] = None):
        """Create all tables and indices from schema.sql."""
        path = schema_path or SCHEMA_PATH
        with open(path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        conn = self._get_connection()
        with self._write_lock:
            conn.executescript(schema_sql)
            conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement (auto-commits for writes)."""
        conn = self._get_connection()
        with self._write_lock:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self._get_connection()
        with self._write_lock:
            conn.executemany(sql, params_list)
            conn.commit()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a query and return all rows."""
        conn = self._get_connection()
        return conn.execute(sql, params).fetchall()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute a query and return one row."""
        conn = self._get_connection()
        return conn.execute(sql, params).fetchone()

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
