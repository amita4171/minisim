"""
Dependency injection for the MiniSim API.

Provides shared resources (Database connections, rate limiter) as
FastAPI dependencies.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Generator

from src.db.database import Database

logger = logging.getLogger(__name__)

# ── Request log database ──

_REQUEST_LOG_DB_PATH = os.environ.get("MINISIM_REQUEST_LOG_DB", "results/request_log.db")


class RequestLogDB:
    """
    Lightweight SQLite wrapper for API request logging.
    Follows the same pattern as src.db.database.Database.
    """

    def __init__(self, path: str = _REQUEST_LOG_DB_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                question TEXT,
                mode TEXT,
                latency_ms REAL,
                status TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_request_log_timestamp ON request_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_request_log_key ON request_log(api_key_hash);
        """)
        self.conn.commit()

    def log_request(
        self,
        api_key: str,
        question: str | None = None,
        mode: str | None = None,
        latency_ms: float | None = None,
        status: str = "ok",
    ) -> None:
        """Log an API request. API key is hashed for privacy."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        truncated_question = question[:100] if question else None
        try:
            self.conn.execute(
                """
                INSERT INTO request_log (timestamp, api_key_hash, question, mode, latency_ms, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat() + "Z",
                    key_hash,
                    truncated_question,
                    mode,
                    latency_ms,
                    status,
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

    def close(self):
        self.conn.close()


# ── Singleton instances ──

_request_log_db: RequestLogDB | None = None


def get_request_log_db() -> RequestLogDB:
    """Get or create the singleton RequestLogDB instance."""
    global _request_log_db
    if _request_log_db is None:
        _request_log_db = RequestLogDB()
    return _request_log_db


def get_database() -> Database:
    """Create a new Database connection (caller must close)."""
    return Database()
