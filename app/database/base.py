from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from app.config import settings
from app.monitoring import get_logger

logger = get_logger("database")


class DatabaseManager:
    _instance: Optional[DatabaseManager] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._local = threading.local()
        self._init_database()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> DatabaseManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path or settings.resolved_sqlite_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close_all()
                cls._instance = None

    def _init_database(self):
        db_path = self._db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_raw_connection()
        try:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA busy_timeout=5000;
                PRAGMA foreign_keys=ON;
                PRAGMA cache_size=-8000;
                PRAGMA temp_store=MEMORY;
                PRAGMA mmap_size=268435456;
            """)
        finally:
            conn.close()
        logger.info("Database initialized (WAL mode): {}", db_path)

    def _get_raw_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._get_raw_connection()
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close_all(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.get_connection().execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.get_connection().executemany(sql, params_seq)

    def executescript(self, script: str):
        self.get_connection().executescript(script)

    def commit(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.commit()
