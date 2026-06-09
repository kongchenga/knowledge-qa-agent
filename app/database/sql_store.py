from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.database.base import DatabaseManager
from app.monitoring import get_logger

logger = get_logger(__name__)


class SQLStore:
    def __init__(self):
        self._db = DatabaseManager.get_instance()
        self._init_tables()

    def _init_tables(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                tags TEXT DEFAULT '',
                category TEXT DEFAULT '',
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
            CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
        """)
        logger.debug("SQL tables initialized")

    def add_document(
        self,
        title: str,
        filename: str,
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        tags_str = ",".join(tags) if tags else ""
        cur = self._db.execute(
            "INSERT INTO documents (title, tags, category, filename, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (title, tags_str, category or "", filename, now, now),
        )
        self._db.commit()
        return cur.lastrowid

    def add_chunk(self, chunk_id: str, doc_id: int, content: str, chunk_index: int):
        self._db.execute(
            "INSERT OR IGNORE INTO chunks (id, doc_id, content, chunk_index) VALUES (?, ?, ?, ?)",
            (chunk_id, doc_id, content, chunk_index),
        )
        self._db.commit()

    def get_document(self, doc_id: int) -> Optional[dict]:
        row = self._db.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self, category: Optional[str] = None) -> list[dict]:
        if category:
            rows = self._db.execute(
                "SELECT * FROM documents WHERE category LIKE ? ORDER BY created_at DESC",
                (f"%{category}%",),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int):
        self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._db.commit()

    def get_chunks_by_doc_id(self, doc_id: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def search_by_tags(self, tags: list[str], top_k: int = 5) -> list[dict]:
        rows = []
        for tag in tags:
            matched = self._db.execute(
                "SELECT * FROM documents WHERE tags LIKE ?", (f"%{tag}%",)
            ).fetchall()
            rows.extend(matched)
        seen = set()
        unique = []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(dict(r))
        return unique[:top_k]
