from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from app.database.base import DatabaseManager
from app.monitoring import get_logger

logger = get_logger(__name__)


class ConversationManager:
    def __init__(self):
        self._db = DatabaseManager.get_instance()
        self._init_tables()

    def _init_tables(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);
        """)

    def create_conversation(self, title: str = "新对话") -> dict:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._db.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        self._db.commit()
        return {"id": cur.lastrowid, "title": title, "created_at": now}

    def list_conversations(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: int) -> Optional[dict]:
        row = self._db.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_title(self, conv_id: int, title: str):
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title[:100], now, conv_id),
        )
        self._db.commit()

    def delete_conversation(self, conv_id: int):
        self._db.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        self._db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self._db.commit()

    def add_message(self, conv_id: int, role: str, content: str, sources: Optional[list] = None) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        sources_str = json.dumps(sources, ensure_ascii=False) if sources else ""
        cur = self._db.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, role, content, sources_str, now),
        )
        self._db.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now, conv_id),
        )
        self._db.commit()

        if conv_id:
            first_msg = self._db.execute(
                "SELECT content FROM messages WHERE conversation_id=? ORDER BY id ASC LIMIT 1",
                (conv_id,),
            ).fetchone()
            if first_msg:
                new_title = first_msg["content"][:50].strip()
                self._db.execute(
                    "UPDATE conversations SET title=? WHERE id=? AND title='新对话'",
                    (new_title, conv_id),
                )
                self._db.commit()

        return {"id": cur.lastrowid, "role": role, "content": content}

    def get_messages(self, conv_id: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        result = []
        for r in rows:
            msg = dict(r)
            if msg.get("sources"):
                try:
                    msg["sources"] = json.loads(msg["sources"])
                except Exception:
                    msg["sources"] = []
            result.append(msg)
        return result

    def close(self):
        self._db.close_all()
