from __future__ import annotations

from pathlib import Path

import pytest

from app.database.base import DatabaseManager
from app.database.sql_store import SQLStore


class TestDatabaseManager:
    def test_singleton(self):
        db1 = DatabaseManager.get_instance()
        db2 = DatabaseManager.get_instance()
        assert db1 is db2

    def test_execute(self):
        db = DatabaseManager.get_instance()
        db.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        db.commit()
        db.execute("INSERT INTO test (name) VALUES (?)", ("hello",))
        db.commit()
        row = db.execute("SELECT name FROM test WHERE id=1").fetchone()
        assert row["name"] == "hello"

    def test_transaction_rollback(self):
        db = DatabaseManager.get_instance()
        db.execute("CREATE TABLE IF NOT EXISTS test_txn (id INTEGER PRIMARY KEY, name TEXT)")
        db.commit()
        with pytest.raises(Exception):
            with db.transaction() as conn:
                conn.execute("INSERT INTO test_txn (name) VALUES (?)", ("txn_test",))
                raise ValueError("rollback")
        row = db.execute("SELECT COUNT(*) as cnt FROM test_txn WHERE name='txn_test'").fetchone()
        assert row["cnt"] == 0


class TestSQLStore:
    def test_add_and_list_documents(self):
        store = SQLStore()
        doc_id = store.add_document("测试文档", "test.md", tags=["test"], category="unit")
        assert doc_id > 0

        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["title"] == "测试文档"

        docs = store.list_documents()
        assert any(d["id"] == doc_id for d in docs)

    def test_chunks(self):
        store = SQLStore()
        doc_id = store.add_document("Chunk Test", "chunk.md")
        store.add_chunk(f"{doc_id}_0", doc_id, "chunk content 1", 0)
        store.add_chunk(f"{doc_id}_1", doc_id, "chunk content 2", 1)

        chunks = store.get_chunks_by_doc_id(doc_id)
        assert len(chunks) == 2
        assert chunks[0]["content"] == "chunk content 1"

    def test_search_by_tags(self):
        store = SQLStore()
        store.add_document("Tag Doc 1", "t1.md", tags=["python", "test"])
        store.add_document("Tag Doc 2", "t2.md", tags=["java", "test"])
        store.add_document("Tag Doc 3", "t3.md", tags=["python"])

        results = store.search_by_tags(["python"])
        assert len(results) == 2

        results = store.search_by_tags(["java"])
        assert len(results) == 1

    def test_delete(self):
        store = SQLStore()
        doc_id = store.add_document("To Delete", "del.md", tags=["delete"])
        store.add_chunk(f"{doc_id}_0", doc_id, "content", 0)

        store.delete_document(doc_id)
        doc = store.get_document(doc_id)
        assert doc is None
        chunks = store.get_chunks_by_doc_id(doc_id)
        assert len(chunks) == 0
