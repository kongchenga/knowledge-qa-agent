from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embeddings import get_embedding_service
from app.monitoring import get_logger

logger = get_logger(__name__)


class BM25Index:
    def __init__(self):
        self.b = 0.75
        self.k1 = 1.5
        self.docs: list[dict] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0
        self.total_docs: int = 0
        self._persist_dir = settings.resolved_chroma_dir / "bm25"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        self._load()

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        cjk = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", text)
        result = [t for t in tokens if len(t) > 1]
        for c in cjk:
            result.append(c)
            if len(c) > 1:
                for char in c:
                    result.append(char)
        return result

    def _init_sqlite(self):
        import sqlite3
        self._conn = sqlite3.connect(str(self._persist_dir / "bm25.db"), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE,
            content TEXT,
            metadata_json TEXT,
            token_count INTEGER
        )""")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS posting (
            token TEXT,
            doc_idx INTEGER,
            position INTEGER,
            PRIMARY KEY (token, doc_idx, position)
        )""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_posting_token ON posting(token)")
        self._conn.commit()
        self._stmt_insert_doc = None
        self._stmt_insert_post = None

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        import json as _json
        for text, meta, tid in zip(texts, metadatas, ids):
            tokens = self._tokenize(text)
            entry = {"id": tid, "tokens": tokens, "metadata": meta, "content": text}
            idx = len(self.docs)
            self.docs.append(entry)
            self.doc_lengths.append(len(tokens))
            # Incremental SQLite insert
            self._conn.execute(
                "INSERT OR REPLACE INTO docs (id, chunk_id, content, metadata_json, token_count) VALUES (?, ?, ?, ?, ?)",
                (idx, tid, text, _json.dumps(meta, ensure_ascii=False), len(tokens)),
            )
            for pos, token in enumerate(tokens):
                self._conn.execute(
                    "INSERT OR REPLACE INTO posting (token, doc_idx, position) VALUES (?, ?, ?)",
                    (token, idx, pos),
                )
            # Update in-memory inverted index
            for pos, token in enumerate(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((idx, pos))
        self._conn.commit()

        self.total_docs = len(self.docs)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = self._tokenize(query)
        if not self.docs or not query_tokens:
            return []

        scores = [0.0] * self.total_docs
        query_tf = Counter(query_tokens)

        for token, q_tf in query_tf.items():
            if token not in self.inverted_index:
                continue
            posting = self.inverted_index[token]
            df = len(posting)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, _ in posting:
                doc_len = self.doc_lengths[doc_idx]
                tf = sum(1 for t in self.docs[doc_idx]["tokens"] if t == token)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * numerator / denominator

        scored = [(scores[i], self.docs[i]) for i in range(self.total_docs) if scores[i] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        output = []
        for score, doc in scored[:top_k]:
            output.append({
                "chunk_id": doc["id"],
                "content": doc["content"],
                "title": doc["metadata"].get("title", ""),
                "doc_id": doc["metadata"].get("doc_id", ""),
                "score": round(float(score), 4),
                "source": "bm25",
            })
        return output

    def delete_by_doc_id(self, doc_id: int):
        str_id = str(doc_id)
        indices_to_remove = {
            i for i, d in enumerate(self.docs)
            if d["metadata"].get("doc_id") == str_id
        }
        if not indices_to_remove:
            return

        # Remove from SQLite
        self._conn.execute(f"DELETE FROM docs WHERE id IN ({','.join('?' for _ in indices_to_remove)})", tuple(sorted(indices_to_remove)))
        self._conn.execute("DELETE FROM posting WHERE doc_idx IN (" + ",".join("?" for _ in indices_to_remove) + ")", tuple(sorted(indices_to_remove)))
        self._conn.commit()

        self.docs = [d for i, d in enumerate(self.docs) if i not in indices_to_remove]
        self.doc_lengths = [l for i, l in enumerate(self.doc_lengths) if i not in indices_to_remove]

        new_inverted = {}
        for token, postings in self.inverted_index.items():
            old_to_new = {}
            new_idx = 0
            new_postings = []
            for old_idx, pos in postings:
                if old_idx in indices_to_remove:
                    continue
                if old_idx not in old_to_new:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
                new_postings.append((old_to_new[old_idx], pos))
            if new_postings:
                new_inverted[token] = new_postings

        self.inverted_index = new_inverted
        self.total_docs = len(self.docs)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

    def count(self) -> int:
        return len(self.docs)

    def _load(self):
        """Load from SQLite (incremental via posting table), fall back to JSON for migration."""
        import json as _json
        idx_file = self._persist_dir / "bm25.json"
        sqlite_loaded = False

        # Prefer SQLite
        try:
            rows = self._conn.execute("SELECT id, chunk_id, content, metadata_json, token_count FROM docs ORDER BY id").fetchall()
            if rows:
                self.docs = []
                self.doc_lengths = []
                doc_idx_map = {}  # sqlite id -> in-memory index
                for mem_idx, row in enumerate(rows):
                    sql_id, tid, content, meta_str, tok_count = row[0], row[1], row[2], row[3], row[4]
                    meta = _json.loads(meta_str) if meta_str else {}
                    tokens = self._tokenize(content)
                    self.docs.append({"id": tid, "tokens": tokens, "metadata": meta, "content": content})
                    self.doc_lengths.append(tok_count)
                    doc_idx_map[sql_id] = mem_idx

                # Bulk-rebuild inverted index from posting table
                post_rows = self._conn.execute("SELECT token, doc_idx, position FROM posting ORDER BY token, doc_idx").fetchall()
                for token, doc_idx, pos in post_rows:
                    if doc_idx in doc_idx_map:
                        mem_idx = doc_idx_map[doc_idx]
                        if token not in self.inverted_index:
                            self.inverted_index[token] = []
                        self.inverted_index[token].append((mem_idx, pos))

                self.total_docs = len(self.docs)
                self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)
                sqlite_loaded = True
                logger.info("BM25 index loaded from SQLite: {} docs, {} tokens", self.total_docs, len(post_rows))
                if idx_file.exists():
                    idx_file.unlink()  # Migrated
        except Exception as e:
            logger.warning("BM25 SQLite load failed, trying JSON: {}", e)

        # Fallback: JSON (old format migration)
        if not sqlite_loaded and idx_file.exists():
            try:
                data = _json.loads(idx_file.read_text("utf-8"))
                self.b = data["b"]
                self.k1 = data["k1"]
                self.avg_doc_length = data["avg_doc_length"]
                self.total_docs = data["total_docs"]
                self.doc_lengths = data["doc_lengths"]
                self.docs = data["docs"]
                self.inverted_index = {k: list(t) for k, t in data["inverted_index"].items()}
                logger.info("BM25 index loaded from JSON (legacy): {} docs", self.total_docs)
                # Migrate to SQLite
                for i, d in enumerate(self.docs):
                    self._conn.execute(
                        "INSERT OR REPLACE INTO docs (id, chunk_id, content, metadata_json, token_count) VALUES (?, ?, ?, ?, ?)",
                        (i, d["id"], d["content"], _json.dumps(d.get("metadata", {}), ensure_ascii=False), len(d.get("tokens", []))),
                    )
                self._conn.commit()
                idx_file.unlink()
                logger.info("BM25 migrated from JSON to SQLite")
            except Exception as e:
                logger.warning("Failed to load BM25 index: {}", e)

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


class VectorStore:
    def __init__(self):
        self.embedding = get_embedding_service()
        self.bm25 = BM25Index()

        persist_dir = str(settings.resolved_chroma_dir)
        self._chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB ready ({} docs)", self._collection.count())

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None):
        if not texts:
            return
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = self.embedding.embed_documents(texts)

        chroma_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if v is not None:
                    clean[k] = str(v)
            chroma_metadatas.append(clean)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=chroma_metadatas,
            documents=texts,
        )

        self.bm25.add_texts(texts, metadatas, ids)

        logger.debug("Added {} texts to vector store", len(texts))

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        query_emb = self.embedding.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )
        output = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] or {}
                output.append({
                    "chunk_id": doc_id,
                    "content": results["documents"][0][i],
                    "title": meta.get("title", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "score": round(1.0 - (results["distances"][0][i] if results.get("distances") else 0), 4),
                    "source": "semantic",
                })
        return output

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
        semantic_results = self.similarity_search(query, top_k=top_k * 2)
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        seen = {}
        for rank, item in enumerate(semantic_results):
            cid = item["chunk_id"]
            score = item["score"] * alpha
            seen[cid] = {**item, "rrf_score": score, "sources": ["semantic"]}

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            bm25_score = item["score"] * (1 - alpha)
            if cid in seen:
                seen[cid]["rrf_score"] += bm25_score
                seen[cid]["sources"].append("bm25")
                seen[cid]["score"] = max(seen[cid]["score"], item["score"])
            else:
                seen[cid] = {**item, "rrf_score": bm25_score, "sources": ["bm25"]}

        merged = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)
        return merged[:top_k]

    def delete_by_doc_id(self, doc_id: int):
        str_id = str(doc_id)

        results = self._collection.get(where={"doc_id": str_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

        self.bm25.delete_by_doc_id(doc_id)

    def count(self) -> int:
        return self._collection.count()

    def close(self):
        try:
            if self._chroma_client:
                # ChromaDB PersistentClient doesn't have explicit close(),
                # but we should release the reference
                del self._chroma_client
                self._chroma_client = None
                self._collection = None
        except Exception:
            pass
        try:
            self.bm25.close()
        except Exception:
            pass
        logger.debug("Vector store closed")


def create_vector_store() -> VectorStore:
    return VectorStore()
