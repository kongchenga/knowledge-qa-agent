from __future__ import annotations

import json
import pickle
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from app.config import settings
from app.monitoring import get_logger

logger = get_logger(__name__)

LSI_N_COMPONENTS = 128


def _cjk_tokenizer(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    for t in re.findall(r"[a-zA-Z0-9]+", text):
        if len(t) >= 2:
            tokens.append(t)
    for c in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", text):
        tokens.append(c)
    return tokens


class SemanticIndex:
    def __init__(self, n_components: int = LSI_N_COMPONENTS):
        self.n_components = n_components
        self._docs: list[dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._normalizer: Optional[Normalizer] = None
        self._doc_vectors: Optional[np.ndarray] = None
        self._fitted = False
        self._pending_adds = False
        self._rebuild_batch_threshold = 20  # rebuild after N additions
        self._addition_count = 0
        self._persist_dir = settings.resolved_chroma_dir / "lsi"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def add_texts(self, texts, metadatas=None, ids=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        for text, meta, tid in zip(texts, metadatas, ids):
            self._docs.append({
                "id": tid,
                "content": text,
                "metadata": meta,
            })

        # Defer rebuild to next search or batch threshold
        self._fitted = False
        self._pending_adds = True
        self._save_docs()

    def _rebuild(self):
        if not self._docs:
            self._vectorizer = None
            self._svd = None
            self._normalizer = None
            self._doc_vectors = None
            self._fitted = False
            return

        texts = [d["content"] for d in self._docs]

        vectorizer = TfidfVectorizer(
            tokenizer=_cjk_tokenizer,
            lowercase=False,
            max_features=5000,
            sublinear_tf=True,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        n_features = tfidf_matrix.shape[1]

        if n_features >= 2 and len(self._docs) >= 3 and self.n_components >= 2:
            n_svd = min(self.n_components, n_features - 1, len(self._docs) - 1, 256)
            svd = TruncatedSVD(n_components=n_svd, random_state=42)
            normalizer = Normalizer(copy=False)
            doc_vectors = normalizer.fit_transform(svd.fit_transform(tfidf_matrix))
            self._svd = svd
            self._normalizer = normalizer
        else:
            doc_vectors = tfidf_matrix.toarray()
            self._svd = None
            self._normalizer = None

        self._vectorizer = vectorizer
        self._doc_vectors = doc_vectors
        self._fitted = True

    def _transform_query(self, text: str):
        tfidf_vec = self._vectorizer.transform([text])
        if self._svd is not None:
            svd_vec = self._svd.transform(tfidf_vec)
            return self._normalizer.transform(svd_vec)
        return tfidf_vec.toarray()

    def _ensure_fitted(self):
        """Rebuild only when necessary (lazy/deferred)."""
        if not self._fitted:
            self._rebuild()
            self._pending_adds = False
            self._addition_count = 0

    def search(self, query: str, top_k: int = 5):
        self._ensure_fitted()
        if not self._fitted or not self._docs:
            return []

        q_vec = self._transform_query(query)
        similarities = cosine_similarity(q_vec, self._doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            doc = self._docs[idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": round(score, 4),
            })
        return results

    def delete_by_doc_id(self, doc_id: int):
        str_id = str(doc_id)
        old_len = len(self._docs)
        self._docs = [d for d in self._docs if d["metadata"].get("doc_id") != str_id]
        if len(self._docs) < old_len:
            self._rebuild()
            self._pending_adds = False
            self._addition_count = 0
            self._save()

    def count(self) -> int:
        return len(self._docs)

    def clear(self):
        self._docs.clear()
        self._vectorizer = None
        self._svd = None
        self._normalizer = None
        self._doc_vectors = None
        self._fitted = False
        self._save()

    def _save_docs(self):
        """Save only documents (lightweight, called on every add)."""
        (self._persist_dir / "docs.json").write_text(
            json.dumps({"docs": self._docs}, ensure_ascii=False), encoding="utf-8"
        )

    def _save(self):
        """Full save including pipeline models (called after rebuild)."""
        self._save_docs()
        data = {}
        if self._vectorizer:
            data["vectorizer"] = self._vectorizer
        if self._svd:
            data["svd"] = self._svd
        if self._normalizer:
            data["normalizer"] = self._normalizer
        if data:
            with open(self._persist_dir / "pipeline.pkl", "wb") as f:
                pickle.dump(data, f)

    def _load(self):
        docs_file = self._persist_dir / "docs.json"
        pipeline_file = self._persist_dir / "pipeline.pkl"
        if docs_file.exists():
            try:
                self._docs = json.loads(docs_file.read_text(encoding="utf-8")).get("docs", [])
            except Exception:
                self._docs = []
        if pipeline_file.exists() and self._docs:
            try:
                with open(pipeline_file, "rb") as f:
                    data = pickle.load(f)
                self._vectorizer = data.get("vectorizer")
                self._svd = data.get("svd")
                self._normalizer = data.get("normalizer")
                self._rebuild()
            except Exception:
                pass
