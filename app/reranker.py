from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Optional

from app.config import settings
from app.monitoring import get_logger

logger = get_logger(__name__)


class RerankerService:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading reranker model: {}", settings.reranker_model)
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            settings.reranker_model,
            device=settings.reranker_device,
        )
        logger.info("Reranker model loaded")

    def _rerank_sync(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[dict]:
        self._load_model()
        pairs = [(query, d["content"]) for d in documents]
        scores = self._model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(float(score), 4)

        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        if not documents:
            return []
        top_k = top_k or settings.reranker_top_k
        return self._rerank_sync(query, documents, top_k)

    async def arerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Async-safe rerank — runs CrossEncoder in thread pool to avoid blocking event loop."""
        if not documents:
            return []
        top_k = top_k or settings.reranker_top_k
        return await asyncio.to_thread(self._rerank_sync, query, documents, top_k)


@lru_cache
def get_reranker() -> RerankerService:
    return RerankerService()
