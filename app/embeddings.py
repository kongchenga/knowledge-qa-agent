from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from numpy.linalg import norm

from app.config import settings
from app.exceptions import EmbeddingError
from app.monitoring import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self._model = None
        self._dim = settings.embedding_dim
        self._load_timeout = 120  # seconds
        self._load_started: float = 0

    def _load_model(self):
        if self._model is not None:
            return
        self._load_started = time.monotonic()
        logger.info("Loading embedding model: {} (timeout={}s)", settings.embedding_model, self._load_timeout)
        from sentence_transformers import SentenceTransformer
        try:
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
                token=settings.huggingface_token or None,
            )
        except Exception as e:
            elapsed = time.monotonic() - self._load_started
            logger.error("Failed to load embedding model after {:.1f}s: {}", elapsed, e)
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e
        self._dim = self._model.get_sentence_embedding_dimension()
        elapsed = time.monotonic() - self._load_started
        logger.info("Embedding model loaded in {:.1f}s, dim={}", elapsed, self._dim)

    def embed_query(self, text: str) -> list[float]:
        self._load_model()
        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embs]

    def embed_single(self, text: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(text, normalize_embeddings=True)

    @property
    def dim(self) -> int:
        if self._model is not None:
            return self._dim
        return settings.embedding_dim


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
