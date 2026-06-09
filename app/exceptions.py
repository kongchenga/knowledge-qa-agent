from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None, extra: dict[str, Any] | None = None):
        self.detail = detail or self.detail
        self.extra = extra or {}


class NotFoundError(AppError):
    status_code = 404
    detail = "Resource not found"


class BadRequestError(AppError):
    status_code = 400
    detail = "Bad request"


class UnauthorizedError(AppError):
    status_code = 401
    detail = "Unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    detail = "Forbidden"


class RateLimitError(AppError):
    status_code = 429
    detail = "Too many requests"


class LLMError(AppError):
    status_code = 502
    detail = "LLM service error"


class EmbeddingError(AppError):
    status_code = 502
    detail = "Embedding service error"


class DatabaseError(AppError):
    status_code = 500
    detail = "Database error"


class DocumentNotFoundError(NotFoundError):
    detail = "Document not found"


class ConversationNotFoundError(NotFoundError):
    detail = "Conversation not found"


class UnsupportedFileFormatError(BadRequestError):
    detail = "Unsupported file format"
