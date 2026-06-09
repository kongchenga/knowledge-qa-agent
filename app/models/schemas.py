from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=10, ge=1, le=50, description="检索文档数量")
    session_id: Optional[int] = Field(default=None, description="会话ID")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    rewritten_query: Optional[str] = None


class SourceInfo(BaseModel):
    title: str
    content_preview: str
    source: str
    score: float
    doc_id: int


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tags: Optional[str] = None
    category: Optional[str] = None


class DocumentResponse(BaseModel):
    id: int
    title: str
    category: str = ""
    tags: list[str] = []
    created_at: str
    chunk_count: int = 0


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: list = []
    created_at: str


class StatsResponse(BaseModel):
    total_documents: int
    vector_chunks: int
    semantic_chunks: int
    lsi_enabled: bool
    llm_rerank_enabled: bool
    cache_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_max: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    llm_configured: bool
    vector_store_ready: bool
    total_documents: int
