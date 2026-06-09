from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.agent.qa_agent import QAAgent
from app.config import settings
from app.exceptions import (
    BadRequestError,
    DocumentNotFoundError,
    ConversationNotFoundError,
    UnsupportedFileFormatError,
)
from app.monitoring import get_logger
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentResponse,
    ConversationResponse,
    MessageResponse,
    HealthResponse,
)
from app.utils import extract_text_from_bytes, parse_document

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".py", ".json", ".yaml", ".yml", ".csv", ".xml", ".html", ".css", ".js",
}


def create_router() -> APIRouter:
    router = APIRouter()
    agent = QAAgent()

    # ─── Health ───────────────────────────────────────────────────────────

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        checks = {"llm": False, "vector": False, "sql": False, "doc_count": 0}

        try:
            docs = agent.sql_store.list_documents()
            checks["doc_count"] = len(docs)
            checks["sql"] = True
        except Exception as e:
            logger.warning("SQLite health check failed: {}", e)

        try:
            checks["vector"] = agent.vector_store.count() >= 0
        except Exception as e:
            logger.warning("Vector store health check failed: {}", e)

        try:
            checks["llm"] = agent.llm_client.is_ready()
        except Exception as e:
            logger.warning("LLM health check failed: {}", e)

        all_healthy = all(checks[k] for k in ("sql", "vector"))
        status = "ok" if all_healthy else ("degraded" if checks["sql"] else "unhealthy")

        return HealthResponse(
            status=status,
            llm_configured=checks["llm"],
            vector_store_ready=checks["vector"],
            total_documents=checks["doc_count"],
        )

    @router.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe — all backends must be available."""
        try:
            agent.sql_store.list_documents()
            agent.vector_store.count()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Not ready: {e}")
        return {"status": "ready"}

    @router.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe — process is alive."""
        return {"status": "alive"}

    # ─── Query ────────────────────────────────────────────────────────────

    @router.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        if not req.question.strip():
            raise BadRequestError("问题不能为空")

        result = await agent.query(
            question=req.question,
            top_k=req.top_k,
            conversation_id=req.session_id,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            rewritten_query=result.get("rewritten_query"),
        )

    @router.post("/query/stream")
    async def query_stream(req: QueryRequest):
        if not req.question.strip():
            raise BadRequestError("问题不能为空")

        return EventSourceResponse(
            agent.stream_query(
                question=req.question,
                top_k=req.top_k,
                conversation_id=req.session_id,
            ),
        )

    # ─── Documents ────────────────────────────────────────────────────────

    @router.post("/documents/upload")
    async def upload_document(
        title: str = Form(...),
        tags: str = Form(""),
        category: str = Form(""),
        file: UploadFile = File(...),
    ):
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileFormatError(f"不支持的文件格式: {ext}")

        content_bytes = await file.read()
        if not content_bytes:
            raise BadRequestError("文件内容为空")

        content = extract_text_from_bytes(content_bytes, file.filename)
        if not content or not content.strip():
            raise BadRequestError("无法提取文件内容或文件为空")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        parse_result = parse_document(content_bytes, file.filename)

        result = agent.add_document(title, content, tag_list, category or None)

        return {
            "message": "文档上传成功",
            "doc_id": result["doc_id"],
            "chunk_count": result["chunk_count"],
            "tables": parse_result.tables if parse_result else 0,
            "images": len(parse_result.images) if parse_result else 0,
        }

    @router.post("/documents/text")
    async def add_text_document(
        title: str = Form(...),
        content: str = Form(...),
        tags: str = Form(""),
        category: str = Form(""),
    ):
        if not content.strip():
            raise BadRequestError("内容不能为空")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        result = agent.add_document(title, content, tag_list, category or None)
        return {
            "message": "文档添加成功",
            "doc_id": result["doc_id"],
            "chunk_count": result["chunk_count"],
        }

    @router.put("/documents/{doc_id}")
    async def update_document(
        doc_id: int,
        title: str = Form(...),
        content: str = Form(...),
        tags: str = Form(""),
    ):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        try:
            result = agent.update_document(doc_id, title, content, tag_list)
            return {"message": "文档更新成功", "doc_id": result["doc_id"], "chunk_count": result["chunk_count"]}
        except DocumentNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/documents", response_model=list[DocumentResponse])
    async def list_documents(
        category: str = Query(default="", description="按分类筛选"),
        offset: int = Query(default=0, ge=0, description="分页偏移"),
        limit: int = Query(default=200, ge=1, le=1000, description="分页大小"),
    ):
        cat = category.strip() or None
        all_docs = agent.list_documents(category=cat)
        return all_docs[offset:offset + limit]

    @router.delete("/documents/{doc_id}")
    async def delete_document(doc_id: int):
        ok = agent.delete_document(doc_id)
        if not ok:
            raise DocumentNotFoundError("文档不存在")
        return {"message": "文档已删除"}

    @router.get("/documents/{doc_id}/markdown")
    async def get_document_markdown(doc_id: int):
        md = agent.get_document_markdown(doc_id)
        if md is None:
            raise DocumentNotFoundError("文档不存在")
        return {"markdown": md}

    @router.get("/stats")
    async def stats():
        return agent.get_stats()

    # ─── Conversations ────────────────────────────────────────────────────

    @router.get("/conversations", response_model=list[ConversationResponse])
    async def list_conversations():
        return agent.conversation.list_conversations()

    @router.post("/conversations")
    async def create_conversation(title: str = Form("新对话")):
        conv = agent.conversation.create_conversation(title)
        return conv

    @router.delete("/conversations/{conv_id}")
    async def delete_conversation(conv_id: int):
        conv = agent.conversation.get_conversation(conv_id)
        if conv is None:
            raise ConversationNotFoundError("对话不存在")
        agent.conversation.delete_conversation(conv_id)
        return {"message": "对话已删除"}

    @router.get("/conversations/{conv_id}/messages", response_model=list[MessageResponse])
    async def get_messages(conv_id: int):
        conv = agent.conversation.get_conversation(conv_id)
        if conv is None:
            raise ConversationNotFoundError("对话不存在")
        return agent.conversation.get_messages(conv_id)

    return router
