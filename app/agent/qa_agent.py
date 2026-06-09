from __future__ import annotations

import re
import time
from typing import AsyncGenerator, Optional

from app.agent.conversation import ConversationManager
from app.agent.llm_client import LLMClient, LLMError
from app.agent.query_processor import QueryProcessor
from app.agent.tools import execute_tool
from app.config import settings
from app.database.markdown_store import MarkdownStore
from app.database.sql_store import SQLStore
from app.database.vector_store import VectorStore, create_vector_store
from app.exceptions import DocumentNotFoundError
from app.monitoring import get_logger, trace, observe_retrieval, set_document_metrics
from app.reranker import get_reranker

logger = get_logger(__name__)

# ── LangChain-free text splitter ────────────────────────────────────────────────

class _SimpleTextSplitter:
    """Drop-in replacement for langchain RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        return self._split_recursive(text, self._separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        chunks = []
        for sep in separators:
            if sep == "":
                chunks = self._char_split(text)
            else:
                parts = text.split(sep)
                new_chunks = []
                for part in parts:
                    if len(part) <= self.chunk_size:
                        if part.strip():
                            new_chunks.append(part.strip())
                    else:
                        new_chunks.extend(self._split_recursive(part, separators[1:]))
                chunks = new_chunks
            if chunks:
                break
        return self._merge_with_overlap(chunks)

    def _char_split(self, text: str) -> list[str]:
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap) if text[i:i + self.chunk_size].strip()]

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks:
            return []
        result = []
        current = chunks[0]
        for chunk in chunks[1:]:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current += "\n" + chunk
            else:
                result.append(current)
                current = chunk
        result.append(current)
        return result

SYSTEM_PROMPT = """你是一个专业的知识问答助手。请严格遵循以下规则：
1. 仅基于提供的知识内容回答问题
2. 如果不确定或找不到答案，请明确说"不知道"，不要编造
3. 回答时引用相关来源，使用 [来源: 标题名] 格式
4. 使用中文回答
5. 保持回答简洁、准确、有条理

当知识库信息不足时，你可以使用提供的函数（functions/tools）来获取外部信息。
常用场景：
- 问当前时间、日期 -> 调用 current_time 函数
- 需要联网搜索 -> 调用 web_search 函数
- 数学计算 -> 调用 calculator 函数"""

FALLBACK_ANSWER = "抱歉，我现在无法处理您的请求。请稍后再试。"


class QAAgent:
    def __init__(self):
        self.vector_store = create_vector_store()
        self.sql_store = SQLStore()
        self.markdown_store = MarkdownStore()
        self.llm_client = LLMClient()
        self.query_processor = QueryProcessor(self.llm_client)
        self.reranker = get_reranker()
        self.conversation = ConversationManager()

        self.text_splitter = _SimpleTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl = settings.cache_ttl_seconds
        self._cache_max = settings.cache_max_size
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def _cache_get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            ts, value = self._cache[key]
            if time.monotonic() - ts < self._cache_ttl:
                self._cache_hits += 1
                return value
            del self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_set(self, key: str, value: dict):
        if len(self._cache) >= self._cache_max:
            oldest = min(self._cache.items(), key=lambda x: x[1][0])
            del self._cache[oldest[0]]
        self._cache[key] = (time.monotonic(), value)

    async def _hybrid_retrieve(self, question: str, top_k: int = 10) -> list[dict]:
        t0 = time.monotonic()

        results = self.vector_store.hybrid_search(
            question,
            top_k=top_k,
            alpha=0.5,
        )

        try:
            tag_keywords = self.query_processor.extract_tags(question)
            if tag_keywords:
                sql_docs = self.sql_store.search_by_tags(tag_keywords, top_k=top_k)
                sql_chunks = []
                for doc in sql_docs:
                    chunks = self.sql_store.get_chunks_by_doc_id(doc["id"])
                    for c in chunks:
                        sql_chunks.append({
                            "chunk_id": c["id"],
                            "content": c["content"],
                            "title": doc["title"],
                            "doc_id": str(doc["id"]),
                            "score": 1.0,
                            "source": "sql_tags",
                            "sources": ["sql_tags"],
                            "rrf_score": 0.5,
                        })
                existing_ids = {r["chunk_id"] for r in results}
                for sc in sql_chunks:
                    if sc["chunk_id"] not in existing_ids:
                        results.append(sc)
        except Exception as e:
            logger.warning("Tag search failed: {}", e)

        try:
            if len(results) > 1:
                results = await self.reranker.arerank(question, results, top_k=top_k)
        except Exception as e:
            logger.warning("Reranker failed, using raw results: {}", e)

        duration = time.monotonic() - t0
        observe_retrieval("hybrid", duration)

        return results[:top_k]

    def add_document(
        self,
        title: str,
        content: str,
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> dict:
        filename = self.markdown_store.save(title, content, tags)
        doc_id = self.sql_store.add_document(title, filename, tags, category)

        chunks = self.text_splitter.split_text(content)
        chunk_ids = []
        chunk_metadatas = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            chunk_ids.append(chunk_id)
            chunk_metadatas.append({
                "doc_id": str(doc_id),
                "title": title,
                "chunk_index": str(i),
                "tags": ",".join(tags) if tags else "",
                "category": category or "",
            })
            self.sql_store.add_chunk(chunk_id, doc_id, chunk_text, i)

        if chunks:
            try:
                self.vector_store.add_texts(chunks, chunk_metadatas, chunk_ids)
            except Exception as e:
                logger.error("Vector store add failed: {}", e)

        trace("document_added", doc_id=doc_id, chunks=len(chunks), title=title)
        self._update_metrics()
        return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}

    def update_document(
        self,
        doc_id: int,
        title: str,
        content: str,
        tags: Optional[list[str]] = None,
    ) -> dict:
        old_doc = self.sql_store.get_document(doc_id)
        if old_doc is None:
            raise DocumentNotFoundError(f"文档 {doc_id} 不存在")

        self.markdown_store.delete(old_doc["filename"])
        filename = self.markdown_store.save(title, content, tags)

        from app.database.base import DatabaseManager
        db = DatabaseManager.get_instance()
        with db.transaction() as conn:
            conn.execute(
                "UPDATE documents SET title=?, tags=?, filename=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (title, ",".join(tags) if tags else "", filename, doc_id),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

        try:
            self.vector_store.delete_by_doc_id(doc_id)
        except Exception as e:
            logger.warning("Vector store delete failed: {}", e)

        chunks = self.text_splitter.split_text(content)
        chunk_ids = []
        chunk_metadatas = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            chunk_ids.append(chunk_id)
            chunk_metadatas.append({
                "doc_id": str(doc_id),
                "title": title,
                "chunk_index": str(i),
                "tags": ",".join(tags) if tags else "",
            })
            self.sql_store.add_chunk(chunk_id, doc_id, chunk_text, i)

        if chunks:
            try:
                self.vector_store.add_texts(chunks, chunk_metadatas, chunk_ids)
            except Exception as e:
                logger.error("Vector store add failed during update: {}", e)

        self._update_metrics()
        return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}

    async def query(
        self,
        question: str,
        top_k: int = 10,
        conversation_id: Optional[int] = None,
    ) -> dict:
        trace("query_started", question=question)

        history = None
        if conversation_id:
            history = self.conversation.get_messages(conversation_id)

        try:
            rewritten = await self.query_processor.rewrite_query(question, history)
        except Exception as e:
            logger.warning("Query rewrite failed: {}", e)
            rewritten = question

        # Check cache first
        cache_key = None
        if settings.cache_enabled:
            cache_key = f"{rewritten}:{top_k}"
            cached = self._cache_get(cache_key)
            if cached and isinstance(cached, dict):
                trace("cache_hit", question=question)
                if conversation_id:
                    self.conversation.add_message(conversation_id, "user", question)
                    self.conversation.add_message(conversation_id, "assistant", cached.get("answer", ""), cached.get("sources", []))
                return cached

        try:
            retrieved = await self._hybrid_retrieve(rewritten, top_k=top_k)
        except Exception as e:
            logger.error("Retrieval failed: {}", e)
            retrieved = []

        user_prompt, sources = self._build_query_context(question, retrieved)

        try:
            answer = await self._llm_invoke_with_tools(SYSTEM_PROMPT, user_prompt, question)
        except LLMError as e:
            logger.error("LLM query failed: {}", e)
            answer = FALLBACK_ANSWER
        except Exception as e:
            logger.error("Unexpected error during LLM call: {}", e)
            answer = FALLBACK_ANSWER

        if conversation_id:
            self.conversation.add_message(conversation_id, "user", question)
            self.conversation.add_message(conversation_id, "assistant", answer, sources)

        # Cache the result
        if settings.cache_enabled and cache_key:
            self._cache_set(cache_key, {
                "answer": answer,
                "sources": sources,
                "rewritten_query": rewritten if rewritten != question else None,
            })

        trace("query_completed", question=question, sources=len(sources))
        return {
            "answer": answer,
            "sources": sources,
            "rewritten_query": rewritten if rewritten != question else None,
        }

    async def _llm_invoke_with_tools(self, system_prompt: str, user_prompt: str, original_question: str) -> str:
        """Invoke LLM with native function-calling support (OpenAI-compatible tools API).
        Falls back to text-parsing if the model doesn't support native tool calls."""
        import json as _json

        # ── Strategy 1: Native function calling (OpenAI tools API) ─────────
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网获取最新信息",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "current_time",
                    "description": "获取当前日期和时间",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        try:
            resp = await self.llm_client.achat_with_tools(
                system_prompt, user_prompt, tools=openai_tools
            )
        except Exception as e:
            logger.warning("Native tool calling failed, falling back to parsed mode: {}", e)
            # Fallback to text parsing
            answer = await self.llm_client.achat(system_prompt, user_prompt)
            return await self._parse_tool_fallback(answer, system_prompt, user_prompt)

        if resp["type"] == "tool_call":
            tool_name = resp["name"]
            tool_args = resp["arguments"]
            if isinstance(tool_args, str):
                try:
                    tool_args = _json.loads(tool_args)
                except Exception:
                    tool_args = {}
            tool_result = execute_tool(tool_name, tool_args)
            logger.info("Tool {} called (native) args={} -> result_len={}", tool_name, tool_args, len(tool_result))

            # Send tool result back for final answer
            final_answer = await self.llm_client.achat(
                system_prompt,
                f"{user_prompt}\n\n[工具 {tool_name} 执行结果]:\n{tool_result}\n\n请基于以上结果给出最终回答。"
            )
            return final_answer
        else:
            return resp["content"]

    async def _parse_tool_fallback(self, answer: str, system_prompt: str, user_prompt: str) -> str:
        """Fallback: parse JSON tool calls from LLM text output."""
        max_rounds = 2
        for _ in range(max_rounds):
            tool_call = self._parse_tool_call(answer)
            if not tool_call:
                return answer

            tool_result = execute_tool(tool_call["tool"], tool_call.get("arguments", {}))
            logger.info("Tool {} called (parsed) args={} -> result_len={}", tool_call["tool"], tool_call.get("arguments", {}), len(tool_result))

            follow_up = f"{user_prompt}\n\n[工具调用结果 {tool_call['tool']}]:\n{tool_result}\n\n请基于以上工具返回结果，给出最终回答。"
            answer = await self.llm_client.achat(system_prompt, follow_up)

        return answer

    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """Extract {{'tool': ..., 'arguments': ...}} JSON from LLM response."""
        import json as _json
        # Try full match
        m = re.search(r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]+\}\s*\}', text)
        if m:
            try:
                return _json.loads(m.group())
            except Exception:
                pass
        # Try to find any JSON object with "tool" key
        for m in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', text):
            try:
                obj = _json.loads(m.group())
                if "tool" in obj:
                    return obj
            except Exception:
                continue
        return None

    def _build_query_context(self, question: str, retrieved: list[dict]) -> tuple[str, list[dict]]:
        """Shared helper: build context string + sources list from retrieved chunks."""
        sources = []
        context_parts = []
        for r in retrieved:
            context_parts.append(
                f"[来源: {r['title']}]"
                f"(#{'/'.join(r.get('sources', [r.get('source', 'unknown')]))}):\n"
                f"{r['content']}"
            )
            sources.append({
                "title": r["title"],
                "content_preview": r["content"][:200],
                "source": "+".join(r.get("sources", [r.get("source", "unknown")])),
                "score": round(r.get("rerank_score", r.get("rrf_score", r.get("score", 0))), 4),
                "doc_id": int(r["doc_id"]) if r.get("doc_id", "").isdigit() else 0,
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else "未找到相关知识。"
        user_prompt = f"""知识内容：
{context}

用户问题：{question}"""
        return user_prompt, sources

    async def stream_query(
        self,
        question: str,
        top_k: int = 10,
        conversation_id: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        trace("stream_query_started", question=question)

        history = None
        if conversation_id:
            history = self.conversation.get_messages(conversation_id)

        try:
            rewritten = await self.query_processor.rewrite_query(question, history)
        except Exception as e:
            logger.warning("Query rewrite failed: {}", e)
            rewritten = question

        try:
            retrieved = await self._hybrid_retrieve(rewritten, top_k=top_k)
        except Exception as e:
            logger.error("Retrieval failed: {}", e)
            retrieved = []

        user_prompt, sources = self._build_query_context(question, retrieved)

        import json
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'rewritten_query': rewritten if rewritten != question else None})}\n\n"

        full_answer = ""
        try:
            async for chunk in self.llm_client.astream(SYSTEM_PROMPT, user_prompt):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        except Exception as e:
            logger.error("Stream error: {}", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        if conversation_id:
            self.conversation.add_message(conversation_id, "user", question)
            self.conversation.add_message(conversation_id, "assistant", full_answer, sources)

        trace("stream_query_completed")

    def list_documents(self, category: Optional[str] = None) -> list[dict]:
        docs = self.sql_store.list_documents(category)
        result = []
        for d in docs:
            chunks = self.sql_store.get_chunks_by_doc_id(d["id"])
            result.append({
                "id": d["id"],
                "title": d["title"],
                "category": d.get("category", ""),
                "tags": d["tags"].split(",") if d["tags"] else [],
                "created_at": d["created_at"],
                "chunk_count": len(chunks),
            })
        return result

    def delete_document(self, doc_id: int) -> bool:
        doc = self.sql_store.get_document(doc_id)
        if doc is None:
            return False

        filename = doc["filename"]

        # Step 1: Delete from vector store first (best-effort)
        try:
            self.vector_store.delete_by_doc_id(doc_id)
        except Exception as e:
            logger.warning("Vector store delete failed for doc {}: {}", doc_id, e)

        # Step 2: Delete from SQLite (atomic)
        from app.database.base import DatabaseManager
        db = DatabaseManager.get_instance()
        with db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        # Step 3: Delete markdown file (best-effort)
        try:
            self.markdown_store.delete(filename)
        except Exception as e:
            logger.warning("Markdown delete failed for {}: {}", filename, e)

        trace("document_deleted", doc_id=doc_id)
        self._update_metrics()
        return True

    def get_document_markdown(self, doc_id: int) -> Optional[str]:
        doc = self.sql_store.get_document(doc_id)
        if doc is None:
            return None
        return self.markdown_store.read(doc["filename"])

    def get_stats(self) -> dict:
        docs = self.sql_store.list_documents()
        return {
            "total_documents": len(docs),
            "vector_chunks": self.vector_store.count(),
            "semantic_chunks": self.vector_store.count(),
            "lsi_enabled": True,
            "llm_rerank_enabled": True,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_max": self._cache_max,
        }

    def _update_metrics(self):
        try:
            docs = self.sql_store.list_documents()
            set_document_metrics(len(docs), self.vector_store.count())
        except Exception:
            pass
