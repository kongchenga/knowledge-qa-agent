from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_embeddings():
    """Mock embedding and reranker services to avoid model downloads."""
    with patch("app.embeddings.get_embedding_service") as mock_emb:
        emb_instance = MagicMock()
        emb_instance.embed_query.return_value = [0.1] * 512
        emb_instance.embed_documents.return_value = [[0.1] * 512]
        emb_instance.dim = 512
        mock_emb.return_value = emb_instance

        with patch("app.database.vector_store.create_vector_store") as mock_store:
            vs_instance = MagicMock()
            vs_instance.count.return_value = 0
            vs_instance.hybrid_search.return_value = []
            mock_store.return_value = vs_instance

            with patch("app.reranker.get_reranker") as mock_rank:
                rank_instance = MagicMock()
                rank_instance.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]
                mock_rank.return_value = rank_instance

                yield


class TestQAAgent:
    def test_init(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        assert agent.llm_client is not None
        assert agent.sql_store is not None
        assert agent.markdown_store is not None

    def test_add_document(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        result = agent.add_document(
            title="测试文档",
            content="这是测试文档的内容。包含一些知识信息。",
            tags=["test", "pytest"],
            category="testing",
        )
        assert result["doc_id"] > 0
        assert result["chunk_count"] >= 1
        assert "filename" in result

    def test_list_documents(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        agent.add_document("Doc1", "内容1", tags=["a"])
        agent.add_document("Doc2", "内容2", tags=["b"])
        docs = agent.list_documents()
        assert len(docs) >= 2

    def test_delete_document(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        result = agent.add_document("ToDelete", "将被删除")
        doc_id = result["doc_id"]
        assert agent.delete_document(doc_id) is True
        assert agent.delete_document(99999) is False

    def test_get_document_markdown(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        result = agent.add_document("MarkdownTest", "这是markdown内容")
        md = agent.get_document_markdown(result["doc_id"])
        assert md is not None
        assert "MarkdownTest" in md

    def test_get_stats(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        stats = agent.get_stats()
        assert "total_documents" in stats
        assert "vector_chunks" in stats

    @pytest.mark.asyncio
    async def test_query_no_llm(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        agent.add_document("知识文档", "北京是中国的首都。")
        result = await agent.query("北京是哪个国家的首都？")
        assert "answer" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_stream_query(self):
        from app.agent.qa_agent import QAAgent
        agent = QAAgent()
        agent.add_document("测试", "这是测试内容")
        chunks = []
        async for chunk in agent.stream_query("测试问题"):
            chunks.append(chunk)
        assert len(chunks) > 0
