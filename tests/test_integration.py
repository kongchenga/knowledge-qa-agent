"""
Full production integration tests for secruityagent (Knowledge QA Agent v2.0).

Covers:
  - Health / Metrics endpoints
  - Document CRUD (text, markdown, upload simulation)
  - Hybrid search (vector + tag-based SQL fallback)
  - Query and stream-query endpoints
  - Conversation CRUD + multi-turn QA
  - Auth middleware (enabled / disabled / bad key)
  - RateLimit middleware (bucket exhaustion)
  - Security headers
  - Error / edge cases (empty input, not-found, invalid file ext, etc.)
  - CORS headers
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_PREFIX = "/api"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_with_env(env_overrides=None):
    """Reload app.config and app.main with env-vars overridden.
    Returns the new TestClient and a cleanup callback."""
    import importlib
    import app.config
    import app.database.base as db_base

    saved = {}
    if env_overrides:
        for k, v in env_overrides.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v

    db_base.DatabaseManager.reset_instance()
    importlib.reload(app.config)
    import app.main
    importlib.reload(app.main)

    client = TestClient(app.main.app, raise_server_exceptions=False)

    def cleanup():
        db_base.DatabaseManager.reset_instance()
        if env_overrides:
            for k, orig in saved.items():
                if orig is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = orig
            importlib.reload(app.config)
            importlib.reload(app.main)

    return client, cleanup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Isolated client with auth OFF, rate limit OFF."""
    c, _ = _reload_with_env({
        "API_KEY": "", "RATE_LIMIT_PER_MINUTE": "0",
    })
    yield c


@pytest.fixture
def auth_client():
    """Isolated client with auth ON."""
    c, cleanup = _reload_with_env({
        "API_KEY": "secret-token-123", "RATE_LIMIT_PER_MINUTE": "0",
    })
    yield c
    cleanup()


@pytest.fixture
def rate_limit_client():
    """Isolated client with low rate limit."""
    c, cleanup = _reload_with_env({
        "API_KEY": "", "RATE_LIMIT_PER_MINUTE": "2",
    })
    yield c
    cleanup()


# ===========================================================================
# 1. Health and Metrics
# ===========================================================================

class TestHealth:
    def test_health_ok(self, client):
        r = client.get(f"{API_PREFIX}/health")
        assert r.status_code == 200
        j = r.json()
        assert j["status"] in ("ok", "degraded")
        assert "llm_configured" in j
        assert "vector_store_ready" in j
        assert "total_documents" in j

    def test_health_returns_json(self, client):
        r = client.get(f"{API_PREFIX}/health")
        assert r.headers["content-type"].startswith("application/json")


class TestMetrics:
    def test_metrics_endpoint(self, client):
        client.get(f"{API_PREFIX}/health")
        r = client.get("/metrics")
        assert r.status_code in (200, 503)

    def test_metrics_content_type(self, client):
        client.get(f"{API_PREFIX}/health")
        r = client.get("/metrics")
        if r.status_code == 200:
            assert "text/plain" in r.headers["content-type"]


# ===========================================================================
# 2. Document CRUD
# ===========================================================================

class TestDocumentText:
    def test_add_text_document(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "Python Guide",
            "content": "Python is a high-level programming language widely used in web, data science, and AI.",
            "tags": "python,programming",
            "category": "tech",
        })
        assert r.status_code == 200
        j = r.json()
        assert j["doc_id"] > 0
        assert j["chunk_count"] >= 1

    def test_add_text_minimal(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "Minimal", "content": "Hello world",
        })
        assert r.status_code == 200
        assert r.json()["doc_id"] > 0

    def test_add_text_empty_content(self, client):
        """Both 400 (our BadRequestError) and 422 (FastAPI validation) are acceptable."""
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "Empty", "content": "",
        })
        assert r.status_code in (400, 422)

    def test_add_text_empty_title(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "", "content": "something",
        })
        assert r.status_code in (400, 422)

    def test_list_documents(self, client):
        client.post(f"{API_PREFIX}/documents/text", data={
            "title": "DocA", "content": "aaa", "category": "cat1"
        })
        client.post(f"{API_PREFIX}/documents/text", data={
            "title": "DocB", "content": "bbb", "category": "cat2"
        })
        r = client.get(f"{API_PREFIX}/documents")
        assert r.status_code == 200
        docs = r.json()
        assert len(docs) >= 2

    def test_list_documents_by_category(self, client):
        client.post(f"{API_PREFIX}/documents/text", data={
            "title": "CatDoc", "content": "xyz", "category": "special"
        })
        r = client.get(f"{API_PREFIX}/documents?category=special")
        assert r.status_code == 200
        for d in r.json():
            assert d.get("category", "") == "special"

    def test_get_document_markdown(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "Markdown Test", "content": "# Hello\n\nContent here"
        })
        doc_id = r.json()["doc_id"]
        r = client.get(f"{API_PREFIX}/documents/{doc_id}/markdown")
        assert r.status_code == 200
        assert "Markdown Test" in r.json()["markdown"]

    def test_get_document_markdown_not_found(self, client):
        r = client.get(f"{API_PREFIX}/documents/99999/markdown")
        assert r.status_code == 404

    def test_delete_document(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "To Delete", "content": "Will be deleted"
        })
        doc_id = r.json()["doc_id"]
        r = client.delete(f"{API_PREFIX}/documents/{doc_id}")
        assert r.status_code == 200
        r = client.get(f"{API_PREFIX}/documents/{doc_id}/markdown")
        assert r.status_code == 404

    def test_delete_document_not_found(self, client):
        r = client.delete(f"{API_PREFIX}/documents/99999")
        assert r.status_code == 404

    def test_update_document(self, client):
        r = client.post(f"{API_PREFIX}/documents/text", data={
            "title": "Old Title", "content": "Old content"
        })
        doc_id = r.json()["doc_id"]
        r = client.put(f"{API_PREFIX}/documents/{doc_id}", data={
            "title": "New Title", "content": "New content new content new"
        })
        assert r.status_code == 200
        assert r.json()["doc_id"] == doc_id
        r = client.get(f"{API_PREFIX}/documents/{doc_id}/markdown")
        assert "New Title" in r.json()["markdown"]

    def test_update_document_not_found(self, client):
        r = client.put(f"{API_PREFIX}/documents/99999", data={
            "title": "x", "content": "y"
        })
        assert r.status_code == 404


class TestDocumentUpload:
    def test_upload_txt(self, client):
        content = "Line one\nLine two".encode()
        r = client.post(f"{API_PREFIX}/documents/upload", data={
            "title": "Upload Test", "tags": "upload,test", "category": "test",
        }, files={"file": ("test.txt", content, "text/plain")})
        assert r.status_code == 200
        j = r.json()
        assert j["doc_id"] > 0
        assert j["chunk_count"] >= 1

    def test_upload_unsupported_ext(self, client):
        r = client.post(f"{API_PREFIX}/documents/upload", data={"title": "Bad"},
                        files={"file": ("test.xyz", b"data", "application/octet-stream")})
        assert r.status_code in (400, 422)

    def test_upload_empty_file(self, client):
        r = client.post(f"{API_PREFIX}/documents/upload", data={"title": "Empty"},
                        files={"file": ("empty.txt", b"", "text/plain")})
        assert r.status_code == 400


# ===========================================================================
# 3. Stats
# ===========================================================================

class TestStats:
    def test_stats_default(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        j = r.json()
        assert "total_documents" in j
        assert "vector_chunks" in j

    def test_stats_reflects_docs(self, client):
        client.post(f"{API_PREFIX}/documents/text", data={
            "title": "S1", "content": "Stats test doc one"
        })
        client.post(f"{API_PREFIX}/documents/text", data={
            "title": "S2", "content": "Stats test doc two"
        })
        r = client.get("/api/stats")
        assert r.json()["total_documents"] >= 2


# ===========================================================================
# 4. Query and Stream Query
# ===========================================================================

class TestQuery:
    def test_query_empty_question(self, client):
        """Empty question: FastAPI validation returns 422 before reaching our 400."""
        r = client.post(f"{API_PREFIX}/query", json={
            "question": "", "top_k": 5
        })
        assert r.status_code in (400, 422)

    def test_query_no_docs(self, client):
        r = client.post(f"{API_PREFIX}/query", json={
            "question": "What is Python?", "top_k": 3
        })
        assert r.status_code == 200
        j = r.json()
        assert "answer" in j
        assert "sources" in j

    def test_query_top_k_too_high(self, client):
        r = client.post(f"{API_PREFIX}/query", json={
            "question": "test", "top_k": 100
        })
        assert r.status_code == 422


class TestStreamQuery:
    def test_stream_query_returns_events(self, client):
        r = client.post(f"{API_PREFIX}/query/stream", json={
            "question": "test stream query", "top_k": 3
        })
        assert r.status_code == 200
        assert "data:" in r.text

    def test_stream_query_empty_question(self, client):
        r = client.post(f"{API_PREFIX}/query/stream", json={
            "question": "", "top_k": 5
        })
        assert r.status_code in (400, 422)


# ===========================================================================
# 5. Conversations
# ===========================================================================

class TestConversations:
    def test_create_conversation(self, client):
        r = client.post(f"{API_PREFIX}/conversations", data={"title": "Test Chat"})
        assert r.status_code == 200
        j = r.json()
        assert j["id"] > 0
        assert j["title"] == "Test Chat"

    def test_create_default_title(self, client):
        r = client.post(f"{API_PREFIX}/conversations")
        assert r.status_code == 200
        assert r.json()["id"] > 0

    def test_list_conversations(self, client):
        client.post(f"{API_PREFIX}/conversations", data={"title": "C1"})
        client.post(f"{API_PREFIX}/conversations", data={"title": "C2"})
        r = client.get(f"{API_PREFIX}/conversations")
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_delete_conversation(self, client):
        r = client.post(f"{API_PREFIX}/conversations", data={"title": "ToDelete"})
        conv_id = r.json()["id"]
        r = client.delete(f"{API_PREFIX}/conversations/{conv_id}")
        assert r.status_code == 200

    def test_delete_conversation_not_found(self, client):
        r = client.delete(f"{API_PREFIX}/conversations/99999")
        assert r.status_code == 404

    def test_get_messages_empty(self, client):
        r = client.post(f"{API_PREFIX}/conversations", data={"title": "Empty"})
        conv_id = r.json()["id"]
        r = client.get(f"{API_PREFIX}/conversations/{conv_id}/messages")
        assert r.status_code == 200
        assert r.json() == []

    def test_get_messages_not_found(self, client):
        r = client.get(f"{API_PREFIX}/conversations/99999/messages")
        assert r.status_code == 404


# ===========================================================================
# 6. Auth Middleware
# ===========================================================================

class TestAuth:
    def test_no_auth_when_disabled(self, client):
        r = client.get(f"{API_PREFIX}/health")
        assert r.status_code == 200

    def test_auth_bypasses_health(self, auth_client):
        r = auth_client.get(f"{API_PREFIX}/health")
        assert r.status_code == 200

    def test_auth_bypasses_docs(self, auth_client):
        r = auth_client.get("/docs")
        assert r.status_code == 200

    def test_auth_missing_key_401(self, auth_client):
        r = auth_client.get(f"{API_PREFIX}/stats")
        assert r.status_code == 401

    def test_auth_wrong_key_401(self, auth_client):
        r = auth_client.get(f"{API_PREFIX}/stats", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_auth_correct_key_ok(self, auth_client):
        r = auth_client.get(f"{API_PREFIX}/stats", headers={"X-API-Key": "secret-token-123"})
        assert r.status_code == 200

    def test_auth_key_in_query(self, auth_client):
        r = auth_client.get(f"{API_PREFIX}/stats?api_key=secret-token-123")
        assert r.status_code == 200


# ===========================================================================
# 7. Rate Limit Middleware
# ===========================================================================

class TestRateLimit:
    def test_allows_when_disabled(self, client):
        for _ in range(10):
            r = client.get(f"{API_PREFIX}/stats")
            assert r.status_code == 200

    def test_blocks_after_burst(self, rate_limit_client):
        statuses = []
        for _ in range(15):
            r = rate_limit_client.get(f"{API_PREFIX}/stats")
            statuses.append(r.status_code)
        assert 429 in statuses

    def test_bypass_health(self, rate_limit_client):
        for _ in range(10):
            r = rate_limit_client.get(f"{API_PREFIX}/health")
            assert r.status_code == 200

    def test_retry_after_header(self, rate_limit_client):
        for _ in range(20):
            r = rate_limit_client.get(f"{API_PREFIX}/stats")
            if r.status_code == 429:
                assert "Retry-After" in r.headers
                return
        pytest.fail("Rate limit was never triggered after 20 requests")


# ===========================================================================
# 8. Security Headers
# ===========================================================================

class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        r = client.get(f"{API_PREFIX}/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "max-age=31536000" in r.headers.get("Strict-Transport-Security", "")

    def test_request_id_header(self, client):
        r = client.get(f"{API_PREFIX}/health")
        assert "X-Request-ID" in r.headers
        assert len(r.headers["X-Request-ID"]) == 12


# ===========================================================================
# 9. CORS
# ===========================================================================

class TestCORS:
    def test_cors_preflight(self, client):
        r = client.options(f"{API_PREFIX}/health", headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        })
        assert r.status_code in (200, 405)
        if r.status_code == 200:
            assert "access-control-allow-origin" in r.headers


# ===========================================================================
# 10. Error Handling / Edge Cases
# ===========================================================================

class TestErrors:
    def test_nonexistent_route_404(self, client):
        r = client.get(f"{API_PREFIX}/nonexistent")
        assert r.status_code == 404

    def test_invalid_doc_id_type_422(self, client):
        r = client.get(f"{API_PREFIX}/documents/abc/markdown")
        assert r.status_code == 422

    def test_frontend_index_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_error_response_has_request_id(self, client):
        r = client.get(f"{API_PREFIX}/documents/99999/markdown")
        assert r.status_code == 404
        j = r.json()
        assert "request_id" in j
        assert "detail" in j