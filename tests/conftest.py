from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def session_env():
    """One-time: forcibly set env vars and create a shared temp dir via os.makedirs
    (avoids tempfile.mkdtemp ACL bug on Windows sandbox)."""
    tmpdir = Path(os.environ.get("TMP", os.environ.get("TEMP", "."))) / f"pytest_{os.urandom(6).hex()}"
    os.makedirs(str(tmpdir), exist_ok=True)
    os.environ["LLM_API_KEY"] = "sk-test-key"
    os.environ["LLM_BASE_URL"] = "https://api.deepseek.com"
    os.environ["LLM_MODEL"] = "deepseek-chat"
    os.environ["SQLITE_PATH"] = str(tmpdir / "test.db")
    os.environ["CHROMA_PERSIST_DIR"] = str(tmpdir / "chroma")
    os.environ["KNOWLEDGE_DIR"] = str(tmpdir / "knowledge")
    os.environ["LOG_DIR"] = str(tmpdir / "logs")
    os.environ["LOG_LEVEL"] = "ERROR"
    os.environ["API_KEY"] = ""
    os.environ["CHUNK_SIZE"] = "500"
    os.environ["CHUNK_OVERLAP"] = "50"
    yield

    gc.collect()
    import app.database.base as db_base
    db_base.DatabaseManager.reset_instance()
    gc.collect()
    time.sleep(0.5)
    shutil.rmtree(str(tmpdir), ignore_errors=True)


@pytest.fixture(autouse=True)
def test_isolation(monkeypatch):
    """Per-test isolation: override storage paths with fresh temp dirs so state
    does not leak between tests.  Applies to every test unconditionally."""
    import app.config as config_module
    s = config_module.settings

    temp_root = Path(os.environ.get("TMP", "."))
    tmpdir = temp_root / f"pt_{os.urandom(6).hex()}"
    os.makedirs(str(tmpdir), exist_ok=True)
    try:
        monkeypatch.setattr(s, "sqlite_path", str(tmpdir / "test.db"))
        monkeypatch.setattr(s, "chroma_persist_dir", str(tmpdir / "chroma"))
        monkeypatch.setattr(s, "knowledge_dir", str(tmpdir / "knowledge"))

        import app.database.base as db_base
        db_base.DatabaseManager.reset_instance()

        yield

        db_base.DatabaseManager.reset_instance()
        gc.collect()
        time.sleep(0.2)
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)