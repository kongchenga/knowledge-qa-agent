from __future__ import annotations


def test_default_values():
    from app.config import Settings
    s = Settings()
    assert s.llm_provider == "deepseek"
    assert s.llm_model == "deepseek-chat"
    assert s.port == 8020
    assert s.hybrid_alpha == 0.5
    assert s.chunk_size >= 100


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4")
    monkeypatch.setenv("PORT", "9000")
    from app.config import Settings
    s = Settings()
    assert s.llm_model == "gpt-4"
    assert s.port == 9000


def test_validation():
    from app.config import Settings
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"

    s = Settings(log_level="invalid")
    assert s.log_level == "INFO"


def test_startup_checks():
    from app.config import Settings
    s = Settings(llm_api_key="", api_key="")
    warnings = s.check_startup()
    assert any("LLM_API_KEY" in w for w in warnings)
    assert any("API_KEY" in w for w in warnings)
