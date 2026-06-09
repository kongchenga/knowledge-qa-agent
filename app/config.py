from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "deepseek"
    llm_api_key: str = Field("", description="DeepSeek/OpenAI compatible API key")
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=64, le=128000)
    llm_request_timeout: int = Field(default=60, ge=10, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_dim: int = 512

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_top_k: int = Field(default=3, ge=1, le=20)

    # Retrieval
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    top_k: int = Field(default=10, ge=1, le=100)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # Storage
    chroma_persist_dir: str = ""
    sqlite_path: str = ""
    knowledge_dir: str = ""

    # Auth
    api_key: str = ""
    api_key_header: str = "X-API-Key"

    # HuggingFace
    huggingface_token: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = Field(default=8020, ge=1, le=65535)
    debug: bool = False
    log_level: str = "INFO"
    log_dir: str = "logs"

    # CORS
    cors_origins: list[str] = ["*"]

    # Rate limit
    rate_limit_per_minute: int = Field(default=60, ge=0, le=10000)

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    cache_max_size: int = Field(default=256, ge=0, le=10000)

    @field_validator("llm_api_key")
    @classmethod
    def warn_empty_api_key(cls, v: str) -> str:
        if not v:
            import logging
            logging.warning(
                "LLM_API_KEY is not set. LLM features will be disabled. "
                "Set the LLM_API_KEY environment variable or add it to .env file."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            return "INFO"
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_dim(cls, v: int) -> int:
        if v <= 0:
            return 512
        return v

    @property
    def resolved_chroma_dir(self) -> Path:
        return Path(self.chroma_persist_dir or (BASE_DIR / "data" / "chroma"))

    @property
    def resolved_sqlite_path(self) -> Path:
        return Path(self.sqlite_path or (BASE_DIR / "data" / "knowledge.db"))

    @property
    def resolved_knowledge_dir(self) -> Path:
        return Path(self.knowledge_dir or (BASE_DIR / "knowledge"))

    @property
    def resolved_log_dir(self) -> Path:
        return Path(self.log_dir)

    def check_startup(self) -> list[str]:
        warnings: list[str] = []

        if not self.llm_api_key:
            warnings.append("LLM_API_KEY not set — LLM query/stream endpoints will fail")

        directories = [
            ("SQLite data", self.resolved_sqlite_path.parent),
            ("Chroma data", self.resolved_chroma_dir),
            ("Knowledge files", self.resolved_knowledge_dir),
            ("Logs", self.resolved_log_dir),
        ]
        for name, path in directories:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.append(f"Cannot create {name} directory at {path}: {e}")

        if self.debug:
            warnings.append("DEBUG mode is ON — do not use in production")

        if self.cors_origins == ["*"]:
            warnings.append("CORS is wide open (*) — restrict in production")

        if not self.api_key:
            warnings.append("API_KEY not set — auth is DISABLED")

        return warnings


BASE_DIR = Path(__file__).resolve().parent.parent
settings = Settings()
