from __future__ import annotations

import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# ── Prometheus metrics (if prometheus_client is available) ──────────────────

_metrics_enabled = False
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

    http_requests_total = Counter(
        "http_requests_total", "Total HTTP requests",
        ["method", "path", "status"],
    )
    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds", "HTTP request duration",
        ["method", "path"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    llm_requests_total = Counter(
        "llm_requests_total", "Total LLM requests",
        ["provider", "status"],
    )
    llm_request_duration_seconds = Histogram(
        "llm_request_duration_seconds", "LLM request duration",
        ["provider"],
        buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
    retrieval_total = Counter(
        "retrieval_total", "Total retrieval operations",
        ["method"],
    )
    retrieval_duration_seconds = Histogram(
        "retrieval_duration_seconds", "Retrieval duration",
        ["method"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    )
    documents_total = Gauge("documents_total", "Total documents in store")
    vector_chunks_total = Gauge("vector_chunks_total", "Total vector chunks")
    active_requests = Gauge("active_requests", "Currently active requests")
    _metrics_enabled = True
except ImportError:
    pass


def setup_logging():
    logger.remove()
    logger.configure(extra={"request_id": ""})

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[request_id]: >12}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    log_dir = settings.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[request_id]: >12} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[request_id]: >12} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str = "", **kwargs):
    if name:
        kwargs.setdefault("name", name)
    return logger.bind(**kwargs)


def trace(msg: str, **extra):
    rid = request_id_var.get()
    logger.bind(request_id=rid).info(msg, **extra)


def make_request_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Metrics helpers ─────────────────────────────────────────────────────────

def get_metrics_registry():
    if _metrics_enabled:
        return REGISTRY
    return None


def generate_metrics() -> Optional[bytes]:
    if _metrics_enabled:
        return generate_latest()
    return None


def observe_llm_request(provider: str, duration: float, status: str = "ok"):
    if _metrics_enabled:
        llm_requests_total.labels(provider=provider, status=status).inc()
        llm_request_duration_seconds.labels(provider=provider).observe(duration)


def observe_retrieval(method: str, duration: float):
    if _metrics_enabled:
        retrieval_total.labels(method=method).inc()
        retrieval_duration_seconds.labels(method=method).observe(duration)


def set_document_metrics(doc_count: int, chunk_count: int):
    if _metrics_enabled:
        documents_total.set(doc_count)
        vector_chunks_total.set(chunk_count)
