from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.config import settings
from app.exceptions import AppError
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security import SecurityHeadersMiddleware
from app.monitoring import (
    setup_logging,
    get_logger,
    request_id_var,
    make_request_id,
    http_requests_total,
    http_request_duration_seconds,
    active_requests,
    generate_metrics,
    _metrics_enabled,
)
from app.routers.api import create_router

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Knowledge QA Agent v2.0.0")
    logger.info("LLM provider={} model={}", settings.llm_provider, settings.llm_model)
    logger.info("Embedding model={}", settings.embedding_model)
    logger.info("Auth={}", "enabled" if settings.api_key else "disabled")

    warnings = settings.check_startup()
    for w in warnings:
        logger.warning("Startup check: {}", w)

    # ── Pre-warm embedding + reranker models ──────────────────────
    import time as _time
    t0 = _time.monotonic()
    try:
        from app.embeddings import get_embedding_service
        get_embedding_service().embed_query("warmup")
        logger.info("Embedding model pre-warmed ({:.1f}s)", _time.monotonic() - t0)
    except Exception as e:
        logger.warning("Embedding warmup failed: {}", e)

    t0 = _time.monotonic()
    try:
        from app.reranker import get_reranker
        get_reranker().rerank("warmup", [{"content": "warmup test document for preloading the cross-encoder model into memory", "chunk_id": "_warmup_", "title": "", "doc_id": "", "score": 1.0}])
        logger.info("Reranker model pre-warmed ({:.1f}s)", _time.monotonic() - t0)
    except Exception as e:
        logger.warning("Reranker warmup failed: {}", e)

    # ── Graceful shutdown ─────────────────────────────────────────
    yield
    logger.info("Shutting down — closing resources...")
    try:
        # Gracefully close can be added here when app state is wired up
        pass
    except Exception as e:
        logger.warning("Shutdown cleanup error: {}", e)
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="知识问答 Agent",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost first) ──────────────────────
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)

    # ── Request metrics middleware (inlined for performance) ────────────
    @app.middleware("http")
    async def metrics_and_tracing(request: Request, call_next):
        rid = make_request_id()
        request_id_var.set(rid)
        t0 = time.monotonic()

        if _metrics_enabled:
            active_requests.inc()

        try:
            response: Response = await call_next(request)
        finally:
            if _metrics_enabled:
                active_requests.dec()

        duration = time.monotonic() - t0
        response.headers["X-Request-ID"] = rid

        if _metrics_enabled and request.url.path != "/metrics":
            http_requests_total.labels(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                path=request.url.path,
            ).observe(duration)

        return response

    # ── Global exception handler ────────────────────────────────────────
    @app.exception_handler(AppError)
    async def app_exception_handler(request: Request, exc: AppError):
        rid = request_id_var.get()
        logger.warning("AppError: {} | path={} | extra={}", exc.detail, request.url.path, exc.extra)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": rid or "",
                **(exc.extra or {}),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        rid = request_id_var.get()
        err_str = str(exc)
        # Sanitize sensitive data from error logs
        if len(settings.llm_api_key) > 8:
            err_str = err_str.replace(settings.llm_api_key, "[REDACTED]")
        logger.error("Unhandled exception: {} | path={}", err_str, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": rid or "",
            },
        )

    # ── Routes ──────────────────────────────────────────────────────────
    app.include_router(create_router(), prefix="/api")

    # ── Metrics endpoint ────────────────────────────────────────────────
    if _metrics_enabled:

        @app.get("/metrics")
        async def metrics():
            data = generate_metrics()
            if data is None:
                return JSONResponse({"error": "Metrics disabled"}, status_code=503)
            return Response(content=data, media_type="text/plain; version=0.0.4")

    # ── Frontend ────────────────────────────────────────────────────────
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    return app


app = create_app()
