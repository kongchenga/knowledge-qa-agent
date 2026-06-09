from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.monitoring import get_logger

logger = get_logger("rate_limit")


class TokenBucket:
    """Async-safe token bucket using asyncio.Lock — never blocks the event loop."""

    def __init__(self, rate: int, per_seconds: int = 60):
        self.rate = rate
        self.per_seconds = per_seconds
        self.tokens = float(rate)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.rate), self.tokens + elapsed * self.rate / self.per_seconds)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._buckets: dict[str, TokenBucket] = {}
        self._cleanup_interval = 300.0
        self._last_cleanup = time.monotonic()
        self._cleanup_lock = asyncio.Lock()

    async def _get_bucket(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(settings.rate_limit_per_minute, 60)
        return self._buckets[key]

    async def _cleanup(self):
        now = time.monotonic()
        if now - self._last_cleanup > self._cleanup_interval:
            async with self._cleanup_lock:
                self._buckets.clear()
                self._last_cleanup = now

    async def dispatch(self, request: Request, call_next):
        if settings.rate_limit_per_minute <= 0:
            return await call_next(request)

        if request.url.path in ("/api/health", "/metrics", "/docs", "/openapi"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = await self._get_bucket(client_ip)

        await self._cleanup()

        if not await bucket.allow():
            logger.warning("Rate limit exceeded for IP: {}", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests", "retry_after": "60s"},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
