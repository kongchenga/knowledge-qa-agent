from __future__ import annotations

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.api_key:
            return await call_next(request)

        if request.url.path.startswith(("/docs", "/openapi", "/health")):
            return await call_next(request)

        api_key = request.headers.get(settings.api_key_header)
        if not api_key:
            api_key = request.query_params.get("api_key")

        if api_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
