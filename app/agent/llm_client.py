from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional

from openai import AsyncOpenAI

from app.config import settings
from app.monitoring import get_logger, observe_llm_request

logger = get_logger(__name__)


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._max_retries = settings.llm_max_retries
        self._timeout = settings.llm_request_timeout
        if settings.llm_api_key:
            self._init_client()
            logger.info(
                "LLM ready: provider={} model={} base_url={} max_retries={}",
                settings.llm_provider, settings.llm_model, settings.llm_base_url,
                self._max_retries,
            )
        else:
            logger.warning("LLM API key not set. LLM features disabled.")

    def _init_client(self):
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=self._timeout,
            max_retries=0,  # we handle retries ourselves
        )

    def _ensure_ready(self):
        if self._client is None:
            if not settings.llm_api_key:
                raise LLMError("LLM_API_KEY is not set")
            self._init_client()

    def is_ready(self) -> bool:
        return bool(settings.llm_api_key)

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Send a chat request and return the text response."""
        self._ensure_ready()
        messages = self._build_messages(system_prompt, user_prompt)
        resp = await self._chat_with_retry(messages, tools=tools)
        return resp.choices[0].message.content or ""

    async def achat_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
    ) -> dict:
        """Send a chat request with native function calling support.
        Returns {"type": "text", "content": "..."} or
                {"type": "tool_call", "name": "...", "arguments": "..."}."""
        self._ensure_ready()
        messages = self._build_messages(system_prompt, user_prompt)
        resp = await self._chat_with_retry(messages, tools=tools)
        choice = resp.choices[0]
        msg = choice.message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return {
                "type": "tool_call",
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "tool_call_id": tc.id,
            }
        return {"type": "text", "content": msg.content or ""}

    async def _chat_with_retry(self, messages: list[dict], tools: Optional[list[dict]] = None):
        last_error: Optional[Exception] = None
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries + 1):
            try:
                import time
                t0 = time.monotonic()
                resp = await self._client.chat.completions.create(**kwargs)
                duration = time.monotonic() - t0
                observe_llm_request(settings.llm_provider, duration, "ok")
                return resp
            except Exception as e:
                last_error = e
                duration = time.monotonic() - t0 if 't0' in dir() else 0
                observe_llm_request(settings.llm_provider, duration, "error")
                if attempt < self._max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "LLM call failed (attempt {}/{}): {}. Retrying in {}s...",
                        attempt + 1, self._max_retries + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    err_str = str(e)
                    if len(settings.llm_api_key) > 8 and settings.llm_api_key[-8:] in err_str:
                        err_str = err_str.replace(settings.llm_api_key[-8:], "***")
                    logger.error("LLM call failed after {} attempts: {}", self._max_retries + 1, err_str)
                    raise LLMError(f"LLM call failed after retries") from e

    async def astream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncGenerator[str, None]:
        self._ensure_ready()
        messages = self._build_messages(system_prompt, user_prompt)
        try:
            stream = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error("LLM stream error: {}", e)
            yield f"\n\n[LLM stream error: {e}]"
