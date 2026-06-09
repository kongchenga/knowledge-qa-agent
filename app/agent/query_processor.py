from __future__ import annotations

import re
from typing import Optional

from app.agent.llm_client import LLMClient
from app.monitoring import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """你是一个查询优化助手。请根据对话历史，将用户的问题改写为独立、完整的检索查询。
要求：
1. 补充代词所指代的具体内容
2. 保持原问题的核心意图
3. 直接输出改写后的查询，不要额外解释

对话历史：
{history}

用户问题：{question}

改写后的查询："""


class QueryProcessor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_tags(self, question: str) -> list[str]:
        tags = re.findall(r"#(\w+)", question)
        return [t for t in tags if not t.isdigit() and len(t) > 1]

    async def rewrite_query(
        self,
        question: str,
        history: Optional[list[dict]] = None,
    ) -> str:
        # Skip LLM call entirely if no conversation history
        if not history or len(history) < 2:
            return question

        # Only include last 3 exchanges (6 messages)
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in history[-6:]
        )

        # Quick check: if the question doesn't contain pronouns/references,
        # skip rewrite to save an LLM round-trip
        ref_indicators = re.findall(r'[它他她]|这个|那个|这些|那些|上面|前面|刚才|刚刚|之前|继续|再', question)
        if not ref_indicators:
            return question

        try:
            rewritten = await self.llm.achat(
                system_prompt="你是一个查询优化助手。",
                user_prompt=REWRITE_PROMPT.format(
                    history=history_text,
                    question=question,
                ),
            )
            rewritten = rewritten.strip().strip('"\'')
            if rewritten and len(rewritten) >= 2:
                logger.debug("Query rewritten: {} -> {}", question, rewritten)
                return rewritten
            return question
        except Exception as e:
            logger.warning("Query rewrite failed: {}", e)
            return question

    def decompose_query(self, question: str) -> list[str]:
        sub_questions = re.split(r"[?？;；]\s*", question)
        return [q.strip() for q in sub_questions if len(q.strip()) > 5] or [question]
