"""Claude backend for the capability experiments (official Anthropic SDK).

Server-side refusal fallbacks are enabled (``fallbacks: "default"``), so a
request a safety classifier declines is retried on a fallback model inside the
same call. A request that is still refused returns an empty string.
"""
from __future__ import annotations

import anthropic

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=5)
    return _client


async def complete(model: str, prompt: str, effort: str = "low") -> str:
    response = await client().beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in response.content if b.type == "text")


async def complete_messages(model: str, messages: list[dict], effort: str = "low") -> tuple[str, bool]:
    """Multi-turn call without refusal fallbacks (the attacker study must stay on one model).

    Returns (text, refused). Consecutive same-role messages are allowed by the API.
    """
    response = await client().messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": effort},
        messages=messages,
    )
    if response.stop_reason == "refusal":
        return "", True
    return "".join(b.text for b in response.content if b.type == "text"), False
