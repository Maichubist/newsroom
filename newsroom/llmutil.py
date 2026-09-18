"""Shared OpenAI chat wrapper — one place to cap output and log real token usage.

Every LLM JSON call goes through `chat_json` so we can (a) bound output tokens per
operation (a runaway JSON response can't cost unboundedly, and a too-small cap only
truncates → the caller's parser falls back safely) and (b) record model + prompt /
completion / cached tokens per op. That usage log is the data we need to optimise
spend by measurement instead of guesswork (which prompts are big, where caching hits).

It does NOT swallow API errors — callers keep their own try/except + retry/fallback,
so behaviour on failure is unchanged.
"""
from __future__ import annotations

import logging

from newsroom.logsetup import bind

log = logging.getLogger("newsroom.llm")


def _cached_tokens(usage) -> int | None:
    """OpenAI reports prompt-cache hits under usage.prompt_tokens_details.cached_tokens
    (absent on older responses/models) — read it defensively."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    return getattr(details, "cached_tokens", None)


def log_usage(op: str, model: str, resp, *, event_id: int | None = None) -> None:
    """Journal one call's token usage to the 'newsroom.llm' logger (structured)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    log.info("llm call", extra=bind(
        op=op, model=model, event_id=event_id,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        cached_tokens=_cached_tokens(usage),
    ))


def chat_json(client, *, model: str, messages: list, op: str, max_tokens: int,
              temperature: float = 0.0, event_id: int | None = None) -> str | None:
    """Call chat.completions with a JSON response format, a hard output cap and usage
    logging. Returns the message content (str) or None. Raises on API error so the
    caller's existing try/except handles it (retry / conservative fallback)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    log_usage(op, model, resp, event_id=event_id)
    return resp.choices[0].message.content
