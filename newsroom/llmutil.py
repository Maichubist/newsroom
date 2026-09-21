"""Shared OpenAI chat wrapper — one place to cap output, log token usage AND record
the USD cost of every call for analysis.

Every LLM JSON call goes through `chat_json`, so this is the single seam where we:
  (a) bound output tokens per operation (a runaway response can't cost unboundedly);
  (b) journal model + prompt/completion/cached tokens to the log; and
  (c) record one `llm_calls` row per call — op, model, tokens, computed cost and the
      (capped) request/response text — via a pluggable recorder set at startup. The DB
      write is best-effort: a recorder failure never breaks the API call.

Embeddings record through the same recorder (op="embed"), so the cost table covers
completions AND embeddings. It does NOT swallow API errors — callers keep their own
try/except + retry/fallback, so behaviour on failure is unchanged.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from newsroom.logsetup import bind

log = logging.getLogger("newsroom.llm")

# How much of the request/response text to store (the "essence"). Capped so the table
# does not grow unbounded; lower it, or prune (newsroom.maintenance.prune_llm_calls), to
# trade detail for size.
MAX_LOG_TEXT = 4000

# USD price per 1,000,000 tokens. APPROXIMATE — edit to match the current OpenAI pricing.
# input = uncached prompt tokens, cached_input = prompt-cache hits (discounted), output =
# completion tokens (embeddings have no output). Unknown model -> cost 0 (tokens still
# stored, so cost can be recomputed later).
PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini":            {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o":                 {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "cached_input": 0.13, "output": 0.0},
}


def cost_for(model: str, prompt_tokens: int = 0, completion_tokens: int = 0,
             cached_tokens: int = 0) -> float:
    """USD cost of one call. prompt_tokens INCLUDES cached_tokens (OpenAI convention), so
    the cached share is billed at the cheaper cached rate. 0.0 for an unpriced model."""
    p = PRICES.get(model)
    if not p:
        return 0.0
    cached = min(max(cached_tokens or 0, 0), max(prompt_tokens or 0, 0))
    uncached = max(prompt_tokens or 0, 0) - cached
    return round(
        uncached * p["input"] / 1_000_000
        + cached * p.get("cached_input", p["input"]) / 1_000_000
        + max(completion_tokens or 0, 0) * p["output"] / 1_000_000,
        8,
    )


@dataclass(frozen=True)
class UsageRecord:
    op: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    event_id: int | None = None
    duration_ms: int | None = None
    request_text: str | None = None
    response_text: str | None = None


# Pluggable sink (set at startup to the DB recorder). None = record nowhere (tests, tools).
_recorder: Callable[[UsageRecord], None] | None = None


def set_usage_recorder(fn: Callable[[UsageRecord], None] | None) -> None:
    """Install the sink that persists each UsageRecord (e.g. the DB recorder). None off."""
    global _recorder
    _recorder = fn


def record_usage(record: UsageRecord) -> None:
    """Hand a record to the sink, if any. Best-effort — telemetry never breaks a call."""
    fn = _recorder
    if fn is None:
        return
    try:
        fn(record)
    except Exception:  # noqa: BLE001 - a logging failure must not break the pipeline
        log.exception("llm usage recorder failed", extra=bind(op=record.op, model=record.model))


def _cached_tokens(usage) -> int | None:
    """OpenAI reports prompt-cache hits under usage.prompt_tokens_details.cached_tokens
    (absent on older responses/models) — read it defensively."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    return getattr(details, "cached_tokens", None)


def messages_text(messages, *, cap: int = MAX_LOG_TEXT) -> str:
    """Flatten chat messages into 'role: content' lines for the request-text column."""
    parts: list[str] = []
    for m in messages or []:
        if isinstance(m, dict):
            role, content = str(m.get("role") or ""), str(m.get("content") or "")
            parts.append(f"{role}: {content}" if role else content)
        else:
            parts.append(str(m))
    return "\n".join(parts)[:cap]


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


def record_completion(op: str, model: str, resp, *, messages, event_id: int | None = None,
                      duration_ms: int | None = None, content: str | None = None) -> None:
    """Build a UsageRecord for a chat completion and hand it to the recorder."""
    usage = getattr(resp, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cached = int(_cached_tokens(usage) or 0) if usage else 0
    record_usage(UsageRecord(
        op=op, model=model, prompt_tokens=prompt, completion_tokens=completion,
        cached_tokens=cached, cost_usd=cost_for(model, prompt, completion, cached),
        event_id=event_id, duration_ms=duration_ms,
        request_text=messages_text(messages), response_text=(content or "")[:MAX_LOG_TEXT] or None))


def chat_json(client, *, model: str, messages: list, op: str, max_tokens: int,
              temperature: float = 0.0, event_id: int | None = None) -> str | None:
    """Call chat.completions with a JSON response format, a hard output cap, usage logging
    and cost recording. Returns the message content (str) or None. Raises on API error so
    the caller's existing try/except handles it (retry / conservative fallback)."""
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    duration_ms = int((time.monotonic() - t0) * 1000)
    log_usage(op, model, resp, event_id=event_id)
    content = resp.choices[0].message.content
    record_completion(op, model, resp, messages=messages, event_id=event_id,
                      duration_ms=duration_ms, content=content)
    return content
