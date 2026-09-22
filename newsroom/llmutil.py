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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

from newsroom.logsetup import bind

log = logging.getLogger("newsroom.llm")

# How much of the request/response text to store (the "essence"). Capped so the table
# does not grow unbounded; lower it, or prune (newsroom.maintenance.prune_llm_calls), to
# trade detail for size.
MAX_LOG_TEXT = 4000

# USD price per 1,000,000 tokens. input = uncached prompt tokens, cached_input = prompt-cache
# hits (discounted), output = completion tokens (embeddings have no output). Unknown model ->
# cost 0 (tokens still stored, so cost can be recomputed). Verified against OpenAI's pricing
# page on 2026-09-21: gpt-4o-mini and gpt-4o unchanged; the newer cheap models are added so a
# model swap still costs correctly. Embeddings kept at their long-stable rate (the current
# page no longer surfaces an embeddings table). EDIT when pricing or the model changes.
PRICES: dict[str, dict[str, float]] = {
    # what the pipeline uses today (gpt-4o-mini for every LLM op, small for embeddings)
    "gpt-4o-mini":            {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "cached_input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "cached_input": 0.13, "output": 0.0},
    # priced so switching the model still computes cost. gpt-5-nano is now cheaper AND newer
    # than gpt-4o-mini ($0.05/$0.40 vs $0.15/$0.60) — a candidate swap for the cheap ops.
    "gpt-4o":                 {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4.1-mini":           {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano":           {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5-mini":             {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano":             {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.6-luna":           {"input": 0.20, "cached_input": 0.02, "output": 1.20},
}


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": required or list(properties), "additionalProperties": False}


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_TEXT = {"type": "string"}
_NUMBER_OR_NULL = _nullable({"type": "number"})
_TEXT_OR_NULL = _nullable(_TEXT)

# One strict contract per production operation. Parsers remain as a defensive boundary
# for provider fallbacks and historical/local-model compatibility, but OpenAI calls no
# longer spend a retry merely because the JSON shape drifted.
JSON_SCHEMAS: dict[str, dict] = {
    "classify": _object({
        "is_event": {"type": "boolean"},
        "rubrics": {"type": "array", "items": _TEXT},
        "side": {"type": "string", "enum": ["ua", "ru", "unknown"]},
        "is_first_source": {"type": "boolean"},
        "is_rumor": {"type": "boolean"},
        "keywords": {"type": "array", "items": _TEXT},
        "facts": {"type": "array", "items": _object({
            "text": _TEXT,
            "modality": {"type": "string", "enum": ["fact", "statement", "forecast"]},
            "attribution": _TEXT_OR_NULL,
            "time_frame": _TEXT_OR_NULL,
            "number": _NUMBER_OR_NULL,
            "unit": _TEXT_OR_NULL,
        })},
        "topic_path": {"type": "array", "items": _TEXT},
        "facets": {"type": "array", "items": _object({
            "dimension": {"type": "string", "enum": [
                "event_type", "geography", "actor", "target", "entity", "sector",
                "impact", "means", "audience_scope", "story",
            ]},
            "path": {"type": "array", "items": _TEXT},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": _TEXT,
        })},
    }),
    "factbase": _object({
        "facts": {"type": "array", "items": _object({
            "text": _TEXT,
            "kind": {"type": "string", "enum": ["fact", "claim", "reaction", "frame"]},
            "number": _NUMBER_OR_NULL,
            "unit": _TEXT_OR_NULL,
            "modality": {"type": "string", "enum": ["fact", "statement", "forecast"]},
            "attribution": _TEXT_OR_NULL,
            "time_frame": _TEXT_OR_NULL,
        })},
    }),
    "factcheck_claims": _object({
        "claims": {"type": "array", "items": _object({
            "text": _TEXT,
            "claim_type": {"type": "string", "enum": [
                "who", "what", "where", "when", "number", "quote", "cause",
            ]},
        })},
    }),
    "factcheck_verdict": _object({
        "verdict": {"type": "string", "enum": ["true", "false", "misleading", "unverifiable"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": _TEXT,
        "stances": {"type": "array", "items": {
            "type": "string", "enum": ["supports", "refutes", "neutral"]}},
    }),
    "story_update": _object({
        "update_type": {"type": "string", "enum": [
            "new_fact", "confirmation", "refutation", "reaction", "consequence", "minor"]},
        "significant": {"type": "boolean"},
        "position_changed": {"type": "boolean"},
        "summary": _TEXT,
    }),
    "twin": _object({
        "decision": {"type": "string", "enum": ["duplicate", "update", "separate"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": _TEXT,
    }),
    "generate": _object({
        "headline": _TEXT, "body": _TEXT, "watching": _TEXT,
        "rubrics": {"type": "array", "items": _TEXT},
    }),
    "curate": _object({
        "decisions": {"type": "array", "items": _object({
            "id": {"type": "integer"},
            "decision": {"type": "string", "enum": ["publish", "hold"]},
        })},
    }),
    "dedup": _object({
        "groups": {"type": "array", "items": {
            "type": "array", "items": {"type": "integer"}}},
    }),
    "taxonomy_merge": _object({
        "groups": {"type": "array", "items": {
            "type": "array", "items": {"type": "integer"}}},
    }),
    "moderate_image": _object({
        "blocked": {"type": "boolean"},
        "labels": {"type": "array", "items": _TEXT},
        "reason": _TEXT,
    }),
}


def response_format_for(op: str) -> dict:
    schema = JSON_SCHEMAS.get(op)
    if schema is None:
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {
        "name": f"newsroom_{op}", "strict": True, "schema": schema,
    }}


def _format_not_supported(exc: Exception) -> bool:
    message = str(exc).lower()
    return ("response_format" in message or "json_schema" in message) and any(
        marker in message for marker in ("unsupported", "not support", "invalid", "unknown")
    )


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
    related_event_id: int | None = None
    context: dict | None = None
    duration_ms: int | None = None
    request_text: str | None = None
    response_text: str | None = None


# Pluggable sink (set at startup to the DB recorder). None = record nowhere (tests, tools).
_recorder: Callable[[UsageRecord], None] | None = None
_call_context: ContextVar[dict] = ContextVar("newsroom_llm_call_context", default={})


@contextmanager
def llm_context(*, event_id: int | None = None, related_event_id: int | None = None,
                **details) -> Iterator[None]:
    """Attach business context to every nested LLM/embedding call.

    Keeping this at the orchestration boundary avoids changing every pluggable
    classifier/extractor interface merely for telemetry. Nested scopes inherit and
    override keys, and ContextVar keeps concurrent asyncio tasks isolated.
    """
    merged = dict(_call_context.get())
    if event_id is not None:
        merged["event_id"] = int(event_id)
    if related_event_id is not None:
        merged["related_event_id"] = int(related_event_id)
    merged.update({k: v for k, v in details.items() if v is not None})
    token = _call_context.set(merged)
    try:
        yield
    finally:
        _call_context.reset(token)


def current_llm_context() -> dict:
    return dict(_call_context.get())


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
    value = "\n".join(parts)
    if len(value) <= cap:
        return value
    # Prompts put instructions first and the actual news/pair at the end. Keeping only
    # the prefix made the telemetry useless for production evals, so retain both ends.
    marker = "\n…[truncated]…\n"
    if cap <= len(marker):
        return value[:cap]
    head = max(0, (cap - len(marker)) // 2)
    tail = max(0, cap - len(marker) - head)
    return value[:head] + marker + (value[-tail:] if tail else "")


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
                      related_event_id: int | None = None, context: dict | None = None,
                      duration_ms: int | None = None, content: str | None = None) -> None:
    """Build a UsageRecord for a chat completion and hand it to the recorder."""
    usage = getattr(resp, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cached = int(_cached_tokens(usage) or 0) if usage else 0
    inherited = current_llm_context()
    resolved_event_id = event_id if event_id is not None else inherited.pop("event_id", None)
    resolved_related_id = (related_event_id if related_event_id is not None
                           else inherited.pop("related_event_id", None))
    merged_context = {**inherited, **(context or {})}
    record_usage(UsageRecord(
        op=op, model=model, prompt_tokens=prompt, completion_tokens=completion,
        cached_tokens=cached, cost_usd=cost_for(model, prompt, completion, cached),
        event_id=resolved_event_id, related_event_id=resolved_related_id,
        context=merged_context or None, duration_ms=duration_ms,
        request_text=messages_text(messages), response_text=(content or "")[:MAX_LOG_TEXT] or None))


def chat_json(client, *, model: str, messages: list, op: str, max_tokens: int,
              temperature: float = 0.0, event_id: int | None = None,
              related_event_id: int | None = None, context: dict | None = None) -> str | None:
    """Call chat.completions with a JSON response format, a hard output cap, usage logging
    and cost recording. Returns the message content (str) or None. Raises on API error so
    the caller's existing try/except handles it (retry / conservative fallback)."""
    t0 = time.monotonic()
    kwargs = dict(model=model, messages=messages, response_format=response_format_for(op),
                  temperature=temperature, max_tokens=max_tokens)
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Local/OpenAI-compatible endpoints may implement JSON mode but not schemas.
        # Fall back only for an explicit capability error; network/rate/auth errors
        # still propagate to the caller's normal retry policy.
        if kwargs["response_format"]["type"] != "json_schema" or not _format_not_supported(exc):
            raise
        log.warning("structured output unsupported; falling back to JSON mode",
                    extra=bind(op=op, model=model))
        kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
    duration_ms = int((time.monotonic() - t0) * 1000)
    inherited = current_llm_context()
    resolved_event_id = event_id if event_id is not None else inherited.get("event_id")
    log_usage(op, model, resp, event_id=resolved_event_id)
    content = resp.choices[0].message.content
    record_completion(op, model, resp, messages=messages, event_id=event_id,
                      related_event_id=related_event_id, context=context,
                      duration_ms=duration_ms, content=content)
    return content
