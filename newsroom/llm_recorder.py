"""Persist LLM cost telemetry: turn each llmutil.UsageRecord into an `llm_calls` row.

Wired at startup with `set_usage_recorder(make_db_recorder(session_factory))` so every
completion and embedding lands in the DB for cost analysis. Best-effort by contract
(llmutil.record_usage catches), and cheap — one INSERT per call.
"""
from __future__ import annotations

from typing import Callable

from newsroom.llmutil import MAX_LOG_TEXT, UsageRecord


def make_db_recorder(session_factory, *, max_text: int = MAX_LOG_TEXT) -> Callable[[UsageRecord], None]:
    """A recorder that inserts one `llm_calls` row per call. Text is capped to max_text."""

    def record(rec: UsageRecord) -> None:
        from newsroom.models import LlmCall

        with session_factory() as s:
            s.add(LlmCall(
                op=(rec.op or "")[:32],
                model=(rec.model or "")[:64],
                prompt_tokens=int(rec.prompt_tokens or 0),
                completion_tokens=int(rec.completion_tokens or 0),
                cached_tokens=int(rec.cached_tokens or 0),
                cost_usd=float(rec.cost_usd or 0.0),
                event_id=rec.event_id,
                duration_ms=rec.duration_ms,
                request_text=((rec.request_text or "")[:max_text] or None),
                response_text=((rec.response_text or "")[:max_text] or None),
            ))
            s.commit()

    return record


def summarize_cost(session_factory, *, days: int = 7) -> dict:
    """Aggregate llm_calls over the last `days`: totals plus breakdowns by op, by model and
    by day. The data an analyst needs to see where spend goes. Read-only."""
    import datetime as dt

    from sqlalchemy import func, literal_column, select

    from newsroom.models import LlmCall

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    tokens = LlmCall.prompt_tokens + LlmCall.completion_tokens
    where = LlmCall.created_at >= cutoff
    # inline the 'day' literal so SELECT and GROUP BY use the SAME expression text (a bound
    # parameter makes Postgres treat them as different columns -> GroupingError).
    day = func.date_trunc(literal_column("'day'"), LlmCall.created_at)
    with session_factory() as s:
        calls, cost, toks = s.execute(select(
            func.count(LlmCall.id), func.coalesce(func.sum(LlmCall.cost_usd), 0.0),
            func.coalesce(func.sum(tokens), 0)).where(where)).one()
        by_op = s.execute(select(
            LlmCall.op, func.count(LlmCall.id), func.coalesce(func.sum(LlmCall.cost_usd), 0.0),
            func.coalesce(func.sum(tokens), 0)).where(where)
            .group_by(LlmCall.op).order_by(func.sum(LlmCall.cost_usd).desc())).all()
        by_model = s.execute(select(
            LlmCall.model, func.count(LlmCall.id), func.coalesce(func.sum(LlmCall.cost_usd), 0.0))
            .where(where).group_by(LlmCall.model).order_by(func.sum(LlmCall.cost_usd).desc())).all()
        by_day = s.execute(select(
            day, func.coalesce(func.sum(LlmCall.cost_usd), 0.0), func.count(LlmCall.id))
            .where(where).group_by(day).order_by(day.desc())).all()
    return {
        "days": days, "calls": int(calls), "cost": float(cost), "tokens": int(toks),
        "by_op": [(o, int(c), float(x), int(t)) for o, c, x, t in by_op],
        "by_model": [(m, int(c), float(x)) for m, c, x in by_model],
        "by_day": [(d.date().isoformat() if d else "?", float(x), int(c)) for d, x, c in by_day],
    }
