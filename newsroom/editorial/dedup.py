"""LLM batch dedup (docs/demand-intelligence.md follow-up; charter: dedup is vector,
not string — but embeddings compress topical similarity, so a comparative LLM pass
catches the same-story-different-wording duplicates the 0.62 story threshold misses).

A tick takes the recent window of events and asks the model to GROUP those that report
the SAME underlying story (not merely the same theme — two drone strikes on different
cities are different stories). For each group the earliest event is the canonical; the
others get events.duplicate_of set and are then skipped by curation/editorial, so only
one post goes out. Comparison-in-a-batch, not an absolute score (CLAUDE.md).

Parsing is pure/offline-tested; the grouper is pluggable (LLM in prod, fake in tests);
the DB step marks duplicates and is pg-tested.
"""
from __future__ import annotations

import json
import logging
from typing import Protocol

from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.editorial.dedup")

POSTABLE_STATUSES = ("reported", "confirmed", "rumor")


def parse_groups(raw: str | None, valid_ids: set[int]) -> list[list[int]]:
    """Parse the model's JSON into groups of event ids. Keeps only known ids, drops
    singletons and empties. {"groups": [[1,2],[3]]} -> [[1,2]]. [] if unparsable."""
    if not raw:
        return []
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return []
    raw_groups = obj.get("groups") if isinstance(obj, dict) else obj
    if not isinstance(raw_groups, list):
        return []
    out: list[list[int]] = []
    for g in raw_groups:
        if not isinstance(g, list):
            continue
        ids: list[int] = []
        for x in g:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i in valid_ids and i not in ids:
                ids.append(i)
        if len(ids) >= 2:                 # a group of one is not a duplicate
            out.append(ids)
    return out


class DedupGrouper(Protocol):
    model: str

    def group(self, events: list[tuple[int, str]]) -> list[list[int]]: ...


DEFAULT_DEDUP_PROMPT = """Ти редактор. Нижче список новинних подій (id і заголовок).
Згрупуй ТІ, ЩО ПОВІДОМЛЯЮТЬ ПРО ОДНУ Й ТУ САМУ ПОДІЮ/новину (та сама подія, різні
джерела чи інше формулювання). НЕ групуй просто схожі за темою: два різні обстріли
різних міст, дві різні заяви — це РІЗНІ події.

Поверни лише JSON: {"groups": [[id, id, ...], ...]} — лише групи з 2+ дублів; події
без пари не включай.

ПОДІЇ:
{events}"""


def _render_events(events: list[tuple[int, str]]) -> str:
    return "\n".join(f"[{eid}] {(title or '').strip()}" for eid, title in events)


class LLMDedupGrouper:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_DEDUP_PROMPT):
        import os

        self.model = model
        self.prompt = prompt
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def group(self, events: list[tuple[int, str]]) -> list[list[int]]:
        valid = {eid for eid, _ in events}
        if len(valid) < 2:
            return []
        content = fill_prompt(self.prompt, events=_render_events(events))
        try:
            from newsroom.llmutil import chat_json

            raw = chat_json(self._ensure_client(), model=self.model,
                            messages=[{"role": "user", "content": content}],
                            op="dedup", max_tokens=1024)
            return parse_groups(raw, valid)
        except Exception as exc:  # noqa: BLE001
            log.warning("dedup group failed", extra={"error": str(exc)})
            return []           # conservative: no merges on failure


def dedup_pending(session_factory, grouper: "DedupGrouper", *, window_hours: int = 24,
                  limit: int = 80) -> dict[str, int]:
    """One dedup tick: group the recent window of postable events and mark
    non-canonical duplicates (events.duplicate_of), so only one of a duplicate set
    is drafted/published. Idempotent — re-marking the same duplicate is a no-op.

    Scans the MOST RECENT `limit` events (not the oldest): at scale there are far more
    than `limit` postable events in the window, and fresh cross-source duplicates
    cluster in time (the same story from N sources within minutes), so recency is
    where dedup pays off. `canonical = min(id)` still keeps the earliest as the kept one."""
    import datetime as dt

    from sqlalchemy import select

    from newsroom.models import Decision, Event

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    with session_factory() as s:
        rows = s.execute(
            select(Event.id, Event.title)
            .where(Event.status.in_(POSTABLE_STATUSES), Event.first_seen_at >= cutoff,
                   Event.title.is_not(None))
            .order_by(Event.first_seen_at.desc())
            .limit(limit)
        ).all()

    events = [(int(i), str(t)) for i, t in rows]
    if len(events) < 2:
        return {"events": len(events), "groups": 0, "duplicates": 0}

    groups = grouper.group(events)
    dups = 0
    merged = 0
    with session_factory() as s:
        for group in groups:
            canonical = min(group)          # earliest event is the canonical
            canon = s.get(Event, canonical)
            canon_story = canon.story_id if canon else None
            for eid in group:
                if eid == canonical:
                    continue
                ev = s.get(Event, eid)
                if ev is None or ev.duplicate_of is not None:
                    continue
                ev.duplicate_of = canonical
                # merge fragmented stories: the same real event split into different
                # story_ids (embeddings diverge on multi-facet news), which defeats the
                # per-story cooldown. Pulling the duplicate into the canonical's story
                # collapses the fragments so one-post-per-story works across sources.
                if canon_story is not None and ev.story_id != canon_story:
                    ev.story_id = canon_story
                    merged += 1
                s.add(Decision(entity_type="event", entity_id=str(eid), stage="edit",
                               decision="duplicate", reason=f"duplicate_of={canonical}",
                               details={"duplicate_of": canonical, "story_merged_into": canon_story},
                               model=getattr(grouper, "model", None)))
                dups += 1
        s.commit()
    return {"events": len(events), "groups": len(groups), "duplicates": dups, "stories_merged": merged}
