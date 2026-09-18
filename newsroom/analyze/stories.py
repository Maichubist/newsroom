"""Story assignment (architecture §7).

A separate step above event clustering: an event is linked to a long-running
story by vector similarity in a longer window (entity overlap is a later
refinement once NER populates entities). Deterministic core — lifecycle, story
versions, hashtag — here; the narrative summary and update_type (which decide
whether a new post is warranted) are the editor's job (1в.4).

Lifecycle: new -> developing -> stable -> dormant (N days idle) -> closed.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass

from newsroom.analyze.clustering import best_match, update_centroid

log = logging.getLogger("newsroom.analyze.stories")

# Looser than event clustering (0.83, near-identical): stories group the SAME
# narrative across sources/wording over a long window. Calibrated on real data —
# cross-source duplicates sit at cosine ~0.65-0.70, while distinct same-theme events
# (e.g. drone strikes on different cities) sit at ~0.53-0.59, so ~0.62 separates
# them. Narrow margin on a small sample — tunable via STORY_THRESHOLD; the robust
# long-term fix is entity/LLM dedup (embeddings alone compress topical similarity).
DEFAULT_STORY_THRESHOLD = 0.62
DEFAULT_STORY_WINDOW_HOURS = 24 * 14    # 14 days
HASHTAG_MIN_EVENTS = 3                   # charter §7: hashtag only after 3+ updates
DEFAULT_DORMANT_DAYS = 5

_SLUG_RE = re.compile(r"[^0-9a-zа-яіїєґ']+")


def slugify(text: str, *, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return s[:max_len] or "story"


@dataclass(frozen=True)
class StoryAssignResult:
    story_id: int
    created_new: bool
    similarity: float
    version: int


class StoryLinker:
    def __init__(self, session_factory, *, threshold: float = DEFAULT_STORY_THRESHOLD,
                 window_hours: int = DEFAULT_STORY_WINDOW_HOURS):
        self.sf = session_factory
        self.threshold = threshold
        self.window_hours = window_hours

    def assign(self, event_id: int) -> StoryAssignResult:
        from sqlalchemy import func, select

        from newsroom.models import Event, Story, StoryVersion

        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(hours=self.window_hours)

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is None or event.centroid is None:
                raise ValueError(f"event {event_id} missing or has no centroid")
            centroid = event.centroid

            stories = list(s.execute(
                select(Story).where(
                    Story.centroid.is_not(None),
                    Story.last_event_at >= cutoff,
                    Story.state != "closed",
                )
            ).scalars().all())

            idx, sim = best_match(centroid, [st.centroid for st in stories], self.threshold)

            if idx is not None:
                story = stories[idx]
                n = int(s.scalar(select(func.count()).select_from(Event).where(Event.story_id == story.id)) or 0)
                story.centroid = update_centroid(story.centroid, n, centroid)
                story.last_event_at = now
                story.state = "developing"          # a fresh event revives a dormant story too
                event.story_id = story.id
                total = n + 1
                if total >= HASHTAG_MIN_EVENTS and not story.hashtag:
                    story.hashtag = "#" + (story.rubric or event.rubric or "сюжет")
                version = int(s.scalar(select(func.max(StoryVersion.version)).where(StoryVersion.story_id == story.id)) or 0) + 1
                s.add(StoryVersion(story_id=story.id, version=version, reason_event_id=event_id))
                s.commit()
                return StoryAssignResult(story.id, False, float(sim), version)

            story = Story(
                slug="pending",
                title=(event.title or "Сюжет"),
                rubric=event.rubric,
                centroid=centroid,
                state="new",
                last_event_at=now,
            )
            s.add(story)
            s.flush()
            story.slug = f"{slugify(event.title or 'story')}-{story.id}"
            event.story_id = story.id
            s.add(StoryVersion(story_id=story.id, version=1, reason_event_id=event_id))
            s.commit()
            return StoryAssignResult(story.id, True, float(sim), 1)


def link_pending(session_factory, linker: "StoryLinker", *, limit: int = 50) -> dict[str, int]:
    """One story-linking tick (architecture §7): attach every clustered event that
    has a centroid but no story yet to a story — an existing one by vector
    similarity in the 14-day window, or a new one. This is the step that lets
    near-identical events share a story (so the update classifier can mark the
    repeats summary-only instead of posting each) and lets story posts reply-chain.
    A poison event is skipped, not allowed to block the queue."""
    from sqlalchemy import select

    from newsroom.models import Event

    with session_factory() as s:
        ids = list(s.execute(
            select(Event.id)
            # a merged duplicate (ingest dedup) is inert — don't link it to a story
            .where(Event.story_id.is_(None), Event.centroid.is_not(None),
                   Event.duplicate_of.is_(None))
            .order_by(Event.id)
            .limit(limit)
        ).scalars().all())

    stats = {"linked": 0, "new_stories": 0, "errors": 0}
    for event_id in ids:
        try:
            result = linker.assign(event_id)
        except Exception:  # noqa: BLE001 - one bad event must not stall linking
            log.exception("story link failed", extra={"event_id": event_id})
            stats["errors"] += 1
            continue
        stats["linked"] += 1
        if result.created_new:
            stats["new_stories"] += 1
    return stats


def mark_dormant(session_factory, *, dormant_days: int = DEFAULT_DORMANT_DAYS) -> int:
    """Move idle stories to 'dormant'. Returns how many were changed."""
    from sqlalchemy import select

    from newsroom.models import Story

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=dormant_days)
    changed = 0
    with session_factory() as s:
        stories = s.execute(
            select(Story).where(
                Story.state.in_(("new", "developing", "stable")),
                Story.last_event_at.is_not(None),
                Story.last_event_at < cutoff,
            )
        ).scalars().all()
        for story in stories:
            story.state = "dormant"
            changed += 1
        s.commit()
    return changed
