"""Editorial pipeline: turn a verified event into a DRAFT publication.

generate -> compose -> critic -> (regenerate once if the critic complains) ->
store a publications row with status='draft'. Nothing is published — the channel
adapter and the publish gate are stage 1г. A hard critic failure is still stored
as a draft but flagged (features.critic_ok=False) so the publish gate refuses it.

Generator is pluggable (LLM in prod, fake in tests); the critic is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

from newsroom.editorial.critic import critic_check
from newsroom.editorial.draft import compose_post, content_is_publishable
from newsroom.editorial.generator import GenerationContext, Generator


@dataclass(frozen=True)
class ProduceResult:
    publication_id: int | None
    critic_ok: bool
    hard: list[str]
    soft: list[str]
    regenerated: bool


class EditorialPipeline:
    def __init__(
        self,
        session_factory,
        *,
        generator: Generator,
        stoplist_rules,
        ai_accent_patterns,
        charter_version: str = "0.2",
        prompt_version: str = "0.2",
    ):
        self.sf = session_factory
        self.generator = generator
        self.stoplist_rules = stoplist_rules
        self.ai_accent_patterns = ai_accent_patterns
        self.charter_version = charter_version
        self.prompt_version = prompt_version

    def produce(self, event_id: int) -> ProduceResult:
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item, Source, Story

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is None:
                return ProduceResult(None, False, ["event_not_found"], [], False)
            story = s.get(Story, event.story_id) if event.story_id else None
            summary = (story.current_summary if story else None) or ""
            rubrics = [event.rubric] if event.rubric else []
            status = event.status or "confirmed"
            hashtags = _hashtags(event.rubric, story.hashtag if story else None)
            source_links = _source_links(s, event_id)
            sources = [name for name, _ in source_links]
            title = event.title or ""
            risk_level = event.risk_level
            facts = _facts_from_base(event.fact_base)
            source_excerpt = _source_excerpt(s, event_id)

        is_rumor = status == "rumor"
        # show "Повідомляють:" only where the confidence level matters — a
        # single-source high/critical item — not on routine reported news.
        reported = status == "reported" and risk_level in ("high", "critical")
        ctx = GenerationContext(title=title, summary=summary, rubrics=rubrics, status=status,
                                facts=facts, source_excerpt=source_excerpt)

        # A generation that failed (fallback) or produced no real content must not
        # become a post: leave the event undrafted so the next tick retries it once
        # the model recovers (e.g. after a 429). Better a delay than a placeholder.
        draft = self._generate_publishable(ctx)
        if draft is None:
            self._journal_generation_failed(event_id)
            return ProduceResult(None, False, ["generation_failed"], [], False)

        post, report = self._compose_and_check(draft, reported, is_rumor, hashtags, sources)

        regenerated = False
        if not report.ok or report.soft:
            feedback = "; ".join(report.hard + report.soft)
            retry = self.generator.generate(ctx, feedback=feedback)
            if content_is_publishable(retry):        # keep the good first draft if the retry failed
                draft = retry
                post, report = self._compose_and_check(draft, reported, is_rumor, hashtags, sources)
            regenerated = True

        # Telegram-specific presentation is rendered at send time from these pieces
        # (a bold headline, linked sources); pub.body stays plain for the critic,
        # the shadow report and other channels.
        render = {
            "headline": draft.headline,
            "body": draft.body,
            "watching": draft.watching,
            "hashtags": hashtags,
            "source_links": [[name, url] for name, url in source_links],
            "is_rumor": is_rumor,
            "reported": reported,
        }
        pub_id = self._store_draft(event_id, draft.headline, post, status, is_rumor,
                                   rubrics, hashtags, report, render)
        return ProduceResult(pub_id, report.ok, report.hard, report.soft, regenerated)

    # ------------------------------------------------------------------
    def _generate_publishable(self, ctx):
        """Generate a draft, retrying once if the first attempt is a fallback or
        thin (a transient model failure). Returns None if both attempts are
        unusable, so the caller can hold the event and retry next tick."""
        for _ in (1, 2):
            draft = self.generator.generate(ctx)
            if content_is_publishable(draft):
                return draft
        return None

    def _journal_generation_failed(self, event_id: int) -> None:
        from newsroom.models import Decision

        with self.sf() as s:
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="edit",
                decision="generation_failed", reason="no publishable draft",
                details={}, charter_version=self.charter_version,
                prompt_version=self.prompt_version, model=getattr(self.generator, "model", None),
            ))
            s.commit()

    def _compose_and_check(self, draft, reported, is_rumor, hashtags, sources):
        post = compose_post(draft, is_rumor=is_rumor, reported=reported, hashtags=hashtags, sources=sources)
        report = critic_check(post, is_rumor=is_rumor, stoplist_rules=self.stoplist_rules,
                              ai_accent_patterns=self.ai_accent_patterns)
        return post, report

    def _store_draft(self, event_id, headline, body, status, is_rumor, rubrics, hashtags, report,
                     render=None) -> int:
        from newsroom.models import Decision, Publication

        features = {
            "status": status,
            "is_rumor": is_rumor,
            "rubrics": rubrics,
            "hashtags": hashtags,
            "length": len(body),
            "critic_ok": report.ok,
            "critic_hard": report.hard,
            "critic_soft": report.soft,
            "render": render,
        }
        with self.sf() as s:
            pub = Publication(
                event_id=event_id, channel="telegram", kind="post",
                headline=headline, body=body, status="draft",
                charter_version=self.charter_version, prompt_version=self.prompt_version,
                model=getattr(self.generator, "model", None), features=features,
            )
            s.add(pub)
            s.flush()
            pub_id = pub.id
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="edit",
                decision="draft_ok" if report.ok else "draft_flagged",
                reason=None if report.ok else ",".join(report.hard),
                details=features, charter_version=self.charter_version,
                prompt_version=self.prompt_version, model=getattr(self.generator, "model", None),
            ))
            s.commit()
        return pub_id


def produce_drafts(session_factory, pipeline: "EditorialPipeline", *, limit: int = 25,
                   classify_grace_seconds: float = 180.0) -> dict[str, int]:
    """One editorial tick: draft posts for publishable events that don't have a
    publication yet. Events the story-update step classified as summary-only
    (confirmation / reaction / minor) are skipped — they only update the story
    summary, not post (architecture §7).

    Draftable: a postworthy classification (new_fact / refutation / consequence),
    OR an unclassified event that is either unlinked (no story to dedup against)
    or has waited out `classify_grace_seconds` — so a story-linked event gets a
    window for story-updates to classify it (and mark a near-duplicate summary-only)
    before it is drafted, without stalling forever if story-updates is off.
    Once produced, the event has a publications row and is skipped next tick.
    Nothing is published."""
    import datetime as dt

    from sqlalchemy import and_, or_, select

    from newsroom.editorial.updates import SUMMARY_ONLY_UPDATE_TYPES, VALID_UPDATE_TYPES
    from newsroom.models import Event, Publication

    postworthy = tuple(VALID_UPDATE_TYPES - SUMMARY_ONLY_UPDATE_TYPES)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=classify_grace_seconds)

    with session_factory() as s:
        have_pub = select(Publication.event_id).where(Publication.event_id.is_not(None))
        ids = list(s.execute(
            select(Event.id)
            .where(
                Event.status.in_(("reported", "confirmed", "rumor")),
                Event.id.not_in(have_pub),
                or_(
                    Event.update_type.in_(postworthy),
                    and_(
                        Event.update_type.is_(None),
                        or_(Event.story_id.is_(None), Event.updated_at < cutoff),
                    ),
                ),
            )
            .order_by(Event.id)
            .limit(limit)
        ).scalars().all())

    stats = {"produced": 0, "ok": 0, "flagged": 0, "held": 0}
    for event_id in ids:
        result = pipeline.produce(event_id)
        if result.publication_id is None:
            stats["held"] += 1          # generation failed; no draft stored, retried next tick
            continue
        stats["produced"] += 1
        stats["ok" if result.critic_ok else "flagged"] += 1
    return stats


def _facts_from_base(fact_base, *, limit: int = 12) -> list[str]:
    """Pull the shared facts out of events.fact_base for the generator, confirmed
    facts first, marking where sources disagree on numbers (§8)."""
    if not isinstance(fact_base, dict):
        return []
    rows = [f for f in (fact_base.get("facts") or []) if isinstance(f, dict) and f.get("text")]
    rows.sort(key=lambda f: int(f.get("confirmed_by") or 0), reverse=True)
    out: list[str] = []
    for f in rows[:limit]:
        text = str(f["text"]).strip()
        if f.get("divergent"):
            text += " (джерела розходяться в цифрах)"
        out.append(text)
    return out


def _source_excerpt(s, event_id: int, *, max_chars: int = 2000) -> str:
    """A trimmed concatenation of the event's source material, so the generator has
    concrete detail to write from when the fact base is thin."""
    from sqlalchemy import select

    from newsroom.models import EventItem, Item

    rows = s.execute(
        select(Item.title, Item.text)
        .join(EventItem, EventItem.item_id == Item.id)
        .where(EventItem.event_id == event_id)
        .limit(6)
    ).all()
    chunks: list[str] = []
    for title, text in rows:
        piece = f"{(title or '').strip()}\n{(text or '').strip()}".strip()
        if piece:
            chunks.append(piece)
    return "\n\n".join(chunks)[:max_chars].strip()


def _source_links(s, event_id: int) -> list[tuple[str, str | None]]:
    """Each distinct source name for the event, paired with one article URL (the
    first non-null item.url from that source), so the post can link the source
    name instead of printing a "Джерела:" line. Preserves first-seen order."""
    from sqlalchemy import select

    from newsroom.models import EventItem, Item, Source

    rows = s.execute(
        select(Source.name, Item.url)
        .join(Item, Item.source_id == Source.id)
        .join(EventItem, EventItem.item_id == Item.id)
        .where(EventItem.event_id == event_id)
    ).all()
    link_by_name: dict[str, str | None] = {}
    for name, url in rows:
        if not name:
            continue
        if name not in link_by_name or (link_by_name[name] is None and url):
            link_by_name[name] = url
    return list(link_by_name.items())


def _hashtags(rubric: str | None, story_hashtag: str | None) -> list[str]:
    tags: list[str] = []
    if rubric:
        tags.append("#" + rubric)
    if story_hashtag:
        tags.append(story_hashtag)
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:2]
