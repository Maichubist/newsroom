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
from newsroom.editorial.draft import compose_post
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
            sources = list(s.execute(
                select(Source.name)
                .join(Item, Item.source_id == Source.id)
                .join(EventItem, EventItem.item_id == Item.id)
                .where(EventItem.event_id == event_id)
                .distinct()
            ).scalars().all())
            title = event.title or ""

        is_rumor = status == "rumor"
        ctx = GenerationContext(title=title, summary=summary, rubrics=rubrics, status=status)

        draft = self.generator.generate(ctx)
        post, report = self._compose_and_check(draft, status, is_rumor, hashtags, sources)

        regenerated = False
        if not report.ok or report.soft:
            feedback = "; ".join(report.hard + report.soft)
            draft = self.generator.generate(ctx, feedback=feedback)
            post, report = self._compose_and_check(draft, status, is_rumor, hashtags, sources)
            regenerated = True

        pub_id = self._store_draft(event_id, draft.headline, post, status, is_rumor,
                                   rubrics, hashtags, report)
        return ProduceResult(pub_id, report.ok, report.hard, report.soft, regenerated)

    # ------------------------------------------------------------------
    def _compose_and_check(self, draft, status, is_rumor, hashtags, sources):
        post = compose_post(draft, status=status, is_rumor=is_rumor, hashtags=hashtags, sources=sources)
        report = critic_check(post, is_rumor=is_rumor, stoplist_rules=self.stoplist_rules,
                              ai_accent_patterns=self.ai_accent_patterns)
        return post, report

    def _store_draft(self, event_id, headline, body, status, is_rumor, rubrics, hashtags, report) -> int:
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
