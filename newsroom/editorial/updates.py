"""Story updates: post or just update the summary? (architecture §7, §8.5).

When a new event joins a story, the editor compares it to the story's current
state and classifies how it moves the story (`update_type`). Only new_fact,
refutation and a *significant* consequence warrant a separate post; everything
else silently updates the story's "що відомо на зараз" (current_summary). Each
event's version in story_versions gets the summary as it stood after that event,
so the timeline stays consistent and a position change is visible (§8.5).

The routing rule and the update-type vocabulary are pure and offline-tested; the
classifier (which reads the fact base and writes the new summary) is a pluggable
LLM step. Every decision is journaled.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.editorial.updates")

# update_type vocabulary (stored in events.update_type) — architecture §5.2, §7
UPDATE_NEW_FACT = "new_fact"
UPDATE_CONFIRMATION = "confirmation"
UPDATE_REFUTATION = "refutation"
UPDATE_REACTION = "reaction"
UPDATE_CONSEQUENCE = "consequence"
UPDATE_MINOR = "minor"
VALID_UPDATE_TYPES = frozenset({
    UPDATE_NEW_FACT, UPDATE_CONFIRMATION, UPDATE_REFUTATION,
    UPDATE_REACTION, UPDATE_CONSEQUENCE, UPDATE_MINOR,
})

# routing outcomes
ROUTE_POST = "post"
ROUTE_SUMMARY = "summary_update"

# only these warrant a standalone post (consequence only when significant)
_ALWAYS_POST = frozenset({UPDATE_NEW_FACT, UPDATE_REFUTATION})

# events classified as one of these never warrant a standalone post — they only
# update the running summary. The editorial gate uses this coarse set (an
# unclassified event, or new_fact / refutation / consequence, is still drafted).
SUMMARY_ONLY_UPDATE_TYPES = frozenset({UPDATE_CONFIRMATION, UPDATE_REACTION, UPDATE_MINOR})


@dataclass(frozen=True)
class UpdateDecision:
    update_type: str = UPDATE_MINOR
    significant: bool = False
    summary: str = ""
    position_changed: bool = False


def route_update(update_type: str, significant: bool) -> str:
    """Architecture §7: new_fact / refutation always post; a consequence posts
    only when significant; confirmation / reaction / minor just update the
    summary. Unknown types are treated as minor by the parser, so they update."""
    if update_type in _ALWAYS_POST:
        return ROUTE_POST
    if update_type == UPDATE_CONSEQUENCE and significant:
        return ROUTE_POST
    return ROUTE_SUMMARY


def parse_update(raw: str | None) -> UpdateDecision | None:
    """Parse the model's JSON. Unknown update types collapse to 'minor' (so an
    uncertain event does not trigger a post). None if unparsable, so the caller
    can retry."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    update_type = str(obj.get("update_type") or "").strip().lower()
    if update_type not in VALID_UPDATE_TYPES:
        update_type = UPDATE_MINOR
    return UpdateDecision(
        update_type=update_type,
        significant=bool(obj.get("significant", False)),
        summary=str(obj.get("summary") or "").strip(),
        position_changed=bool(obj.get("position_changed", False)),
    )


class UpdateClassifier(Protocol):
    model: str

    def classify(self, current_summary: str | None, event_title: str | None,
                 fact_base: dict | None) -> UpdateDecision: ...


DEFAULT_UPDATE_PROMPT = """Ти головний редактор сюжету. Є поточний стан сюжету і нова
подія. Визнач, як подія змінює сюжет, і онови стислий підсумок «що відомо зараз».

update_type:
- new_fact — істотно новий факт;
- confirmation — підтвердження вже відомого;
- refutation — спростування;
- reaction — реакція/коментар учасника;
- consequence — наслідок (significant=true, якщо важливий);
- minor — дрібне уточнення.

Якщо поточного підсумку немає — це new_fact. position_changed=true лише коли нові
факти змінюють позицію сюжету.

Поверни лише JSON: {"update_type": "...", "significant": false,
"position_changed": false, "summary": "оновлений стан у 1-3 реченнях"}

ПОТОЧНИЙ СТАН:
{summary}

НОВА ПОДІЯ:
{event}"""


def _render_fact_base(fact_base: dict | None) -> str:
    if not isinstance(fact_base, dict):
        return ""
    lines: list[str] = []
    for fact in fact_base.get("facts", []) or []:
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        mark = " (розходяться цифри)" if fact.get("divergent") else ""
        lines.append(f"- {text}{mark}")
    return "\n".join(lines)


class LLMUpdateClassifier:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_UPDATE_PROMPT):
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

    def _call(self, current_summary: str | None, event_title: str | None, fact_base: dict | None) -> str | None:
        event_view = "\n".join(filter(None, [event_title or "", _render_fact_base(fact_base)])).strip()
        content = fill_prompt(self.prompt, summary=(current_summary or "(немає)"),
                              event=event_view or "(без опису)")
        try:
            from newsroom.llmutil import chat_json

            return chat_json(self._ensure_client(), model=self.model,
                             messages=[{"role": "user", "content": content}],
                             op="story_update", max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            log.warning("update classification failed", extra={"error": str(exc)})
            return None

    def classify(self, current_summary, event_title, fact_base) -> UpdateDecision:
        for attempt in (1, 2):
            parsed = parse_update(self._call(current_summary, event_title, fact_base))
            if parsed is not None:
                return parsed
            log.warning("update unparsable", extra={"attempt": attempt})
        # conservative fallback: a minor update, no post, no summary change
        return UpdateDecision(update_type=UPDATE_MINOR, significant=False, summary="", position_changed=False)


@dataclass(frozen=True)
class UpdateResult:
    event_id: int
    update_type: str = UPDATE_MINOR
    route: str = ROUTE_SUMMARY
    position_changed: bool = False
    skipped: bool = False


class StoryUpdater:
    def __init__(self, session_factory, *, classifier: UpdateClassifier,
                 charter_version: str = "0.2", prompt_version: str = "0.2"):
        self.sf = session_factory
        self.classifier = classifier
        self.charter_version = charter_version
        self.prompt_version = prompt_version

    def classify_event(self, event_id: int) -> UpdateResult:
        from sqlalchemy import func, select

        from newsroom.models import Decision, Event, Story, StoryVersion

        # 1. load state; skip if not linked to a story or already classified
        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is None or event.story_id is None or event.update_type is not None:
                return UpdateResult(event_id, skipped=True)
            story = s.get(Story, event.story_id)
            current_summary = story.current_summary if story else None
            event_title = event.title
            fact_base = event.fact_base

        # 2. classify (LLM, outside the session)
        decision = self.classifier.classify(current_summary, event_title, fact_base)
        route = route_update(decision.update_type, decision.significant)

        # 3. persist: event.update_type, story summary + this event's version, journal
        with self.sf() as s:
            event = s.get(Event, event_id)
            event.update_type = decision.update_type
            story = s.get(Story, event.story_id) if event.story_id else None
            if story is not None and decision.summary:
                story.current_summary = decision.summary
                version = s.execute(
                    select(StoryVersion)
                    .where(StoryVersion.story_id == story.id, StoryVersion.reason_event_id == event_id)
                    .order_by(StoryVersion.version.desc())
                ).scalars().first()
                if version is None:
                    next_version = int(s.scalar(
                        select(func.max(StoryVersion.version)).where(StoryVersion.story_id == story.id)
                    ) or 0) + 1
                    version = StoryVersion(story_id=story.id, version=next_version, reason_event_id=event_id)
                    s.add(version)
                version.summary = decision.summary
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="edit",
                decision=decision.update_type, reason=route,
                details={"route": route, "significant": decision.significant,
                         "position_changed": decision.position_changed},
                charter_version=self.charter_version, prompt_version=self.prompt_version,
                model=getattr(self.classifier, "model", None),
            ))
            s.commit()

        return UpdateResult(event_id, decision.update_type, route, decision.position_changed)


def classify_pending(session_factory, updater: "StoryUpdater", *, limit: int = 25,
                     significance_threshold: float | None = None,
                     classify_grace_seconds: float = 180.0,
                     require_dedup_settled: bool = False,
                     dedup_grace_seconds: float = 300.0) -> dict[str, int]:
    """One story-update tick: classify publishable events linked to a story that
    have no update_type yet. Returns counts, including how many warrant a post. When
    a significance threshold is given, low-significance events are skipped — they are
    not drafted anyway, so their update_type costs a call for nothing. With
    require_dedup_settled, an event waits for the ingest-dedup verdict first."""
    import datetime as dt

    from sqlalchemy import select

    from newsroom.analyze.ingest_dedup import dedup_settled_clause
    from newsroom.analyze.significance import significance_ready_clause
    from newsroom.models import Event

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(seconds=classify_grace_seconds)
    conditions = [
        Event.status.in_(("reported", "confirmed", "rumor")),
        Event.story_id.is_not(None),
        Event.update_type.is_(None),
        Event.duplicate_of.is_(None),        # a merged duplicate is inert — don't classify it
    ]
    clause = significance_ready_clause(significance_threshold, cutoff)
    if clause is not None:
        conditions.append(clause)
    dedup_clause = dedup_settled_clause(require_dedup_settled,
                                        now - dt.timedelta(seconds=dedup_grace_seconds))
    if dedup_clause is not None:
        conditions.append(dedup_clause)

    with session_factory() as s:
        ids = list(s.execute(
            select(Event.id).where(*conditions).order_by(Event.id).limit(limit)
        ).scalars().all())

    stats = {"classified": 0, "posts": 0, "summary_updates": 0}
    for event_id in ids:
        result = updater.classify_event(event_id)
        if result.skipped:
            continue
        stats["classified"] += 1
        stats["posts" if result.route == ROUTE_POST else "summary_updates"] += 1
    return stats
