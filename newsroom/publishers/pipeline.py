"""Publish orchestration (architecture §10, §13).

Takes a draft publication that already passed the editor's critic and decides —
through the publish gate — whether it may go out right now, then sends it via the
Telegram adapter and records the outcome. The gate inputs are assembled fresh
from the database (a re-run of the stop-list on the final body, the event's risk
level and official-source status, the stop button, and rate/surge counts), so
the last check before publishing never trusts a stale flag.

Nothing goes out unless PUBLISH_ENABLED is on and the gate allows it. Every
attempt leaves a decision trace; a blocked draft stays a draft and is retried
later (a hold may clear), with a fresh trace only when the reason changes.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from newsroom.analyze.stoplist import check as stoplist_check
from newsroom.analyze.stoplist import is_blocked
from newsroom.publishers.gate import GateInputs, Limits, evaluate_gate, is_publishing_stopped

log = logging.getLogger("newsroom.publishers.pipeline")


@dataclass(frozen=True)
class PublishOutcome:
    publication_id: int
    published: bool = False
    reasons: list[str] = field(default_factory=list)
    message_id: int | None = None
    skipped: bool = False


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Publisher:
    def __init__(self, session_factory, *, telegram, stoplist_rules, limits: Limits,
                 supervisor=None, charter_version: str = "0.2"):
        self.sf = session_factory
        self.telegram = telegram
        self.stoplist_rules = stoplist_rules
        self.limits = limits
        self.supervisor = supervisor
        self.charter_version = charter_version

    # ------------------------------------------------------------------
    def _gather_inputs(self, s, pub, event) -> GateInputs:
        from sqlalchemy import func, select

        from newsroom.models import EventItem, Item, Publication, Source

        features = pub.features or {}
        headline, body = pub.headline or "", pub.body or ""

        violations = stoplist_check(headline, body, self.stoplist_rules)
        needs_official = any(v.action == "needs_official" for v in violations)

        has_official = False
        if pub.event_id is not None:
            has_official = bool(s.scalar(
                select(func.count()).select_from(Source)
                .join(Item, Item.source_id == Source.id)
                .join(EventItem, EventItem.item_id == Item.id)
                .where(EventItem.event_id == pub.event_id, Source.is_official.is_(True))
            ))

        risk_level = event.risk_level if event else None
        rubric = event.rubric if event else None
        is_rumor = bool(features.get("is_rumor")) or (event is not None and event.status == "rumor")
        now = _utc_now()

        from newsroom.models import Event as _Event
        urgent_last_hour = int(s.scalar(
            select(func.count()).select_from(Publication).join(_Event, _Event.id == Publication.event_id)
            .where(Publication.status == "published", Publication.published_at >= now - dt.timedelta(hours=1),
                   _Event.risk_level == "critical")
        ) or 0)
        rumors_last_day = int(s.scalar(
            select(func.count()).select_from(Publication).join(_Event, _Event.id == Publication.event_id)
            .where(Publication.status == "published", Publication.published_at >= now - dt.timedelta(days=1),
                   _Event.status == "rumor")
        ) or 0)
        surge_same_rubric = 0
        if rubric:
            surge_same_rubric = int(s.scalar(
                select(func.count()).select_from(Publication).join(_Event, _Event.id == Publication.event_id)
                .where(Publication.status == "published",
                       Publication.published_at >= now - dt.timedelta(minutes=self.limits.surge_window_minutes),
                       _Event.rubric == rubric)
            ) or 0)

        return GateInputs(
            stopped=is_publishing_stopped(s),
            critic_ok=bool(features.get("critic_ok", False)),
            stoplist_blocked=is_blocked(violations) or (needs_official and not has_official),
            has_official_source=has_official,
            is_rumor=is_rumor,
            rumor_labeled=body.strip().startswith("Чутка"),
            risk_level=risk_level,
            urgent_last_hour=urgent_last_hour,
            rumors_last_day=rumors_last_day,
            surge_same_rubric=surge_same_rubric,
        )

    def publish_one(self, publication_id: int) -> PublishOutcome:
        from newsroom.models import Decision, Event, Publication

        if not self.telegram.is_enabled():
            return PublishOutcome(publication_id, reasons=["publish_disabled"], skipped=True)

        with self.sf() as s:
            pub = s.get(Publication, publication_id)
            if pub is None or pub.status != "draft":
                return PublishOutcome(publication_id, skipped=True)
            event = s.get(Event, pub.event_id) if pub.event_id else None
            inputs = self._gather_inputs(s, pub, event)
            body = pub.body or ""

        decision = evaluate_gate(inputs, self.limits)
        if not decision.allow:
            self._record_block(publication_id, decision.reasons)
            return PublishOutcome(publication_id, published=False, reasons=decision.reasons)

        result = self.telegram.send_text(body)
        with self.sf() as s:
            pub = s.get(Publication, publication_id)
            if result.ok:
                pub.status = "published"
                pub.channel_ref = str(result.message_id) if result.message_id is not None else None
                pub.published_at = _utc_now()
                headline = pub.headline
                s.add(Decision(
                    entity_type="publication", entity_id=str(publication_id), stage="publish",
                    decision="published", reason=None,
                    details={"message_id": result.message_id, "channel": pub.channel},
                    charter_version=self.charter_version,
                ))
                s.commit()
                self._notify_supervisor(headline, inputs, result.message_id)
                return PublishOutcome(publication_id, published=True, message_id=result.message_id)
            s.add(Decision(
                entity_type="publication", entity_id=str(publication_id), stage="publish",
                decision="publish_failed", reason=result.error, details={"error": result.error},
                charter_version=self.charter_version,
            ))
            s.commit()
            return PublishOutcome(publication_id, published=False, reasons=["send_failed"])

    def _notify_supervisor(self, headline, inputs: GateInputs, message_id) -> None:
        if self.supervisor is None:
            return
        try:
            self.supervisor.notify_published(
                headline=headline, risk_level=inputs.risk_level, is_rumor=inputs.is_rumor,
                channel_ref=str(message_id) if message_id is not None else None,
            )
        except Exception:  # noqa: BLE001 - a failed notice must not fail the publish
            log.exception("supervisor notify failed")

    def _record_block(self, publication_id: int, reasons: list[str]) -> None:
        """Journal a block, but only when it is new or the reasons changed —
        a persistent hold does not spam identical decisions every tick."""
        from sqlalchemy import select

        from newsroom.models import Decision

        reason_str = ",".join(reasons)
        with self.sf() as s:
            last = s.execute(
                select(Decision)
                .where(Decision.entity_type == "publication", Decision.entity_id == str(publication_id),
                       Decision.stage == "publish")
                .order_by(Decision.id.desc())
            ).scalars().first()
            if last is not None and last.decision == "blocked" and last.reason == reason_str:
                return
            s.add(Decision(
                entity_type="publication", entity_id=str(publication_id), stage="publish",
                decision="blocked", reason=reason_str, details={"reasons": reasons},
                charter_version=self.charter_version,
            ))
            s.commit()

    def publish_pending(self, *, limit: int = 25) -> dict[str, int]:
        """One publish tick: attempt every critic-passed draft post. If the master
        switch is off, do nothing (and touch no network)."""
        from sqlalchemy import select

        from newsroom.models import Publication

        if not self.telegram.is_enabled():
            return {"published": 0, "blocked": 0, "disabled": 1}

        with self.sf() as s:
            ids = list(s.execute(
                select(Publication.id)
                .where(
                    Publication.status == "draft",
                    Publication.kind == "post",
                    Publication.event_id.is_not(None),
                    Publication.features["critic_ok"].as_boolean().is_(True),
                )
                .order_by(Publication.id)
                .limit(limit)
            ).scalars().all())

        stats = {"published": 0, "blocked": 0, "disabled": 0}
        for pid in ids:
            outcome = self.publish_one(pid)
            if outcome.skipped:
                continue
            stats["published" if outcome.published else "blocked"] += 1
        return stats
