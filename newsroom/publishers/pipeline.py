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


def _render_send_body(pub) -> str:
    """The text actually sent to Telegram: HTML rendered from the stored pieces
    (bold headline, linked sources) when present, else the plain pub.body. Gate
    and stop-list checks still run on the plain body, never on this."""
    render = (pub.features or {}).get("render")
    if isinstance(render, dict):
        from newsroom.publishers.format import render_telegram_html

        return render_telegram_html(
            headline=render.get("headline") or pub.headline or "",
            body=render.get("body") or "",
            watching=render.get("watching") or "",
            hashtags=render.get("hashtags") or [],
            source_links=[tuple(x) for x in (render.get("source_links") or [])],
            is_rumor=bool(render.get("is_rumor")),
            reported=bool(render.get("reported")),
        )
    return pub.body or ""


class Publisher:
    def __init__(self, session_factory, *, telegram, stoplist_rules, limits: Limits,
                 supervisor=None, media_store=None, purge_media_after_publish: bool = False,
                 require_vision: bool = True, charter_version: str = "0.3"):
        self.sf = session_factory
        self.telegram = telegram
        self.stoplist_rules = stoplist_rules
        self.limits = limits
        self.supervisor = supervisor
        # media_store lets the publisher read stored Telegram media to upload it;
        # purge_media_after_publish additionally deletes local files once a post is out.
        self.media_store = media_store
        self.purge_media_after_publish = bool(purge_media_after_publish)
        # require a vision (media stop-list) verdict before attaching media. False when
        # vision moderation is off — media then attaches on the reuse check alone.
        self.require_vision = bool(require_vision)
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

        now = _utc_now()
        from newsroom.models import Event as _Event

        # posts already out for THIS event's story in the cooldown window (one/story)
        story_recent_posts = 0
        story_id = event.story_id if event else None
        if story_id is not None:
            story_recent_posts = int(s.scalar(
                select(func.count()).select_from(Publication).join(_Event, _Event.id == Publication.event_id)
                .where(Publication.status == "published",
                       Publication.published_at >= now - dt.timedelta(minutes=self.limits.story_cooldown_minutes),
                       _Event.story_id == story_id)
            ) or 0)
        is_refutation = bool(event and (event.update_type or "").lower() == "refutation")

        return GateInputs(
            stopped=is_publishing_stopped(s),
            critic_ok=bool(features.get("critic_ok", False)),
            stoplist_blocked=is_blocked(violations) or (needs_official and not has_official),
            story_recent_posts=story_recent_posts,
            is_refutation=is_refutation,
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
            body = _render_send_body(pub)
            media_choice = self._media_choice(s, pub.event_id)
            reply_to_pub_id, reply_to_message_id = self._story_reply_target(s, event)

        decision = evaluate_gate(inputs, self.limits)
        if not decision.allow:
            self._record_block(publication_id, decision.reasons)
            return PublishOutcome(publication_id, published=False, reasons=decision.reasons)

        # Prefer uploading the locally stored bytes over sending a URL: Telegram often
        # cannot fetch a source URL (telesco.pe / hotlink-protected images 400 with
        # "failed to get HTTP URL content"). If there is no file and no fetchable URL,
        # drop the media.
        media_file = self._media_file(media_choice) if media_choice is not None else None
        if media_choice is not None and media_file is None and not media_choice.url:
            media_choice = None

        result = self.telegram.send_post(body, media_choice, reply_to_message_id=reply_to_message_id,
                                         file=media_file)
        if not result.ok and media_choice is not None:
            # a post must never be lost to a bad image — retry once as text only
            log.warning("media send failed; retrying as text",
                        extra={"publication_id_": publication_id, "error": result.error})
            result = self.telegram.send_post(body, None, reply_to_message_id=reply_to_message_id)
        with self.sf() as s:
            pub = s.get(Publication, publication_id)
            if result.ok:
                shadow = bool(getattr(self.telegram, "shadow", False))
                pub.status = "published"
                pub.channel_ref = str(result.message_id) if result.message_id is not None else None
                pub.published_at = _utc_now()
                pub.features = {**(pub.features or {}), "shadow": shadow}
                if reply_to_pub_id is not None:
                    pub.reply_to_publication_id = reply_to_pub_id
                headline = pub.headline
                s.add(Decision(
                    entity_type="publication", entity_id=str(publication_id), stage="publish",
                    decision="published", reason=None,
                    details={"message_id": result.message_id, "channel": pub.channel, "shadow": shadow},
                    charter_version=self.charter_version,
                ))
                s.commit()
                feats = pub.features or {}
                pub_is_rumor = bool(feats.get("is_rumor") or (feats.get("render") or {}).get("is_rumor"))
                from newsroom.models import Event as _Ev
                ev = s.get(_Ev, pub.event_id) if pub.event_id else None
                self._notify_supervisor(publication_id, headline,
                                        risk_level=(ev.risk_level if ev else None),
                                        is_rumor=pub_is_rumor, message_id=result.message_id)
                self._purge_media(pub.event_id)
                return PublishOutcome(publication_id, published=True, message_id=result.message_id)
            s.add(Decision(
                entity_type="publication", entity_id=str(publication_id), stage="publish",
                decision="publish_failed", reason=result.error, details={"error": result.error},
                charter_version=self.charter_version,
            ))
            s.commit()
            return PublishOutcome(publication_id, published=False, reasons=["send_failed"])

    def _media_choice(self, s, event_id):
        """Pick media to attach — only for an event whose media passed the §9.4 reuse
        check (media_clean, no media_reuse) and the image stop-list / vision check
        (media_vision_ok, no media_vision_block). In doubt, no media (§3.5).

        When vision moderation is disabled (require_vision=False), the vision-ok
        requirement is dropped — media attaches on the reuse check alone — but an
        explicit media_vision_block already on record is still honoured. NOTE: this
        relaxes the charter media stop-list; intended only for the closed test channel."""
        from sqlalchemy import select

        from newsroom.publishers.cascade import MediaItem, choose_media

        if event_id is None:
            return None
        from newsroom.models import Decision, EventItem, Item, MediaAsset

        decisions = set(s.execute(
            select(Decision.decision).where(
                Decision.entity_type == "event", Decision.entity_id == str(event_id),
                Decision.stage == "verify",
                Decision.decision.in_(("media_clean", "media_reuse",
                                       "media_vision_ok", "media_vision_block")),
            )
        ).scalars().all())
        reuse_ok = "media_clean" in decisions and "media_reuse" not in decisions
        vision_blocked = "media_vision_block" in decisions
        if self.require_vision:
            vision_ok = "media_vision_ok" in decisions and not vision_blocked
        else:
            vision_ok = not vision_blocked          # attach without a vision verdict
        if not (reuse_ok and vision_ok):
            return None

        from sqlalchemy import and_, or_

        rows = s.execute(
            select(MediaAsset.kind, MediaAsset.url, MediaAsset.width, MediaAsset.size_bytes,
                   MediaAsset.storage_key)
            .join(Item, Item.id == MediaAsset.item_id)
            .join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == event_id,
                   or_(MediaAsset.url.is_not(None),
                       and_(MediaAsset.storage_key.is_not(None), MediaAsset.purged_at.is_(None))))
        ).all()
        items = [MediaItem(kind=k, url=u, width=w, size_bytes=sb, storage_key=sk)
                 for k, u, w, sb, sk in rows]
        return choose_media(items)

    def _media_file(self, choice):
        """Read the choice's locally stored bytes for upload (filename, bytes), or None
        if there is no store or the file is gone (caller then sends the URL, or text).
        Uploading the bytes is preferred over a URL — Telegram can't always fetch it."""
        if self.media_store is None or not choice.storage_key:
            return None
        data = self.media_store.get(choice.storage_key)
        if not data:
            return None
        ext = "mp4" if choice.param == "video" else "jpg"
        return (f"{choice.param}.{ext}", data)

    def _story_reply_target(self, s, event):
        """The story's most recent published Telegram post in the current channel
        context, so this post replies to it (architecture §7). Returns
        (publication_id, message_id) or (None, None)."""
        from sqlalchemy import select

        from newsroom.models import Event, Publication

        if event is None or event.story_id is None:
            return None, None
        shadow = bool(getattr(self.telegram, "shadow", False))
        row = s.execute(
            select(Publication.id, Publication.channel_ref)
            .join(Event, Event.id == Publication.event_id)
            .where(
                Event.story_id == event.story_id,
                Publication.channel == "telegram",
                Publication.status == "published",
                Publication.channel_ref.is_not(None),
                Publication.features["shadow"].as_boolean().is_(shadow),
            )
            .order_by(Publication.published_at.desc())
        ).first()
        if row is None:
            return None, None
        pub_id, channel_ref = row
        try:
            return pub_id, int(channel_ref)
        except (TypeError, ValueError):
            return pub_id, None

    def _purge_media(self, event_id) -> None:
        """Delete the event's local media files right after a successful publish
        (opt-in). A cleanup failure must never fail the publish."""
        if not self.purge_media_after_publish or self.media_store is None or event_id is None:
            return
        try:
            from newsroom.media import purge_event_media

            purge_event_media(self.sf, self.media_store, event_id)
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            log.exception("post-publish media purge failed")

    def _notify_supervisor(self, publication_id, headline, *, risk_level, is_rumor, message_id) -> None:
        if self.supervisor is None:
            return
        try:
            self.supervisor.notify_published(
                headline=headline, risk_level=risk_level, is_rumor=is_rumor,
                channel_ref=str(message_id) if message_id is not None else None,
                publication_id=publication_id,
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

    def publish_pending(self, *, limit: int = 1, scan: int = 60) -> dict[str, int]:
        """One publish tick: publish up to `limit` posts, scanning up to `scan` drafts
        by significance and SKIPPING blocked ones so a perpetually-blocked top draft
        (e.g. a rate-limited critical post) does not head-of-line-block the queue —
        a publishable lower post still goes out. If the master switch is off, do
        nothing (and touch no network)."""
        from sqlalchemy import func, or_, select

        from newsroom.models import Event, Publication

        if not self.telegram.is_enabled():
            return {"published": 0, "blocked": 0, "disabled": 1}

        now = _utc_now()
        debounce_cut = now - dt.timedelta(minutes=self.limits.publish_debounce_minutes)
        with self.sf() as s:
            # most-significant first; scan a window wider than `limit` so blocked
            # posts are stepped over, not stuck on (unscored significance sorts last)
            ids = list(s.execute(
                select(Publication.id)
                .join(Event, Event.id == Publication.event_id)
                .where(
                    Publication.status == "draft",
                    Publication.kind == "post",
                    Publication.event_id.is_not(None),
                    Publication.features["critic_ok"].as_boolean().is_(True),
                    # honour dedup/digest at publish time: a draft may have been created
                    # before the event was marked a duplicate or reserved to the digest
                    Event.duplicate_of.is_(None),
                    Event.curated.is_distinct_from("digest"),
                    Event.curated.is_distinct_from("digested"),
                    # debounce: hold a young event so cross-source twins arrive and get
                    # merged before the first publishes; a refutation goes out immediately
                    or_(Event.first_seen_at < debounce_cut,
                        func.lower(func.coalesce(Event.update_type, "")) == "refutation"),
                )
                .order_by(func.coalesce(Event.significance, 0.0).desc(), Publication.id)
                .limit(max(limit, scan))
            ).scalars().all())

        stats = {"published": 0, "blocked": 0, "disabled": 0}
        for pid in ids:
            if stats["published"] >= limit:
                break
            outcome = self.publish_one(pid)
            if outcome.skipped:
                continue
            stats["published" if outcome.published else "blocked"] += 1
        return stats
