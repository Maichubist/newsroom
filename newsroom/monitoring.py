"""Post-publication monitoring (architecture §9, step 7).

After we publish, the sources behind a post can change their minds: a channel
deletes the original post, or edits it into a retraction. When a source that fed
a *published* event deletes its item, we raise a correction: a draft follow-up
publication (kind="correction", replying to the original) plus a journal entry,
so a human is prompted to update the post and issue a separate notice (§10).
Nothing is auto-published — the correction stays a draft.

Detection keys off items.deleted_at (unambiguous; the Telegram collector sets it
on deletion — §12). Edit-into-retraction detection is semantic and left for later.
Idempotent: one correction per original publication. The correction TEXT is the
editor's job; monitoring only creates the draft slot and records why.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("newsroom.monitoring")

CORRECTION_KIND = "correction"


@dataclass(frozen=True)
class Retraction:
    item_id: int
    event_id: int
    publication_id: int
    source_id: int


def find_source_retractions(session_factory, *, limit: int = 50) -> list[Retraction]:
    """Published events whose source item was later deleted and which have no
    correction yet."""
    from sqlalchemy import select

    from newsroom.models import EventItem, Item, Publication

    with session_factory() as s:
        already_corrected = (
            select(Publication.reply_to_publication_id)
            .where(Publication.kind == CORRECTION_KIND, Publication.reply_to_publication_id.is_not(None))
        )
        rows = s.execute(
            select(Publication.id, Publication.event_id, Item.id, Item.source_id)
            .join(EventItem, EventItem.event_id == Publication.event_id)
            .join(Item, Item.id == EventItem.item_id)
            .where(
                Item.deleted_at.is_not(None),
                Publication.status == "published",
                Publication.kind != CORRECTION_KIND,
                Publication.id.not_in(already_corrected),
            )
            .order_by(Publication.id)
            .limit(limit)
        ).all()

    # one retraction per (publication, item); dedup within this batch
    seen: set[tuple[int, int]] = set()
    out: list[Retraction] = []
    for pub_id, event_id, item_id, source_id in rows:
        key = (pub_id, item_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(Retraction(item_id=item_id, event_id=event_id,
                              publication_id=pub_id, source_id=source_id))
    return out


def issue_correction(session_factory, retraction: Retraction, *, charter_version: str = "0.2") -> int | None:
    """Create a draft correction replying to the original publication, and
    journal it. Returns the new publication id, or None if one already exists."""
    from sqlalchemy import select

    from newsroom.models import Decision, Publication

    with session_factory() as s:
        original = s.get(Publication, retraction.publication_id)
        if original is None:
            return None
        exists = s.scalar(
            select(Publication.id).where(
                Publication.kind == CORRECTION_KIND,
                Publication.reply_to_publication_id == retraction.publication_id,
            )
        )
        if exists is not None:
            return None

        correction = Publication(
            event_id=retraction.event_id,
            channel=original.channel,
            kind=CORRECTION_KIND,
            reply_to_publication_id=original.id,
            headline="Оновлення: джерело видалило допис",
            body=None,   # the editor generates the correction text; this is the draft slot
            status="draft",
            charter_version=charter_version,
            features={"trigger": "source_retraction", "item_id": retraction.item_id,
                      "source_id": retraction.source_id, "needs_generation": True},
        )
        s.add(correction)
        s.flush()
        correction_id = correction.id
        s.add(Decision(
            entity_type="publication", entity_id=str(original.id), stage="publish",
            decision="correction_drafted",
            reason=f"source {retraction.source_id} deleted item {retraction.item_id}",
            details={"trigger": "source_retraction", "item_id": retraction.item_id,
                     "event_id": retraction.event_id, "correction_publication_id": correction_id},
            charter_version=charter_version,
        ))
        s.commit()
    return correction_id


def monitor_publications(session_factory, *, limit: int = 50) -> dict[str, int]:
    """One monitoring tick: draft corrections for published posts whose source
    retracted. Nothing is published."""
    stats = {"retractions": 0, "corrections": 0}
    for retraction in find_source_retractions(session_factory, limit=limit):
        stats["retractions"] += 1
        if issue_correction(session_factory, retraction):
            stats["corrections"] += 1
    return stats
