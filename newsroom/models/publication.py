"""Publication layer (architecture §5.3)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from newsroom.db.base import Base

# publication.status — single source of truth (a raw-string typo would otherwise hide a
# post from the queue forever). Kept as a plain column (not a DB enum) so new states are
# additive-migration-free; validate against PUBLICATION_STATUSES at the write sites.
STATUS_DRAFT = "draft"
STATUS_PUBLISHED = "published"
STATUS_EDITED = "edited"
STATUS_RETRACTED = "retracted"
STATUS_DELETED = "deleted"
STATUS_REVIEW = "review"              # held by publish-time dedup for a human decision
STATUS_SUPERSEDED = "superseded"     # a duplicate / folded-update draft that never publishes
STATUS_PUBLISHING = "publishing"     # claimed for delivery (outbox); reconciled after a crash
STATUS_PUBLISH_AMBIGUOUS = "ambiguous"   # crashed mid-send — a human must confirm delivery (fits varchar(16))
PUBLICATION_STATUSES = frozenset({
    STATUS_DRAFT, STATUS_PUBLISHED, STATUS_EDITED, STATUS_RETRACTED, STATUS_DELETED,
    STATUS_REVIEW, STATUS_SUPERSEDED, STATUS_PUBLISHING, STATUS_PUBLISH_AMBIGUOUS,
})


def is_valid_publication_status(status: str) -> bool:
    return status in PUBLICATION_STATUSES


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(16))          # telegram | site | viber
    channel_ref: Mapped[str | None] = mapped_column(Text, nullable=True)  # message id in the channel

    kind: Mapped[str] = mapped_column(String(24))             # post|update|correction|refutation|rumor_followup
    reply_to_publication_id: Mapped[int | None] = mapped_column(ForeignKey("publications.id"), nullable=True)

    headline: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)

    # one of PUBLICATION_STATUSES (see module top). String, not a DB enum, so new states
    # need no migration; the app validates on write (is_valid_publication_status).
    status: Mapped[str] = mapped_column(String(16), default=STATUS_DRAFT)

    charter_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # emotional vector, headline type, length, rubric
    # Persisted media readiness snapshot. It is refreshed by the media workers and
    # again immediately before publishing, so restarts never lose the reason a draft
    # is waiting. media_status: none | discovering | pending | checking | ready |
    # unavailable | blocked.
    has_media: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    media_status: Mapped[str] = mapped_column(String(16), default="none", server_default="none")
    media_expected_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    media_ready_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    media_failed_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Set only after the final text publish gate and pre-publish dedup pass. Media
    # workers use this as their authorization boundary; it does not bypass a fresh
    # gate evaluation immediately before delivery.
    media_approved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    metrics: Mapped[list["PublicationMetric"]] = relationship(back_populates="publication")


class PublicationMetric(Base):
    __tablename__ = "publication_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id"), index=True)
    measured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reactions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    forwards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comments: Mapped[int | None] = mapped_column(Integer, nullable=True)

    publication: Mapped["Publication"] = relationship(back_populates="metrics")


class ChannelMetric(Base):
    __tablename__ = "channel_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)
    measured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    subscribers: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ItemMetric(Base):
    """Engagement snapshot for a SOURCE post over time (demand intelligence). Mirrors
    PublicationMetric but for other channels' items — how the ecosystem reacts."""
    __tablename__ = "item_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    measured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reactions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    forwards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comments: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SourceMetric(Base):
    """Subscriber-count snapshot for a source, for normalising engagement by reach."""
    __tablename__ = "source_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    measured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    subscribers: Mapped[int | None] = mapped_column(Integer, nullable=True)
