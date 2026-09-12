"""Publication layer (architecture §5.3)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
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

    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft|published|edited|retracted|deleted

    charter_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # emotional vector, headline type, length, rubric
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


class SourceMetric(Base):
    """Subscriber-count snapshot for a source, for normalising engagement by reach."""
    __tablename__ = "source_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    measured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    subscribers: Mapped[int | None] = mapped_column(Integer, nullable=True)
