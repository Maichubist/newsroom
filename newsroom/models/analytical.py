"""Analytical layer (architecture §5.2). Tables exist from 1a; populated from 1b-1c."""
from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from newsroom.db.base import EMBEDDING_DIM, Base


class Story(Base):
    __tablename__ = "stories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slug: Mapped[str] = mapped_column(String(160), unique=True)
    title: Mapped[str] = mapped_column(Text)
    rubric: Mapped[str | None] = mapped_column(String(64), nullable=True)

    hashtag: Mapped[str | None] = mapped_column(String(64), nullable=True)  # empty until 3+ updates
    state: Mapped[str] = mapped_column(String(16), default="new")           # new|developing|stable|dormant|closed
    current_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    centroid: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    last_event_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["Event"]] = relationship(back_populates="story")
    versions: Mapped[list["StoryVersion"]] = relationship(back_populates="story")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)

    fact_base: Mapped[dict | None] = mapped_column(JSONB, nullable=True)   # shared fact base across sources
    status: Mapped[str] = mapped_column(String(16), default="signal")      # signal|rumor|reported|confirmed|refuted
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)  # critical|high|low
    rubric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)

    first_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    independent_source_count: Mapped[int] = mapped_column(Integer, default=0)

    update_type: Mapped[str | None] = mapped_column(String(24), nullable=True)  # new_fact|confirmation|refutation|reaction|consequence|minor

    significance: Mapped[float | None] = mapped_column(Float, nullable=True)  # T1 gate score (analyze/significance.py)
    curated: Mapped[str | None] = mapped_column(String(16), nullable=True)  # publish|hold — editorial curation (editorial/curation.py)

    rumor_deadline_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rumor_outcome: Mapped[str | None] = mapped_column(String(24), nullable=True)

    centroid: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    story: Mapped["Story | None"] = relationship(back_populates="events")
    items: Mapped[list["EventItem"]] = relationship(back_populates="event")
    claims: Mapped[list["Claim"]] = relationship(back_populates="event")


class EventItem(Base):
    __tablename__ = "event_items"
    __table_args__ = (UniqueConstraint("event_id", "item_id", name="uq_event_items"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    role: Mapped[str | None] = mapped_column(String(16), nullable=True)   # origin|copy|reaction|official|evidence
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    added_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    event: Mapped["Event"] = relationship(back_populates="items")


class StoryVersion(Base):
    __tablename__ = "story_versions"
    __table_args__ = (UniqueConstraint("story_id", "version", name="uq_story_versions"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    story: Mapped["Story"] = relationship(back_populates="versions")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    event: Mapped["Event"] = relationship(back_populates="claims")
    evidence: Mapped[list["ClaimEvidence"]] = relationship(back_populates="claim")


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    item_id: Mapped[int | None] = mapped_column(ForeignKey("items.id"), nullable=True)
    external_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    stance: Mapped[str | None] = mapped_column(String(16), nullable=True)         # supports|refutes|neutral
    evidence_kind: Mapped[str | None] = mapped_column(String(24), nullable=True)  # corpus|official|factcheck_db|media_check

    claim: Mapped["Claim"] = relationship(back_populates="evidence")
