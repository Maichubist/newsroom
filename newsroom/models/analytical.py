"""Analytical layer (architecture §5.2). Tables exist from 1a; populated from 1b-1c."""
from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
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

    # Classification cache (analyze/verify.py): the LLM classifier runs ONCE per
    # event, not per item — reprints that join an already-classified event reuse
    # these fields and cost nothing. classifier_model non-null = event classified.
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)            # ua|ru|unknown
    is_first_source: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # document/court ruling/party's own statement
    is_rumor: Mapped[bool | None] = mapped_column(Boolean, nullable=True)         # leak-channel rumor (charter 3.7)
    classifier_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    keywords: Mapped[list | None] = mapped_column(JSONB, nullable=True)           # 5-10 topic keywords (data-driven hot-topics layer)
    # learned taxonomy pyramid (charter v0.3 §3.1): the broad->specific topic path and
    # the id of its leaf node in taxonomy_nodes (for engagement roll-up up the tree).
    topic_path: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    topic_leaf_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)

    significance: Mapped[float | None] = mapped_column(Float, nullable=True)  # T1 gate score (analyze/significance.py)
    curated: Mapped[str | None] = mapped_column(String(16), nullable=True)  # publish|hold — editorial curation (editorial/curation.py)
    duplicate_of: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # same-story canonical event id (LLM batch dedup); non-null = don't post

    rumor_deadline_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rumor_outcome: Mapped[str | None] = mapped_column(String(24), nullable=True)

    centroid: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    story: Mapped["Story | None"] = relationship(back_populates="events")
    items: Mapped[list["EventItem"]] = relationship(back_populates="event")
    claims: Mapped[list["Claim"]] = relationship(back_populates="event")


class TaxonomyNode(Base):
    """A node in the learned topic pyramid (charter v0.3 §3.1). The tree is built from
    the topic paths the classifier emits per event — not hand-authored. `parent_id` is
    NULL for a top-level node; `slug` is the normalized label, unique among siblings so
    the same path always maps to the same node chain. `depth` is 0 at the root level."""

    __tablename__ = "taxonomy_nodes"
    __table_args__ = (UniqueConstraint("parent_id", "slug", name="uq_taxonomy_parent_slug"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("taxonomy_nodes.id"), nullable=True, index=True)
    slug: Mapped[str] = mapped_column(String(120))          # normalized label (match key)
    label: Mapped[str] = mapped_column(String(200))         # display label as first seen
    depth: Mapped[int] = mapped_column(Integer, default=0)
    event_count: Mapped[int] = mapped_column(Integer, default=0)   # lifetime count of events whose path passes through this node
    # engagement heat (Phase 2): recency-weighted Telegram engagement rolled up the tree,
    # normalized within the node's depth level to [0,1]; recomputed each analytics tick.
    heat: Mapped[float] = mapped_column(Float, default=0.0)
    heat_events: Mapped[int] = mapped_column(Integer, default=0)   # events under this node in the heat window
    heat_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # audience-engagement popularity (competitor engagement, NO recency), normalized within
    # the node's depth level to [0,1]; recomputed with heat. Curation reads it at L2.
    demand: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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
