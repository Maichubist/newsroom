"""Raw layer (architecture §5.1) — immutable record of everything sources gave us."""
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


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("kind", "handle_or_url", name="uq_sources_kind_handle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))          # rss | telegram | site | x | threads
    handle_or_url: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)

    origin: Mapped[str] = mapped_column(String(16))         # ua | world
    lang: Mapped[str | None] = mapped_column(String(16), nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)

    tier: Mapped[str] = mapped_column(String(32))           # official | media | aggregator | leak | anonymous
    is_official: Mapped[bool] = mapped_column(Boolean, default=False)

    active: Mapped[bool] = mapped_column(Boolean, default=True)
    poll_interval: Mapped[int] = mapped_column(Integer, default=300)  # seconds

    # source health monitoring
    last_success_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    items: Mapped[list["Item"]] = relationship(back_populates="source")


class Item(Base):
    __tablename__ = "items"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_items_source_external"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    external_id: Mapped[str] = mapped_column(Text)

    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    lang: Mapped[str | None] = mapped_column(String(16), nullable=True)

    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    forwarded_from: Mapped[str | None] = mapped_column(Text, nullable=True)   # Telegram original source
    grouped_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # Telegram album id

    content_hash: Mapped[str] = mapped_column(String(64), index=True)         # exact cross-source dedup
    simhash: Mapped[int | None] = mapped_column(BigInteger, nullable=True)     # rewritten-copy detection

    status: Mapped[str] = mapped_column(String(32), default="new", index=True)  # new | filtered_out | accepted | clustered

    edited_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    raw_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    source: Mapped["Source"] = relationship(back_populates="items")
    versions: Mapped[list["ItemVersion"]] = relationship(back_populates="item")
    media: Mapped[list["MediaAsset"]] = relationship(back_populates="item")


class ItemVersion(Base):
    __tablename__ = "item_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    item: Mapped["Item"] = relationship(back_populates="versions")


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))   # image | video | embed
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    phash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # reused-image search
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # set only after download (post-filter)
    # Set when the local file was deleted after publication (the bytes are sent to
    # Telegram by URL and never read locally again; the phash reuse-archive lives in
    # the DB). storage_key is kept so nothing re-downloads; purged_at means gone-locally.
    purged_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    first_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    item: Mapped["Item"] = relationship(back_populates="media")


class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"
    __table_args__ = (UniqueConstraint("item_id", "model", name="uq_item_embeddings_item_model"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    model: Mapped[str] = mapped_column(String(128))
    vector: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("kind", "canonical_name", name="uq_entities_kind_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))   # person | org | place
    canonical_name: Mapped[str] = mapped_column(Text)
    aliases: Mapped[list | None] = mapped_column(JSONB, nullable=True)


class ItemEntity(Base):
    __tablename__ = "item_entities"

    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
