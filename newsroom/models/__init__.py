"""All ORM models. Importing this module registers every table on Base.metadata.

Statuses, risk levels, tiers, roles etc. are stored as plain strings (not DB
enums) because the charter defines them as versioned configuration, not
hard-wired constants (architecture §6, CLAUDE.md).
"""
from newsroom.models.raw import (
    Entity,
    Item,
    ItemEmbedding,
    ItemEntity,
    ItemVersion,
    MediaAsset,
    Source,
)
from newsroom.models.analytical import (
    Claim,
    ClaimEvidence,
    Event,
    EventItem,
    Story,
    StoryVersion,
)
from newsroom.models.publication import (
    ChannelMetric,
    ItemMetric,
    Publication,
    PublicationMetric,
    SourceMetric,
)
from newsroom.models.service import Decision, ReputationEvent, SystemState

__all__ = [
    # raw
    "Source", "Item", "ItemVersion", "MediaAsset", "ItemEmbedding", "Entity", "ItemEntity",
    # analytical
    "Event", "EventItem", "Story", "StoryVersion", "Claim", "ClaimEvidence",
    # publication
    "Publication", "PublicationMetric", "ChannelMetric", "ItemMetric", "SourceMetric",
    # service
    "ReputationEvent", "Decision", "SystemState",
]
