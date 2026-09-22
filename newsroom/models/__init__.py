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
    EventFacet,
    EventItem,
    FacetDimension,
    FacetPairMetric,
    FacetValue,
    Story,
    StoryVersion,
    TaxonomyNode,
)
from newsroom.models.publication import (
    PUBLICATION_STATUSES,
    STATUS_DRAFT,
    STATUS_PUBLISH_AMBIGUOUS,
    STATUS_PUBLISHED,
    STATUS_PUBLISHING,
    STATUS_REVIEW,
    STATUS_SUPERSEDED,
    ChannelMetric,
    ItemMetric,
    Publication,
    PublicationMetric,
    SourceMetric,
    is_valid_publication_status,
)
from newsroom.models.service import Decision, LlmCall, ReputationEvent, SystemState

__all__ = [
    # raw
    "Source", "Item", "ItemVersion", "MediaAsset", "ItemEmbedding", "Entity", "ItemEntity",
    # analytical
    "Event", "EventItem", "Story", "StoryVersion", "Claim", "ClaimEvidence", "TaxonomyNode",
    "FacetDimension", "FacetValue", "EventFacet", "FacetPairMetric",
    # publication
    "Publication", "PublicationMetric", "ChannelMetric", "ItemMetric", "SourceMetric",
    "PUBLICATION_STATUSES", "is_valid_publication_status",
    "STATUS_DRAFT", "STATUS_PUBLISHED", "STATUS_PUBLISHING", "STATUS_REVIEW",
    "STATUS_SUPERSEDED", "STATUS_PUBLISH_AMBIGUOUS",
    # service
    "ReputationEvent", "Decision", "LlmCall", "SystemState",
]
