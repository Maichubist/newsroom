from newsroom.publishers.telegram import PublishResult, TelegramPublisher, split_text
from newsroom.publishers.cascade import MediaChoice, MediaItem, MediaLimits, choose_media
from newsroom.publishers.gate import (
    STOP_KEY,
    GateDecision,
    GateInputs,
    Limits,
    evaluate_gate,
    is_publishing_stopped,
    load_limits,
    set_publishing_stopped,
)
from newsroom.publishers.pipeline import Publisher, PublishOutcome
from newsroom.publishers.supervision import Supervisor, format_publish_notice
from newsroom.publishers.metrics import (
    MessageStats,
    MetricsCollector,
    MetricsSource,
    TelethonMetricsSource,
    record_channel_metric,
    record_publication_metric,
)
from newsroom.publishers.shadow import (
    ShadowCriteria,
    ShadowReport,
    dup_rate_from_simhashes,
    load_shadow_criteria,
    shadow_report,
)

__all__ = [
    "TelegramPublisher", "PublishResult", "split_text",
    # media cascade (§13)
    "MediaItem", "MediaChoice", "MediaLimits", "choose_media",
    # gate (§10)
    "GateInputs", "GateDecision", "Limits", "evaluate_gate", "load_limits",
    "is_publishing_stopped", "set_publishing_stopped", "STOP_KEY",
    # orchestration (§10, §13)
    "Publisher", "PublishOutcome",
    # supervision (§10)
    "Supervisor", "format_publish_notice",
    # metrics (§5.3)
    "MessageStats", "MetricsSource", "MetricsCollector", "TelethonMetricsSource",
    "record_publication_metric", "record_channel_metric",
    # shadow mode (§2)
    "ShadowCriteria", "ShadowReport", "load_shadow_criteria", "shadow_report",
    "dup_rate_from_simhashes",
]
