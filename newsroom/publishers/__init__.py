from newsroom.publishers.telegram import PublishResult, TelegramPublisher, split_text
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

__all__ = [
    "TelegramPublisher", "PublishResult", "split_text",
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
]
