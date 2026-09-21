from newsroom.analyze.independence import (
    SourceItem,
    group_materials,
    independent_source_count,
)
from newsroom.analyze.risk import (
    GateDecision,
    RiskConfigError,
    RiskMatrix,
    decide,
    load_risk_matrix,
)
from newsroom.analyze.signal import (
    FilterConfigError,
    FiltersConfig,
    NoiseVerdict,
    classify_noise,
    ipso_markers,
    load_filters,
)
from newsroom.analyze.stoplist import (
    StopListConfigError,
    StopRule,
    Violation,
    check,
    is_blocked,
    load_stoplist,
    worst_action,
)
from newsroom.analyze.clustering import (
    AssignResult,
    EventClusterer,
    best_match,
    cosine,
    update_centroid,
)
from newsroom.analyze.verify import (
    Classification,
    Classifier,
    VerifyResult,
    Verifier,
)
from newsroom.analyze.stories import (
    StoryAssignResult,
    StoryLinker,
    mark_dormant,
    slugify,
)
from newsroom.analyze.ai_accent import (
    AccentHit,
    AiAccentConfigError,
    check_ai_accent,
    load_ai_accent,
)
# NB: ingest_dedup (Phase B) is intentionally NOT re-exported here — it imports from
# newsroom.publishers.predup, and pulling the publishers package into every
# `import newsroom.analyze` would risk an import cycle. Import it by full path
# (from newsroom.analyze.ingest_dedup import ...), like taxonomy/demand.

__all__ = [
    "SourceItem", "group_materials", "independent_source_count",
    "RiskMatrix", "GateDecision", "RiskConfigError", "load_risk_matrix", "decide",
    "FiltersConfig", "FilterConfigError", "NoiseVerdict", "load_filters",
    "classify_noise", "ipso_markers",
    "StopRule", "Violation", "StopListConfigError", "load_stoplist", "check",
    "is_blocked", "worst_action",
    "EventClusterer", "AssignResult", "cosine", "update_centroid", "best_match",
    "Verifier", "Classifier", "Classification", "VerifyResult",
    "StoryLinker", "StoryAssignResult", "mark_dormant", "slugify",
    "AccentHit", "AiAccentConfigError", "load_ai_accent", "check_ai_accent",
]
