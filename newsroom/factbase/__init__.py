from newsroom.factbase.builder import (
    DEFAULT_FACT_THRESHOLD,
    FactExtractor,
    LLMFactExtractor,
    MergedFact,
    SourceFact,
    VectorFact,
    fact_base_json,
    merge_facts,
    parse_facts,
)
from newsroom.factbase.pipeline import (
    FACTBASE_STATUSES,
    FactBaseBuilder,
    FactBaseResult,
    build_pending,
)

__all__ = [
    # facts + merge (§8.1-8.2)
    "SourceFact", "VectorFact", "MergedFact", "FactExtractor", "LLMFactExtractor",
    "parse_facts", "merge_facts", "fact_base_json", "DEFAULT_FACT_THRESHOLD",
    # orchestration
    "FactBaseBuilder", "FactBaseResult", "build_pending", "FACTBASE_STATUSES",
]
