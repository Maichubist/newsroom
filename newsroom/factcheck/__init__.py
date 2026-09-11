from newsroom.factcheck.claims import (
    ClaimDraft,
    ClaimExtractor,
    LLMClaimExtractor,
    parse_claims,
    store_claims,
)
from newsroom.factcheck.evidence import (
    CompositeEvidenceSearcher,
    CorpusEvidenceSearcher,
    EvidenceRef,
    EvidenceSearcher,
    OfficialRegistrySearcher,
    select_corpus_evidence,
    store_evidence,
)
from newsroom.factcheck.external import (
    ExternalHit,
    FactCheckDB,
    FactCheckDBSearcher,
    RefutationClient,
    hits_to_evidence,
    load_factcheck_sources,
)
from newsroom.factcheck.verdict import (
    LLMVerdictJudge,
    VerdictJudge,
    VerdictResult,
    apply_verdict,
    parse_verdict,
)
from newsroom.factcheck.media import (
    DEFAULT_PHASH_MAX_DISTANCE,
    MediaChecker,
    MediaCheckResult,
    MediaMatch,
    find_reused_media,
    phash_distance,
    select_reused,
)
from newsroom.factcheck.pipeline import (
    CHECKABLE_STATUSES,
    FactChecker,
    FactCheckResult,
    check_pending,
)

__all__ = [
    # claims (§9 step 1)
    "ClaimDraft", "ClaimExtractor", "LLMClaimExtractor", "parse_claims", "store_claims",
    # evidence (§9 step 2)
    "EvidenceRef", "EvidenceSearcher", "CorpusEvidenceSearcher", "OfficialRegistrySearcher",
    "CompositeEvidenceSearcher", "select_corpus_evidence", "store_evidence",
    # external refutation DBs (§9 step 2)
    "FactCheckDB", "load_factcheck_sources", "ExternalHit", "RefutationClient",
    "FactCheckDBSearcher", "hits_to_evidence",
    # verdict (§9 step 6)
    "VerdictResult", "VerdictJudge", "LLMVerdictJudge", "parse_verdict", "apply_verdict",
    # media reuse check (§9 step 4)
    "phash_distance", "select_reused", "find_reused_media", "MediaMatch",
    "MediaChecker", "MediaCheckResult", "DEFAULT_PHASH_MAX_DISTANCE",
    # orchestration
    "FactChecker", "FactCheckResult", "check_pending", "CHECKABLE_STATUSES",
]
