"""Evidence search (architecture §9, step 2).

The verdict step never lets the LLM judge a claim "from its head" — it only
compares the claim against evidence. This module gathers that evidence. The
first and always-available source is our own corpus: items already collected,
found by vector similarity to the claim (embedder-agnostic, like the clusterer).
Official registry and refutation databases (StopFake, VoxCheck, ЦПД) are
additional pluggable searchers added later.

Corpus hits are *candidate context*, not a stance: they are stored neutral and
the verdict step (§9 step 6) decides supports/refutes by reading them. The
selection policy (threshold, top-k) is pure/offline-tested; the DB query and the
embedder are the only non-pure parts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

log = logging.getLogger("newsroom.factcheck.evidence")

# supports | refutes | neutral
# corpus | official | factcheck_db | media_check


@dataclass(frozen=True)
class EvidenceRef:
    stance: str = "neutral"
    evidence_kind: str = "corpus"
    item_id: int | None = None
    external_ref: str | None = None
    score: float | None = None


def select_corpus_evidence(
    rows: Iterable[tuple[int, float]], *, top_k: int, min_similarity: float,
    evidence_kind: str = "corpus",
) -> list[EvidenceRef]:
    """Turn (item_id, similarity) candidates into evidence refs: drop anything
    below the similarity floor, keep the top_k most similar, store neutral (the
    verdict step assigns the stance). `evidence_kind` distinguishes the plain
    corpus from the official-source registry."""
    ranked = sorted(
        ((int(iid), float(sim)) for iid, sim in rows if sim >= min_similarity),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        EvidenceRef(stance="neutral", evidence_kind=evidence_kind, item_id=iid, score=sim)
        for iid, sim in ranked[:top_k]
    ]


class EvidenceSearcher(Protocol):
    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]: ...


class CorpusEvidenceSearcher:
    """Finds related material in our own item_embeddings by cosine similarity.

    Only embeddings from the same model are comparable, so the search is scoped
    to `embedder.model`. Items already tied to the event under review are passed
    as `exclude_item_ids` so a claim is not "confirmed" by its own source. With
    `official_only`, the search is restricted to items from official sources —
    the "official registry" leg of §9 step 2.
    """

    def __init__(self, session_factory, embedder, *, top_k: int = 5, min_similarity: float = 0.75,
                 official_only: bool = False, evidence_kind: str = "corpus"):
        self.session_factory = session_factory
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.official_only = official_only
        self.evidence_kind = evidence_kind

    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]:
        from sqlalchemy import select

        from newsroom.models import Item, ItemEmbedding, Source

        vec = self.embedder.embed(claim_text)
        vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        exclude = [int(i) for i in exclude_item_ids]

        dist = ItemEmbedding.vector.cosine_distance(vec_list).label("dist")
        query = select(ItemEmbedding.item_id, dist).where(ItemEmbedding.model == self.embedder.model)
        if self.official_only:
            query = (
                query.join(Item, Item.id == ItemEmbedding.item_id)
                .join(Source, Source.id == Item.source_id)
                .where(Source.is_official.is_(True))
            )
        if exclude:
            query = query.where(ItemEmbedding.item_id.not_in(exclude))
        query = query.order_by(dist).limit(self.top_k)

        with self.session_factory() as s:
            rows = s.execute(query).all()

        # cosine_distance = 1 - cosine_similarity
        candidates = [(item_id, 1.0 - float(d)) for item_id, d in rows]
        return select_corpus_evidence(candidates, top_k=self.top_k,
                                      min_similarity=self.min_similarity,
                                      evidence_kind=self.evidence_kind)


class OfficialRegistrySearcher(CorpusEvidenceSearcher):
    """Corpus search restricted to official sources (architecture §9 step 2):
    does an official statement in our archive corroborate the claim?"""

    def __init__(self, session_factory, embedder, *, top_k: int = 5, min_similarity: float = 0.78):
        super().__init__(session_factory, embedder, top_k=top_k, min_similarity=min_similarity,
                         official_only=True, evidence_kind="official")


class CompositeEvidenceSearcher:
    """Fans a claim out to several searchers (corpus + official registry +
    refutation DBs) and concatenates their evidence, dropping duplicate refs.
    A searcher that raises is skipped so one flaky external source never sinks
    the whole search."""

    def __init__(self, searchers: list[EvidenceSearcher], *, max_total: int = 12):
        self.searchers = list(searchers)
        self.max_total = max_total

    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]:
        seen: set[tuple] = set()
        out: list[EvidenceRef] = []
        for searcher in self.searchers:
            try:
                refs = searcher.search(claim_text, exclude_item_ids=exclude_item_ids)
            except Exception:  # noqa: BLE001 - one bad source must not sink the rest
                log.warning("evidence searcher failed", extra={"searcher": type(searcher).__name__})
                continue
            for ref in refs:
                key = (ref.evidence_kind, ref.item_id, ref.external_ref)
                if key in seen:
                    continue
                seen.add(key)
                out.append(ref)
                if len(out) >= self.max_total:
                    return out
        return out


def store_evidence(session, claim_id: int, refs: list[EvidenceRef]) -> list[int]:
    """Persist evidence rows for a claim. Flushes but does not commit."""
    from newsroom.models import ClaimEvidence

    ids: list[int] = []
    for ref in refs:
        row = ClaimEvidence(
            claim_id=claim_id,
            item_id=ref.item_id,
            external_ref=ref.external_ref,
            stance=ref.stance,
            evidence_kind=ref.evidence_kind,
        )
        session.add(row)
        session.flush()
        ids.append(row.id)
    return ids
