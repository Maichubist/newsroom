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

from dataclasses import dataclass
from typing import Iterable, Protocol

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
    rows: Iterable[tuple[int, float]], *, top_k: int, min_similarity: float
) -> list[EvidenceRef]:
    """Turn (item_id, similarity) candidates into corpus evidence refs: drop
    anything below the similarity floor, keep the top_k most similar, store
    neutral (the verdict step assigns the stance)."""
    ranked = sorted(
        ((int(iid), float(sim)) for iid, sim in rows if sim >= min_similarity),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [
        EvidenceRef(stance="neutral", evidence_kind="corpus", item_id=iid, score=sim)
        for iid, sim in ranked[:top_k]
    ]


class EvidenceSearcher(Protocol):
    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]: ...


class CorpusEvidenceSearcher:
    """Finds related material in our own item_embeddings by cosine similarity.

    Only embeddings from the same model are comparable, so the search is scoped
    to `embedder.model`. Items already tied to the event under review are passed
    as `exclude_item_ids` so a claim is not "confirmed" by its own source.
    """

    def __init__(self, session_factory, embedder, *, top_k: int = 5, min_similarity: float = 0.75):
        self.session_factory = session_factory
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity

    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]:
        from sqlalchemy import select

        from newsroom.models import ItemEmbedding

        vec = self.embedder.embed(claim_text)
        vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        exclude = [int(i) for i in exclude_item_ids]

        dist = ItemEmbedding.vector.cosine_distance(vec_list).label("dist")
        query = select(ItemEmbedding.item_id, dist).where(ItemEmbedding.model == self.embedder.model)
        if exclude:
            query = query.where(ItemEmbedding.item_id.not_in(exclude))
        query = query.order_by(dist).limit(self.top_k)

        with self.session_factory() as s:
            rows = s.execute(query).all()

        # cosine_distance = 1 - cosine_similarity
        candidates = [(item_id, 1.0 - float(d)) for item_id, d in rows]
        return select_corpus_evidence(candidates, top_k=self.top_k, min_similarity=self.min_similarity)


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
