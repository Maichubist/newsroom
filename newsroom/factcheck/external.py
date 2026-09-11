"""Refutation-database evidence (architecture §9, step 2).

Fact-check databases — StopFake, VoxCheck, ЦПД, Детектор медіа — publish
debunks. If one of them has already addressed a claim, that is strong evidence
for the verdict. These are external services with no clean shared API, so the
actual query/parse is a pluggable client (`# pragma: no cover`, off until
configured); this module keeps the parts that must be correct and testable: the
config registry and the mapping from a normalized hit to an EvidenceRef.

A refutation hit is stored with stance "refutes" and evidence_kind
"factcheck_db"; the verdict step still weighs it. Nothing here reaches the
network in tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import yaml

from newsroom.factcheck.evidence import EvidenceRef

log = logging.getLogger("newsroom.factcheck.external")


class FactCheckSourcesError(ValueError):
    """Raised when factcheck_sources.yaml is malformed."""


@dataclass(frozen=True)
class FactCheckDB:
    key: str
    name: str
    homepage: str
    lang: str
    kind: str
    enabled: bool


def load_factcheck_sources(path: str | Path) -> list[FactCheckDB]:
    path = Path(path)
    if not path.exists():
        raise FactCheckSourcesError(f"factcheck sources config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("databases")
    if not isinstance(raw, list):
        raise FactCheckSourcesError("databases must be a list")
    out: list[FactCheckDB] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("key"):
            raise FactCheckSourcesError(f"bad database entry: {entry!r}")
        out.append(FactCheckDB(
            key=str(entry["key"]),
            name=str(entry.get("name") or entry["key"]),
            homepage=str(entry.get("homepage") or ""),
            lang=str(entry.get("lang") or ""),
            kind=str(entry.get("kind") or "refutation"),
            enabled=bool(entry.get("enabled", False)),
        ))
    return out


@dataclass(frozen=True)
class ExternalHit:
    """One normalized result from a fact-check database."""
    url: str
    title: str = ""
    stance: str = "refutes"      # refutation DBs debunk; a client may override
    score: float | None = None


class RefutationClient(Protocol):
    """Queries one fact-check database and returns normalized hits. Real clients
    (HTTP/scraping/search) are supplied at wiring time; tests use fakes."""
    db_key: str

    def query(self, claim_text: str) -> list[ExternalHit]: ...


def hits_to_evidence(hits: Iterable[ExternalHit], *, min_score: float = 0.0,
                     max_hits: int = 3) -> list[EvidenceRef]:
    """Map normalized external hits to evidence refs. Drops hits below the score
    floor (when a client reports a score), keeps the best `max_hits`, stores the
    source URL as external_ref."""
    usable = [h for h in hits if h.url and (h.score is None or h.score >= min_score)]
    usable.sort(key=lambda h: (h.score is not None, h.score or 0.0), reverse=True)
    return [
        EvidenceRef(stance=(h.stance or "refutes"), evidence_kind="factcheck_db",
                    item_id=None, external_ref=h.url, score=h.score)
        for h in usable[:max_hits]
    ]


class FactCheckDBSearcher:
    """EvidenceSearcher backed by one or more refutation-database clients.
    Plugs into the CompositeEvidenceSearcher alongside corpus/official search."""

    def __init__(self, clients: list[RefutationClient], *, min_score: float = 0.0,
                 max_hits_per_db: int = 3):
        self.clients = list(clients)
        self.min_score = min_score
        self.max_hits_per_db = max_hits_per_db

    def search(self, claim_text: str, *, exclude_item_ids: Iterable[int] = ()) -> list[EvidenceRef]:
        out: list[EvidenceRef] = []
        for client in self.clients:
            try:
                hits = client.query(claim_text)
            except Exception:  # noqa: BLE001 - a flaky external DB must not sink the search
                log.warning("refutation client failed", extra={"db": getattr(client, "db_key", "?")})
                continue
            out.extend(hits_to_evidence(hits, min_score=self.min_score, max_hits=self.max_hits_per_db))
        return out
