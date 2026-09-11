"""Shared fact base — one viewpoint per event (architecture §8).

The generator must write from facts, not from any single source's framing. So for
each event we:
  1. extract facts, unique claims and reactions from every source separately (LLM);
  2. merge them into a shared base — which facts several independent sources
     confirm, and where their numbers diverge;
  3. store that base on events.fact_base (JSONB) for the generator to write from.

Facts are clustered by vector similarity, never by matching LLM-generated strings
(CLAUDE.md), reusing the event clusterer's cosine primitives at fact granularity.
Extraction and embedding are pluggable (LLM in prod, fakes in tests); parsing and
the merge are pure and offline-tested.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from newsroom.analyze.clustering import best_match, update_centroid

log = logging.getLogger("newsroom.factbase")

DEFAULT_FACT_THRESHOLD = 0.85

# kind: fact | claim | reaction | frame  (architecture §8.1)
_MERGEABLE_KINDS = frozenset({"fact", "claim", ""})


@dataclass(frozen=True)
class SourceFact:
    text: str
    kind: str = "fact"
    number: float | None = None
    unit: str | None = None


@dataclass(frozen=True)
class VectorFact:
    fact: SourceFact
    source_id: int
    vector: Sequence[float]


@dataclass(frozen=True)
class MergedFact:
    text: str
    source_ids: list[int]
    confirmed_by: int             # number of independent sources asserting it
    values: list[float] = field(default_factory=list)   # distinct numbers reported
    divergent: bool = False       # sources report different numbers
    variants: list[str] = field(default_factory=list)   # distinct wordings seen


def parse_facts(raw: str | None) -> list[SourceFact] | None:
    """Parse the model's JSON into source facts. Accepts {"facts": [...]} or a
    bare list; items are objects {text, kind, number, unit} or plain strings.
    None if unusable (so the caller can retry)."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    items = obj.get("facts") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return None
    out: list[SourceFact] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            out.append(SourceFact(
                text=text,
                kind=(str(item.get("kind") or "fact").strip().lower() or "fact"),
                number=_coerce_number(item.get("number")),
                unit=(str(item.get("unit")).strip() if item.get("unit") else None),
            ))
        elif isinstance(item, str) and item.strip():
            out.append(SourceFact(text=item.strip()))
    return out


def _coerce_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _distinct_numbers(nums: list[float], ndigits: int = 4) -> list[float]:
    seen: list[float] = []
    for n in nums:
        r = round(n, ndigits)
        if r not in seen:
            seen.append(r)
    return seen


def merge_facts(items: list[VectorFact], *, threshold: float = DEFAULT_FACT_THRESHOLD) -> list[MergedFact]:
    """Cluster facts across sources by vector similarity and summarise each
    cluster: which independent sources assert it (confirmed_by), the distinct
    numbers reported, and whether those numbers diverge. Reactions/frames are not
    merged here (they carry attribution and are handled separately)."""
    clusters: list[dict] = []
    for it in items:
        if it.fact.kind not in _MERGEABLE_KINDS:
            continue
        idx, _sim = best_match(it.vector, [c["centroid"] for c in clusters], threshold)
        if idx is None:
            clusters.append({
                "centroid": list(it.vector), "count": 1,
                "source_ids": [it.source_id], "numbers": _num_list(it.fact),
                "variants": [it.fact.text],
            })
        else:
            c = clusters[idx]
            c["centroid"] = update_centroid(c["centroid"], c["count"], it.vector)
            c["count"] += 1
            if it.source_id not in c["source_ids"]:
                c["source_ids"].append(it.source_id)
            c["numbers"].extend(_num_list(it.fact))
            if it.fact.text not in c["variants"]:
                c["variants"].append(it.fact.text)

    merged: list[MergedFact] = []
    for c in clusters:
        values = _distinct_numbers(c["numbers"])
        merged.append(MergedFact(
            text=c["variants"][0],
            source_ids=c["source_ids"],
            confirmed_by=len(c["source_ids"]),
            values=values,
            divergent=len(values) > 1,
            variants=c["variants"],
        ))
    return merged


def _num_list(fact: SourceFact) -> list[float]:
    return [fact.number] if fact.number is not None else []


def fact_base_json(merged: list[MergedFact], reactions: list[tuple[int, SourceFact]]) -> dict:
    """Serialise the shared base into the events.fact_base shape. Reactions keep
    their source attribution (architecture §8.3)."""
    return {
        "facts": [
            {
                "text": m.text,
                "source_ids": m.source_ids,
                "confirmed_by": m.confirmed_by,
                "values": m.values,
                "divergent": m.divergent,
                "variants": m.variants,
            }
            for m in merged
        ],
        "reactions": [
            {"source_id": sid, "text": f.text, "unit": f.unit}
            for sid, f in reactions
        ],
    }


class FactExtractor(Protocol):
    model: str

    def extract(self, source_name: str | None, title: str | None, text: str | None) -> list[SourceFact]: ...


DEFAULT_FACT_PROMPT = """Ти редактор. З матеріалу ОДНОГО джерела виділи окремо:
факти (перевірювані твердження, з цифрами якщо є), унікальні твердження цього
джерела та реакції (з атрибуцією). Не додавай нічого, чого немає в тексті.

Поверни лише JSON: {"facts": [{"text": "...", "kind": "fact|claim|reaction|frame",
"number": 13.0, "unit": "%"}]}. number/unit — лише для кількісних фактів.

Матеріал:
{news_text}"""


class LLMFactExtractor:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_FACT_PROMPT):
        import os

        self.model = model
        self.prompt = prompt
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _call(self, news_text: str) -> str | None:
        try:
            resp = self._ensure_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt.format(news_text=news_text)}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            log.warning("fact extraction failed", extra={"error": str(exc)})
            return None

    def extract(self, source_name: str | None, title: str | None, text: str | None) -> list[SourceFact]:
        news_text = f"{title or ''}\n{text or ''}".strip()
        for attempt in (1, 2):
            parsed = parse_facts(self._call(news_text))
            if parsed is not None:
                return parsed
            log.warning("facts unparsable", extra={"attempt": attempt})
        return []   # fall back to no facts rather than inventing
