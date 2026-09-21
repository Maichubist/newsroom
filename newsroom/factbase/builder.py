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
from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.factbase")

DEFAULT_FACT_THRESHOLD = 0.85

# kind: fact | claim | reaction | frame  (architecture §8.1)
_MERGEABLE_KINDS = frozenset({"fact", "claim", ""})

# modality (charter: don't overstate) — fact = established; statement = someone asserts it,
# not proven; forecast = intent/possibility ("планує", "може", "розглядає"). Higher rank =
# more cautious: when sources disagree, the merged fact takes the more cautious modality so a
# claim is never presented as an established fact.
_MODALITIES = frozenset({"fact", "statement", "forecast"})
_MODALITY_RANK = {"fact": 0, "statement": 1, "forecast": 2}


@dataclass(frozen=True)
class SourceFact:
    text: str
    kind: str = "fact"
    number: float | None = None
    unit: str | None = None
    modality: str = "fact"           # fact | statement | forecast
    attribution: str | None = None   # WHO asserts it (statement/forecast) — the actor named in the text
    time_frame: str | None = None    # temporal/conditional qualifier to preserve ("з грудня 2023")


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
    modality: str = "fact"        # most-cautious modality across the cluster
    attribution: str | None = None
    time_frame: str | None = None


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
            modality = (str(item.get("modality") or "fact").strip().lower() or "fact")
            if modality not in _MODALITIES:
                modality = "fact"
            attribution = str(item.get("attribution")).strip() if item.get("attribution") else None
            time_frame = str(item.get("time_frame")).strip() if item.get("time_frame") else None
            out.append(SourceFact(
                text=text,
                kind=(str(item.get("kind") or "fact").strip().lower() or "fact"),
                number=_coerce_number(item.get("number")),
                unit=(str(item.get("unit")).strip() if item.get("unit") else None),
                modality=modality,
                attribution=attribution or None,
                time_frame=time_frame or None,
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
                "modality": it.fact.modality, "attribution": it.fact.attribution,
                "time_frame": it.fact.time_frame,
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
            # keep the MORE CAUTIOUS modality; fill attribution/time_frame if still missing
            if _MODALITY_RANK.get(it.fact.modality, 0) > _MODALITY_RANK.get(c["modality"], 0):
                c["modality"] = it.fact.modality
            if not c["attribution"] and it.fact.attribution:
                c["attribution"] = it.fact.attribution
            if not c["time_frame"] and it.fact.time_frame:
                c["time_frame"] = it.fact.time_frame

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
            modality=c["modality"],
            attribution=c["attribution"],
            time_frame=c["time_frame"],
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
                "modality": m.modality,
                "attribution": m.attribution,
                "time_frame": m.time_frame,
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


DEFAULT_FACT_PROMPT = """Ти редактор. З матеріалу ОДНОГО джерела виділи окремо факти,
унікальні твердження цього джерела та реакції. Не додавай нічого, чого немає в тексті.

Для КОЖНОГО пункту познач:
- kind: fact|claim|reaction|frame.
- modality: fact (встановлений факт) | statement (ЗАЯВА/твердження когось, не доведене) |
  forecast (прогноз/намір/можливість: «планує», «має намір», «може», «розглядає», «готує»).
  НЕ подавай заяву чи прогноз як доконаний факт.
- attribution: ХТО це заявив/прогнозує (орган, посадовець, джерело) — обов'язково для
  statement і forecast.
- time_frame: часова або умовна рамка, якщо є («з грудня 2023», «у 2026–2027», «після виборів»).
- number/unit: лише для кількісних фактів.

Приклади: «Генштаб РФ опрацював варіант мобілізації 600 тис.» → modality statement,
attribution «українська розвідка», time_frame «у 2026–2027». «Естонія може закрити кордон» →
forecast. «Росія захопила 1,5% території» → time_frame «з грудня 2023». «Україна отримала
3,3 млрд євро від ЄС» → fact.

Поверни лише JSON: {"facts": [{"text": "...", "kind": "fact|claim|reaction|frame",
"modality": "fact|statement|forecast", "attribution": "...", "time_frame": "...",
"number": 13.0, "unit": "%"}]}. attribution/time_frame/number/unit — лише коли доречно.

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
            from newsroom.llmutil import chat_json

            return chat_json(
                self._ensure_client(), model=self.model,
                messages=[{"role": "user", "content": fill_prompt(self.prompt, news_text=news_text)}],
                op="factbase", max_tokens=2048)
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
