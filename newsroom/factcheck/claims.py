"""Atomic claim extraction (architecture §9, step 1).

The LLM extracts atomic claims (who/what/where/when/numbers) it can only read
from the text — evidence search (own corpus, official registry, StopFake/VoxCheck)
and the verdict come later (§9 steps 2–6). Claims are stored in the `claims`
table (created in 1а). Parsing is pure/offline-tested; the LLM extractor retries
once and falls back to no claims (never invents).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.factcheck.claims")

# who | what | where | when | number | quote | cause — free-form, lowercased.


@dataclass(frozen=True)
class ClaimDraft:
    text: str
    claim_type: str = ""


def parse_claims(raw: str | None) -> list[ClaimDraft] | None:
    """Parse the model's JSON into claims. Accepts {"claims": [...]} or a bare
    list; items may be objects {text, claim_type} or plain strings. None if the
    payload is unusable (so the caller can retry)."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    items = obj.get("claims") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return None
    out: list[ClaimDraft] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                out.append(ClaimDraft(text=text, claim_type=str(item.get("claim_type") or "").strip().lower()))
        elif isinstance(item, str) and item.strip():
            out.append(ClaimDraft(text=item.strip()))
    return out


class ClaimExtractor(Protocol):
    model: str

    def extract(self, title: str | None, text: str | None) -> list[ClaimDraft]: ...


DEFAULT_CLAIM_PROMPT = """Ти фактчекер. Виділи атомарні твердження з тексту —
кожне окреме перевірюване твердження (хто, що, де, коли, ключові цифри). Не додавай
нічого, чого немає в тексті. Поверни лише JSON: {"claims": [{"text": "...",
"claim_type": "who|what|where|when|number|quote"}]}.

Матеріал:
{news_text}"""


class LLMClaimExtractor:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_CLAIM_PROMPT):
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
                messages=[{"role": "user", "content": fill_prompt(self.prompt, news_text=news_text)}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            log.warning("claim extraction failed", extra={"error": str(exc)})
            return None

    def extract(self, title: str | None, text: str | None) -> list[ClaimDraft]:
        news_text = f"{title or ''}\n{text or ''}".strip()
        for attempt in (1, 2):
            parsed = parse_claims(self._call(news_text))
            if parsed is not None:
                return parsed
            log.warning("claims unparsable", extra={"attempt": attempt})
        return []   # fall back to no claims rather than inventing


def store_claims(session, event_id: int, claims: list[ClaimDraft]) -> list[int]:
    """Persist claims for an event. Flushes but does not commit (caller commits)."""
    from newsroom.models import Claim

    ids: list[int] = []
    for claim in claims:
        row = Claim(event_id=event_id, text=claim.text, claim_type=claim.claim_type or None)
        session.add(row)
        session.flush()
        ids.append(row.id)
    return ids
