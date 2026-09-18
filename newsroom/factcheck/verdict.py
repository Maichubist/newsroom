"""Claim verdict (architecture §9, step 6).

The model compares one atomic claim against the evidence gathered in step 2 and
returns a verdict, a confidence and a short explanation, plus the stance of each
evidence piece (supports / refutes / neutral). It judges only from the evidence
it is shown — never from memory (CLAUDE.md: LLM compares, it does not verify from
its head). When the payload is unusable the judge falls back to the conservative
"unverifiable" verdict rather than asserting anything (asymmetry of errors, §3.5).

Parsing and the verdict/stance vocabularies are pure and offline-tested; the LLM
call and its retry/fallback are the only non-pure parts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.factcheck.verdict")

# claim-level verdict vocabulary (stored in claims.verdict)
VERDICT_TRUE = "true"
VERDICT_FALSE = "false"
VERDICT_MISLEADING = "misleading"
VERDICT_UNVERIFIABLE = "unverifiable"
VALID_VERDICTS = frozenset({VERDICT_TRUE, VERDICT_FALSE, VERDICT_MISLEADING, VERDICT_UNVERIFIABLE})

# per-evidence stance vocabulary (stored in claim_evidence.stance)
STANCE_SUPPORTS = "supports"
STANCE_REFUTES = "refutes"
STANCE_NEUTRAL = "neutral"
VALID_STANCES = frozenset({STANCE_SUPPORTS, STANCE_REFUTES, STANCE_NEUTRAL})


@dataclass(frozen=True)
class VerdictResult:
    verdict: str = VERDICT_UNVERIFIABLE
    confidence: float = 0.0
    explanation: str = ""
    stances: list[str] = field(default_factory=list)  # aligned to the evidence order passed in


def _coerce_confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


def parse_verdict(raw: str | None, *, evidence_count: int = 0) -> VerdictResult | None:
    """Parse the model's JSON verdict. Unknown verdicts collapse to
    'unverifiable' and unknown stances to 'neutral' (never invent a stronger
    claim than the model returned). The stance list is aligned to the evidence
    order and padded/truncated to `evidence_count`. None if unparsable so the
    caller can retry."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in VALID_VERDICTS:
        verdict = VERDICT_UNVERIFIABLE

    raw_stances = obj.get("stances")
    stances: list[str] = []
    if isinstance(raw_stances, list):
        for item in raw_stances:
            s = str(item or "").strip().lower()
            stances.append(s if s in VALID_STANCES else STANCE_NEUTRAL)
    # align to the evidence we actually have
    if evidence_count:
        stances = (stances + [STANCE_NEUTRAL] * evidence_count)[:evidence_count]

    return VerdictResult(
        verdict=verdict,
        confidence=_coerce_confidence(obj.get("confidence")),
        explanation=str(obj.get("explanation") or "").strip(),
        stances=stances,
    )


class VerdictJudge(Protocol):
    model: str

    def judge(self, claim_text: str, evidence_snippets: list[str]) -> VerdictResult: ...


DEFAULT_VERDICT_PROMPT = """Ти фактчекер. Оціни ТВЕРДЖЕННЯ лише на основі наведених
ДОКАЗІВ. Не спирайся на власні знання: якщо доказів недостатньо — вердикт
"unverifiable". Для кожного доказу вкажи позицію: "supports", "refutes" або
"neutral", у тому самому порядку.

Поверни лише JSON:
{"verdict": "true|false|misleading|unverifiable", "confidence": 0.0-1.0,
 "explanation": "...", "stances": ["supports|refutes|neutral", ...]}

ТВЕРДЖЕННЯ:
{claim}

ДОКАЗИ:
{evidence}"""


class LLMVerdictJudge:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_VERDICT_PROMPT):
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

    def _call(self, claim_text: str, evidence_snippets: list[str]) -> str | None:
        evidence_block = "\n".join(f"[{i}] {snip}" for i, snip in enumerate(evidence_snippets)) or "(немає)"
        content = fill_prompt(self.prompt, claim=claim_text, evidence=evidence_block)
        try:
            from newsroom.llmutil import chat_json

            return chat_json(self._ensure_client(), model=self.model,
                             messages=[{"role": "user", "content": content}],
                             op="factcheck_verdict", max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            log.warning("verdict call failed", extra={"error": str(exc)})
            return None

    def judge(self, claim_text: str, evidence_snippets: list[str]) -> VerdictResult:
        for attempt in (1, 2):
            parsed = parse_verdict(self._call(claim_text, evidence_snippets),
                                   evidence_count=len(evidence_snippets))
            if parsed is not None:
                return parsed
            log.warning("verdict unparsable", extra={"attempt": attempt})
        # conservative fallback: assert nothing
        return VerdictResult(verdict=VERDICT_UNVERIFIABLE, confidence=0.0, explanation="",
                             stances=[STANCE_NEUTRAL] * len(evidence_snippets))


def apply_verdict(session, claim_id: int, evidence_ids: list[int], result: VerdictResult) -> None:
    """Write the verdict onto the claim and the per-evidence stances. Flushes but
    does not commit. `evidence_ids` are the claim_evidence rows in the same order
    the snippets were passed to the judge, so stances line up positionally."""
    from newsroom.models import Claim, ClaimEvidence

    claim = session.get(Claim, claim_id)
    if claim is not None:
        claim.verdict = result.verdict
        claim.confidence = result.confidence

    for ev_id, stance in zip(evidence_ids, result.stances):
        row = session.get(ClaimEvidence, ev_id)
        if row is not None:
            row.stance = stance
    session.flush()
