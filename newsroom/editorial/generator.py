"""Post generator (pluggable). LLM in production, fake in tests.

Writes the charter §10 structured content from an event's fact base, in the
charter §6 voice. JSON is retried once and falls back to a minimal factual draft
(headline+lead from the given context) so a generation glitch never fabricates —
the critic still gates whatever comes out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from newsroom.editorial.draft import DraftContent, parse_draft

log = logging.getLogger("newsroom.editorial.generator")

MAX_SOURCE_EXCERPT_CHARS = 2000


@dataclass(frozen=True)
class GenerationContext:
    title: str
    summary: str = ""
    rubrics: list[str] = field(default_factory=list)
    status: str = "confirmed"
    facts: list[str] = field(default_factory=list)   # from the event's shared fact base (§8)
    source_excerpt: str = ""                          # trimmed source text, for concrete detail


class Generator(Protocol):
    model: str

    def generate(self, context: GenerationContext, *, feedback: str | None = None) -> DraftContent: ...


def build_material(context: GenerationContext) -> str:
    """Assemble what the generator writes from: the shared facts first (§8 — write
    from the fact base, not source framing), the story state, and a source excerpt
    for concrete detail. Pure and offline-tested."""
    parts: list[str] = []
    if context.title:
        parts.append(f"Подія: {context.title.strip()}")
    if context.facts:
        bullets = "\n".join(f"- {f.strip()}" for f in context.facts if f and f.strip())
        if bullets:
            parts.append("Факти (пиши з них):\n" + bullets)
    if context.summary.strip():
        parts.append("Стан сюжету: " + context.summary.strip())
    if context.source_excerpt.strip():
        parts.append("Витяг із джерел (лише для конкретних деталей, не копіюй стиль):\n"
                     + context.source_excerpt.strip())
    return "\n\n".join(parts).strip()


DEFAULT_PROMPT = """Ти — редактор українського новинного Telegram-каналу. Твоє завдання —
написати живий, конкретний пост СУВОРО за наданими фактами.

Пиши природною розмовною українською, як пише жива людина, а не машина. Поверни лише JSON:
{{"headline": "...", "lead": "...", "why_important": "...", "what_it_means": "...",
"watching": "...", "rubrics": ["..."]}}

ЖОРСТКІ ПРАВИЛА:
1. КОНКРЕТИКА. Кожне речення має нести факт із матеріалу — хто, що, де, коли, цифри,
   назви, імена. Жодних порожніх фраз на кшталт «варто бути готовими», «голоси можуть
   бути почуті», «читачам слід підготуватися», «це вплине на рівень життя».
2. ЗАБОРОНЕНО клікбейт і штампи в заголовку: «Сенсаційний», «Несподіване», «Критичний
   момент», «Скандал», «шокує», порожнє «що далі?». Заголовок = суть події конкретно.
3. Заголовок і текст мають збігатися: якщо в заголовку є теза — вона розкрита фактами в тексті.
4. Тільки те, що є в матеріалі. НЕ вигадуй деталей, цифр, подій. Якщо фактів мало —
   напиши коротко (headline + 1-2 фактичні речення) і залиш why_important/what_it_means
   порожніми, а не додавай воду.
5. what_it_means — конкретний наслідок для читача, а не банальність.
{feedback}
МАТЕРІАЛ:
{material}"""


class LLMGenerator:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_PROMPT):
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

    def _call(self, material: str, feedback: str | None) -> str | None:
        fb = f"Врахуй зауваження критика і перепиши: {feedback}\n" if feedback else ""
        try:
            resp = self._ensure_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt.format(feedback=fb, material=material)}],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            log.warning("generator call failed", extra={"error": str(exc)})
            return None

    def generate(self, context: GenerationContext, *, feedback: str | None = None) -> DraftContent:
        material = build_material(context)
        for attempt in (1, 2):
            draft = parse_draft(self._call(material, feedback))
            if draft is not None:
                return draft
            log.warning("draft unparsable", extra={"attempt": attempt})
        # conservative fallback: minimal factual draft from the context
        return DraftContent(headline=context.title or "Новина",
                            lead=context.summary or context.title or "",
                            rubrics=context.rubrics)
