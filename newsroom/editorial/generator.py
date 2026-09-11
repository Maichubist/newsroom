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


@dataclass(frozen=True)
class GenerationContext:
    title: str
    summary: str = ""
    rubrics: list[str] = field(default_factory=list)
    status: str = "confirmed"


class Generator(Protocol):
    model: str

    def generate(self, context: GenerationContext, *, feedback: str | None = None) -> DraftContent: ...


DEFAULT_PROMPT = """Ти — редактор українського новинного Telegram-каналу. Пиши живою
розмовною українською, конкретно, без канцеляриту й без штампів. Поверни лише JSON.

Структура (charter §10):
- headline: гострий заголовок з суттю або інтригою;
- lead: суть у першому реченні;
- why_important: чому це важливо;
- what_it_means: що це означає для читача;
- watching: за чим стежити далі (для сюжетів; інакше порожньо);
- rubrics: рубрики матеріалу.

Тільки факти з наданого матеріалу. Без вигаданих деталей і цифр.
{feedback}
Матеріал:
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
        material = f"{context.title}\n\n{context.summary}".strip()
        for attempt in (1, 2):
            draft = parse_draft(self._call(material, feedback))
            if draft is not None:
                return draft
            log.warning("draft unparsable", extra={"attempt": attempt})
        # conservative fallback: minimal factual draft from the context
        return DraftContent(headline=context.title or "Новина",
                            lead=context.summary or context.title or "",
                            rubrics=context.rubrics)
