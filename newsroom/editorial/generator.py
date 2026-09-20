"""Post generator (pluggable). LLM in production, fake in tests.

Writes the charter §10 structured content from an event's fact base, in the
charter §6 voice. JSON is retried once and falls back to a minimal factual draft
(headline+lead from the given context) so a generation glitch never fabricates —
the critic still gates whatever comes out.

Posts are FACTS ONLY — no analysis / interpretation / "what it means". Editorial
analysis is a separate rubric (later), so the news feed stays dry and credible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from newsroom.editorial.draft import DraftContent, parse_draft
from newsroom.promptutil import fill_prompt

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
    """Assemble what the generator writes from: the rubric (register), the shared
    facts first (§8 — write from the fact base, not source framing), the story
    state, and a source excerpt for concrete detail. Pure and offline-tested."""
    parts: list[str] = []
    if context.title:
        parts.append(f"Подія: {context.title.strip()}")
    if context.rubrics:
        parts.append("Рубрика: " + context.rubrics[0].strip())
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


DEFAULT_PROMPT = """Ти — редактор українського новинного Telegram-каналу. Напиши живий,
конкретний пост СУВОРО за наданими фактами — так, як написала б жива людина, а не машина.

Поверни лише JSON: {"headline": "...", "body": "...", "watching": "...", "rubrics": ["..."]}

ЯК ПИСАТИ ТЕКСТ (body):
- Це 1–3 короткі природні абзаци суцільним текстом, а не набір рубрик і ярликів.
- Почни з найважливішого або найцікавішого в події — знайди суть, а не переказуй заголовок.
- Переходи між абзацами роби природно. НЕ вживай канцелярських зв'язок: «Таким чином»,
  «Цей крок свідчить», «Цей випадок демонструє», «Це підкреслює», «Варто зазначити».
- ЖОДНОГО аналізу, інтерпретації, прогнозів чи оцінки значення («що це означає», «це
  свідчить про…», «це вплине на…», «це важливо, бо…»). Пиши ТІЛЬКИ факти: що сталося,
  де, коли, хто, скільки, з чиїх слів. Аналітику винесемо в окрему рубрику — тут її не місце.
- НЕ спотворюй суть факту при переказі: «оголосили / виділили / планують / домовились» ≠
  «зробили / отримали / збудували». Зберігай КЛЮЧОВІ уточнення — мету, суму, умову, хто саме
  й з чиїх слів (напр. «3,3 млрд євро НА ОБОРОННІ ПОТРЕБИ», а не просто «3,3 млрд євро»).
  Без цих уточнень факт змінює зміст. Не змішуй у пості різні події, якщо їх кілька в матеріалі.

ЗАБОРОНЕНО (саме через це попередні пости були мертві):
- Очевидні або порожні «висновки»: «чоловік тепер служитиме в армії», «це свідчить про
  високу довіру», «це може вплинути на рівень життя», «читачам варто стежити». Якщо
  твердження й так випливає з факту — не пиши його.
- Клікбейт і штампи в заголовку: «Сенсаційний», «Несподіване», «Скандал», «шокує»,
  порожнє «що далі?». Заголовок = суть події конкретно.
- Вигадані деталі, цифри, події. Тільки те, що є в матеріалі. Якщо фактів мало — напиши
  коротко (заголовок + 1 абзац), не додавай воду.

РЕГІСТР ЗА РУБРИКОЮ:
- спорт: результат і що далі, без пафосу й глобальних висновків.
- війна / оборона / безпека: лише факти з атрибуцією джерела, жодних припущень.
- економіка: конкретна цифра і кого саме вона стосується.
- політика / суспільство: суть рішення і хто за ним стоїть.

watching — окремий короткий рядок лише про конкретне продовження: дата, очікуване рішення,
названа сторона. НЕ пиши «дивіться відео/фото», «стежте за новинами». Якщо конкретного
продовження немає — залиш "".
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
            from newsroom.llmutil import chat_json

            return chat_json(
                self._ensure_client(), model=self.model,
                messages=[{"role": "user", "content": fill_prompt(self.prompt, feedback=fb, material=material)}],
                op="generate", max_tokens=2048, temperature=0.4)
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
        # conservative fallback: minimal factual draft from the context, marked so
        # the pipeline never publishes it — it holds the event for the next tick
        # (a transient failure, e.g. a 429, must not become a placeholder post).
        return DraftContent(headline=context.title or "Новина",
                            body=context.summary or context.title or "",
                            rubrics=context.rubrics, fallback=True)
