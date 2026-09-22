"""LLM classifier for the verification pass.

Produces the two judgment inputs the deterministic layer cannot: is this a news
event, and its rubric(s) — plus side, first-source and rumor flags. JSON parsing
is retried once and falls back conservatively (CLAUDE.md): on failure we return
is_event=True with no rubric, which resolves to the highest risk level and thus
the strictest sourcing requirement, so nothing slips through under-verified.
"""
from __future__ import annotations

import json
import logging

from newsroom.analyze.facets import FACET_DIMENSIONS, FacetAssignment
from newsroom.analyze.verify import Classification
from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.analyze.classifier")

_SIDES = {"ua", "ru", "unknown"}

DEFAULT_PROMPT = """Ти редактор українського новинного каналу. Класифікуй матеріал
СТРОГО за наданим текстом. Поверни лише JSON без пояснень.

Поля:
- is_event: boolean — це конкретна новинна подія (факт, рішення, інцидент), а не
  колонка, реклама, добірка чи загальні роздуми.
- rubrics: масив рубрик зі списку (обери НАЙТОЧНІШУ, зазвичай одну):
  war (війна, фронт, атаки, удари, обстріли), defense (оборона, ОПК, озброєння,
  військова допомога), security (безпека, диверсії, тероризм), mobilization
  (мобілізація, ТЦК, призов), politics (внутрішня політика, влада, вибори),
  geopolitics (міжнародні відносини, санкції, дипломатія, зовнішня політика),
  economy (економіка, фінанси, бюджет, енергетика, бізнес), corruption (корупція,
  зловживання, розслідування НАБУ/САП), law_crime (кримінал, злочини, суди, право,
  звинувачення), society (суспільство, інциденти, транспорт, інфраструктура,
  освіта, здоров'я), tech_science (технології, наука, ІТ, космос), environment
  (довкілля, екологія, природа, погода), sport (спорт), culture (культура,
  шоубізнес, стиль життя).
- side: "ua" | "ru" | "unknown" — чиєї сторони стосується безпекова інформація.
- is_first_source: boolean — це першоджерело (документ, рішення суду, заява самої
  сторони), а не переказ.
- is_rumor: boolean — це неофіційна чутка зі зливного каналу.
- keywords: масив із 5–10 конкретних тем/сутностей матеріалу для аналітики трендів —
  ключові особи, організації, місця, події, явища. Українською, у називному відмінку
  однини (базова форма: "дрон", "Покровськ", "мобілізація", "Зеленський"). Без
  службових слів і загальників ("новина", "Україна", "сьогодні"). Синоніми зводь до
  одного слова ("шахед"/"безпілотник" → "дрон").
- facts: масив із 1–6 найважливіших атомарних фактів, достатніх для короткої новини.
  Кожен: {"text": "...", "modality": "fact|statement|forecast",
  "attribution": "хто заявив або null", "time_frame": "часова рамка або null",
  "number": число або null, "unit": "одиниця або null"}. Не дублюй формулювання,
  не додавай знань поза текстом і не перетворюй заяву/прогноз на встановлений факт.
- topic_path: масив із 2–5 рівнів теми від НАЙШИРШОГО до найвужчого — «піраміда» теми.
  Кожен рівень — стисла тема українською в називному відмінку однини, у нижньому
  регістрі. Приклади: ["війна", "атака рф", "удар бпла", "одеса"];
  ["технології", "пристрої", "смартфон", "новинка"]; ["економіка", "бюджет", "пенсії"].
  Перший рівень — широка сфера, далі дедалі конкретніше. Без службових слів.
- facets: масив незалежних фасетів. Це НЕ ще один topic_path. Кожен елемент:
  {"dimension": "event_type|geography|actor|target|entity|sector|impact|means|audience_scope|story",
   "path": ["ширше значення", "конкретніше значення"],
   "confidence": 0.0–1.0, "evidence": "коротка дослівна опора з матеріалу"}.
  Додавай лише підтверджені текстом значення. Не вигадуй масштаб, наслідки, засіб атаки
  чи загальнонаціональний вплив. Для geography дозволена ієрархія країна→область→місто;
  для target/sector — широка група→конкретний об'єкт. Максимум 16 фасетів.

Матеріал:
{news_text}"""


def parse_classification(raw: str | None) -> Classification | None:
    """Parse the model's JSON into a Classification, or None if unusable."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    side = str(obj.get("side") or "unknown").strip().lower()
    if side not in _SIDES:
        side = "unknown"
    rubrics = [str(r).strip().lower() for r in (obj.get("rubrics") or []) if str(r).strip()]
    kw_raw = obj.get("keywords") or []
    keywords = [str(k).strip() for k in kw_raw if str(k).strip()][:10] if isinstance(kw_raw, list) else []
    tp_raw = obj.get("topic_path") or []
    topic_path = ([str(t).strip().lower() for t in tp_raw if str(t).strip()][:5]
                  if isinstance(tp_raw, list) else [])
    compact_facts: list[dict] = []
    facts_raw = obj.get("facts") or []
    if isinstance(facts_raw, list):
        for row in facts_raw[:6]:
            if not isinstance(row, dict):
                continue
            fact_text = str(row.get("text") or "").strip()
            if not fact_text:
                continue
            modality = str(row.get("modality") or "fact").strip().lower()
            if modality not in {"fact", "statement", "forecast"}:
                modality = "fact"
            number = row.get("number")
            if number is not None and not isinstance(number, bool):
                try:
                    number = float(number)
                except (TypeError, ValueError):
                    number = None
            else:
                number = None
            compact_facts.append({
                "text": fact_text,
                "modality": modality,
                "attribution": (str(row.get("attribution")).strip()
                                if row.get("attribution") else None),
                "time_frame": (str(row.get("time_frame")).strip()
                               if row.get("time_frame") else None),
                "number": number,
                "unit": str(row.get("unit")).strip() if row.get("unit") else None,
            })
    facets: list[FacetAssignment] = []
    facets_raw = obj.get("facets") or []
    if isinstance(facets_raw, list):
        for row in facets_raw[:16]:
            if not isinstance(row, dict):
                continue
            dimension = str(row.get("dimension") or "").strip().lower()
            if dimension not in FACET_DIMENSIONS or dimension == "domain":
                continue
            path_raw = row.get("path") or []
            if isinstance(path_raw, str):
                path_raw = [path_raw]
            if not isinstance(path_raw, list):
                continue
            path = tuple(str(x).strip() for x in path_raw[:4] if str(x).strip())
            if not path:
                continue
            try:
                confidence = min(1.0, max(0.0, float(row.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0
            facets.append(FacetAssignment(
                dimension=dimension, path=path, confidence=confidence,
                evidence=str(row.get("evidence") or "").strip()[:500],
            ))
    return Classification(
        is_event=bool(obj.get("is_event")),
        rubrics=rubrics,
        side=side,
        is_first_source=bool(obj.get("is_first_source")),
        is_rumor=bool(obj.get("is_rumor")),
        keywords=keywords,
        compact_facts=compact_facts,
        topic_path=topic_path,
        facets=facets,
    )


# Conservative fallback: treat as an event with unknown rubric -> highest risk
# level -> strictest sourcing. Nothing publishes easily on a classifier failure.
_FALLBACK = Classification(is_event=True, rubrics=[], side="unknown")


class LLMClassifier:  # pragma: no cover - network
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

    def _call(self, news_text: str) -> str | None:
        try:
            from newsroom.llmutil import chat_json

            return chat_json(
                self._ensure_client(), model=self.model,
                messages=[{"role": "user", "content": fill_prompt(self.prompt, news_text=news_text)}],
                op="classify", max_tokens=768)
        except Exception as exc:  # noqa: BLE001
            log.warning("classifier call failed", extra={"error": str(exc)})
            return None

    def classify(self, title: str | None, text: str | None) -> Classification:
        news_text = f"{title or ''}\n{text or ''}".strip()
        for attempt in (1, 2):  # retry broken JSON once, then fall back
            parsed = parse_classification(self._call(news_text))
            if parsed is not None:
                return parsed
            log.warning("classification unparsable", extra={"attempt": attempt})
        return _FALLBACK
