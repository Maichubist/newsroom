"""Порівняти якість і вартість LLM-моделей на РЕАЛЬНИХ завданнях пайплайна.

Мета (запит власника): дослідити, наскільки новіші моделі кращі/гірші за поточну
`gpt-4o-mini`, і виміряти «якість за 1 цент» — скільки правильних рішень купує кожен цент.

Що робить:
  1. classify — ганяє СПРАВЖНІЙ промпт класифікатора (14-рубриковий хребет) на золотому
     наборі з 14 українських новин (по одній на рубрику) і рахує точність рубрики.
  2. generate — ганяє СПРАВЖНІЙ промпт генератора на 3 фактбазах-пастках (спотворення суми,
     заява/прогноз, чистий спорт) і перевіряє збереження ключових уточнень + маркерів.
  3. Для кожної моделі підсумовує токени й вартість (та сама PRICES, що й у llm_calls) і
     рахує метрику QPC = точність / (вартість одного виклику в центах).

Параметри викликів адаптуються під сімейство моделі: reasoning-моделі (gpt-5*, o*) не
приймають `max_tokens`/`temperature=0` — їм ідуть `max_completion_tokens` + `reasoning_effort`.
Це саме по собі висновок: поточний chat_json НЕ вміє викликати gpt-5 без правки параметрів.

Запуск (робить платні виклики API — кілька центів):
    python -m scripts.model_eval
    python -m scripts.model_eval --models gpt-4o-mini,gpt-5-nano  # звузити список
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --- .env (лише OPENAI_API_KEY, значення не друкуємо) ------------------------------------
def _load_env() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


_load_env()

from newsroom.analyze.classifier import DEFAULT_PROMPT as CLS_PROMPT, parse_classification
from newsroom.analyze.spine import load_spine
from newsroom.editorial.draft import parse_draft
from newsroom.editorial.generator import DEFAULT_PROMPT as GEN_PROMPT, GenerationContext, build_material
from newsroom.llmutil import UsageRecord, record_completion, response_format_for, set_usage_recorder
from newsroom.promptutil import fill_prompt

SPINE = load_spine(Path(__file__).resolve().parent.parent / "config" / "taxonomy_spine.yaml")

# Кандидати. Базова (поточна) — перша. Порядок від дешевших до дорожчих орієнтовно.
DEFAULT_MODELS = [
    "gpt-4o-mini",   # ← поточна в пайплайні (базова лінія)
    "gpt-4.1-nano",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-5-mini",
    "gpt-4o",        # сильний «стеля»-орієнтир
]
BASELINE = "gpt-4o-mini"

# --- Золотий набір класифікації: по одній однозначній (за нашими визначеннями) новині на рубрику -
@dataclass(frozen=True)
class ClsItem:
    expected: str
    title: str
    text: str

GOLD_CLASSIFY = [
    ClsItem("war", "Генштаб: ЗСУ відбили 12 атак на Покровському напрямку",
            "За добу росіяни втратили близько 1420 військових. Тривають бої під Покровськом."),
    ClsItem("war", "Росіяни вдарили ракетами по центру Херсона, є поранені",
            "Унаслідок обстрілу пошкоджено житлові будинки, семеро людей дістали поранення."),
    ClsItem("defense", "Данія передасть Україні партію снарядів для артилерії",
            "Міноборони Данії оголосило новий пакет військової допомоги — снаряди калібру 155 мм."),
    ClsItem("mobilization", "Рада ухвалила зміни до закону про мобілізацію",
            "Депутати підтримали законопроєкт про уточнення правил призову та роботи ТЦК."),
    ClsItem("politics", "Кабмін призначив нового міністра енергетики",
            "Уряд затвердив кандидатуру на посаду міністра після відставки попередника."),
    ClsItem("geopolitics", "Зеленський і канцлер ФРН обговорили нові санкції проти РФ",
            "На переговорах у Берліні сторони узгодили посилення санкційного тиску та дипломатичну підтримку."),
    ClsItem("economy", "НБУ підвищив облікову ставку до 15%",
            "Правління Національного банку ухвалило рішення підняти ставку для стримування інфляції."),
    ClsItem("corruption", "НАБУ викрило схему розкрадання 200 млн грн у Міноборони",
            "Детективи повідомили про підозру посадовцям у зловживанні під час закупівель."),
    ClsItem("law_crime", "Суд обрав запобіжний захід підозрюваному у вбивстві",
            "Печерський суд відправив чоловіка під варту без права застави у справі про умисне вбивство."),
    ClsItem("society", "На трасі Київ–Одеса сталася ДТП, двоє загиблих",
            "Унаслідок зіткнення легковика й вантажівки загинули двоє людей, рух ускладнено."),
    ClsItem("tech_science", "Український стартап представив ШІ для медичної діагностики",
            "Команда розробила модель, що аналізує знімки й допомагає лікарям виявляти патології."),
    ClsItem("environment", "Синоптики попереджають про сильні зливи на заході країни",
            "Укргідрометцентр оголосив жовтий рівень небезпеки через грози та підтоплення."),
    ClsItem("sport", "Динамо перемогло Шахтар 2:1 у матчі УПЛ",
            "У центральному матчі туру переможний гол забив Ванат наприкінці зустрічі."),
    ClsItem("culture", "У Львові відкрився міжнародний кінофестиваль",
            "Програма триватиме тиждень, глядачам покажуть понад пів сотні стрічок з різних країн."),
]

# --- Генерація: фактбази-пастки, що перевіряють збереження суті ---------------------------
@dataclass(frozen=True)
class GenCase:
    key: str
    context: GenerationContext
    # (мітка, функція(body_lower, headline_lower)->bool) — автоматичні перевірки якості
    checks: list = field(default_factory=list)

def _no_markers(b: str, h: str) -> bool:
    t = b + " " + h
    return ("[заява" not in t) and ("[прогноз" not in t) and ("[рамка" not in t)

GEN_CASES = [
    GenCase(
        "eu_money",
        GenerationContext(
            title="ЄС ухвалив новий пакет допомоги Україні",
            rubrics=["geopolitics"], register="геополітика", status="confirmed",
            facts=[
                "Єврокомісія оголосила про виділення 3,3 млрд євро [рамка: до кінця 2026 року]",
                "Кошти призначені на оборонні потреби — виробництво снарядів і систем ППО",
                "[заява: голова Єврокомісії] кошти надходитимуть траншами",
            ],
        ),
        checks=[
            ("маркери прибрані", _no_markers),
            ("збережено «оборон»", lambda b, h: "оборон" in b),
            ("збережено рамку 2026", lambda b, h: ("2026" in b) or ("кінц" in b)),
            ("є атрибуція заяви", lambda b, h: any(w in b for w in ("за слов", "заяв", "за даними", "єврокоміс"))),
        ],
    ),
    GenCase(
        "intel_forecast",
        GenerationContext(
            title="Розвідка про можливу нову хвилю мобілізації в РФ",
            rubrics=["war"], register="війна", status="rumor",
            facts=[
                "[заява: Головне управління розвідки] РФ [прогноз] може оголосити нову хвилю мобілізації",
                "[прогноз] ідеться орієнтовно про 600 тисяч осіб",
                "Офіційного підтвердження від Кремля немає",
            ],
        ),
        checks=[
            ("маркери прибрані", _no_markers),
            ("атрибуція розвідці", lambda b, h: "розвід" in b),
            ("збережено модальність (може/планує/готує)",
             lambda b, h: any(w in b for w in ("мож", "план", "готу", "нібито", "за оцінк"))),
            ("не подано як доконаний факт",
             lambda b, h: not any(w in (b + h) for w in ("оголосила нову хвил", "провела мобіліз", "розпочала мобіліз"))),
        ],
    ),
    GenCase(
        "sport_clean",
        GenerationContext(
            title="Динамо перемогло Шахтар у матчі УПЛ",
            rubrics=["sport"], register="спорт", status="confirmed",
            facts=[
                "Матч завершився з рахунком 2:1 на користь Динамо",
                "Переможний гол забив Ванат на 88-й хвилині",
                "Це третя поспіль перемога Динамо в чемпіонаті",
            ],
        ),
        checks=[
            ("конкретний рахунок 2:1", lambda b, h: "2:1" in (b + h)),
            ("названо автора гола", lambda b, h: "ванат" in b),
            ("без клікбейту в заголовку",
             lambda b, h: not any(w in h for w in ("сенсац", "шок", "неймовір", "розгром віку"))),
        ],
    ),
]


# --- Виклик моделі з адаптацією параметрів під сімейство ----------------------------------
def _is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))

_records: list[UsageRecord] = []


def call_model(client, *, model: str, messages: list, op: str, max_out: int) -> str | None:
    """Виклик chat.completions з JSON-форматом. Для reasoning-моделей — інші параметри.
    Записує UsageRecord (op, model, токени, вартість) у _records через record_completion.
    Кидає виняток при недоступності моделі — викликач ловить і пропускає модель."""
    reasoning = _is_reasoning(model)
    base = dict(model=model, messages=messages, response_format=response_format_for(op))
    if reasoning:
        # reasoning-токени їдять бюджет → даємо запас понад корисний вихід
        base["max_completion_tokens"] = max_out + 1500
        base["reasoning_effort"] = "low"
    else:
        base["max_tokens"] = max_out
        base["temperature"] = 0.0

    # м'яка деградація: якщо API не приймає якийсь параметр — прибрати і повторити
    drop_order = ["reasoning_effort"]
    kw = dict(base)
    last_exc = None
    for _ in range(3):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            dropped = False
            for p in drop_order:
                if p in kw and p in msg:
                    kw.pop(p)
                    dropped = True
                    break
            if not dropped and "response_format" in kw and (
                    "response_format" in msg or "json_schema" in msg):
                # First preserve JSON mode when only strict schemas are unsupported.
                # Drop the format entirely only if the endpoint rejects JSON mode too.
                if kw["response_format"].get("type") == "json_schema":
                    kw["response_format"] = {"type": "json_object"}
                else:
                    kw.pop("response_format")
                dropped = True
            # gpt-5 інколи вимагає temperature=default: приберемо явну temperature
            if not dropped and "temperature" in kw and "temperature" in msg:
                kw.pop("temperature")
                dropped = True
            if not dropped and "max_tokens" in kw and "max_tokens" in msg:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
                dropped = True
            if dropped:
                last_exc = exc
                continue
            raise
        dur = int((time.monotonic() - t0) * 1000)
        content = resp.choices[0].message.content
        record_completion(op, model, resp, messages=messages, duration_ms=dur, content=content)
        return content
    if last_exc:
        raise last_exc
    return None


# --- Прогони -----------------------------------------------------------------------------
def run_classify(client, model: str) -> dict:
    correct = 0
    misses = []
    for item in GOLD_CLASSIFY:
        news = f"{item.title}\n{item.text}".strip()
        pred = None
        for _ in (1, 2):  # ретрай зіпсутого JSON, як у продакшн-класифікаторі
            content = call_model(client, model=model,
                                 messages=[{"role": "user",
                                            "content": fill_prompt(CLS_PROMPT, news_text=news)}],
                                 op="classify", max_out=512)
            parsed = parse_classification(content)
            if parsed is not None:
                pred = SPINE.resolve(parsed.rubrics[0]) if parsed.rubrics else None
                break
        ok = (pred == item.expected)
        correct += int(ok)
        if not ok:
            misses.append(f"{item.expected}→{pred or '∅'}")
    n = len(GOLD_CLASSIFY)
    return {"accuracy": correct / n, "correct": correct, "n": n, "misses": misses}


def run_generate(client, model: str) -> dict:
    passed = total = 0
    samples = []
    for case in GEN_CASES:
        material = build_material(case.context)
        content = call_model(client, model=model,
                             messages=[{"role": "user",
                                        "content": fill_prompt(GEN_PROMPT, feedback="", material=material)}],
                             op="generate", max_out=2048)
        draft = parse_draft(content)
        if draft is None:
            samples.append((case.key, "∅ (не розпарсено)", "", []))
            total += len(case.checks)
            continue
        b = (getattr(draft, "body", "") or "").lower()
        h = (getattr(draft, "headline", "") or "").lower()
        results = [(label, bool(fn(b, h))) for label, fn in case.checks]
        passed += sum(1 for _, ok in results if ok)
        total += len(results)
        samples.append((case.key, getattr(draft, "headline", ""), getattr(draft, "body", ""), results))
    return {"pass": passed, "total": total, "rate": (passed / total if total else 0.0), "samples": samples}


def cost_summary(model: str) -> dict:
    recs = [r for r in _records if r.model == model]
    cls = [r for r in recs if r.op == "classify"]
    gen = [r for r in recs if r.op == "generate"]
    def agg(rs):
        n = len(rs)
        cost = sum(r.cost_usd for r in rs)
        pt = sum(r.prompt_tokens for r in rs)
        ct = sum(r.completion_tokens for r in rs)
        dur = [r.duration_ms for r in rs if r.duration_ms is not None]
        return {"calls": n, "cost": cost, "cost_per_call": (cost / n if n else 0.0),
                "avg_in": (pt / n if n else 0), "avg_out": (ct / n if n else 0),
                "avg_ms": (sum(dur) / len(dur) if dur else 0)}
    return {"classify": agg(cls), "generate": agg(gen), "all": agg(recs)}


def load_production_cases(path: str | None) -> list[dict]:
    """Load human-reviewed JSONL cases exported from real production calls.

    A case stores the exact historical user prompt plus a small expected subset. Only
    reviewed=true rows run, so an old model output can never silently become its own gold.
    """
    if not path:
        return []
    rows: list[dict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not row.get("reviewed"):
            continue
        expected = row.get("expected")
        if not row.get("op") or not row.get("prompt") or not isinstance(expected, dict):
            raise ValueError(f"{path}:{lineno}: reviewed case needs op, prompt and expected")
        if not any(expected.get(key) for key in ("equals", "contains", "not_contains")):
            raise ValueError(f"{path}:{lineno}: reviewed case needs at least one assertion")
        row.setdefault("id", f"line-{lineno}")
        rows.append(row)
    return rows


def _subset_matches(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset_matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


def score_production_output(raw: str | None, case: dict) -> tuple[bool, str]:
    if not raw:
        return False, "empty"
    expected = case["expected"]
    try:
        actual = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False, "invalid_json"
    equals = expected.get("equals", {})
    if equals and not _subset_matches(actual, equals):
        return False, f"expected subset {equals!r}, got {actual!r}"
    haystack = raw.casefold()
    missing = [x for x in expected.get("contains", []) if str(x).casefold() not in haystack]
    forbidden = [x for x in expected.get("not_contains", []) if str(x).casefold() in haystack]
    if missing:
        return False, "missing: " + ", ".join(map(str, missing))
    if forbidden:
        return False, "forbidden: " + ", ".join(map(str, forbidden))
    return True, ""


def run_production_cases(client, model: str, cases: list[dict]) -> dict:
    passed = 0
    misses: list[str] = []
    by_op: dict[str, dict[str, int]] = {}
    for case in cases:
        op = str(case["op"])
        try:
            raw = call_model(
                client, model=model,
                messages=[{"role": "user", "content": str(case["prompt"])}],
                op=op, max_out=int(case.get("max_out") or 1024),
            )
            ok, reason = score_production_output(raw, case)
        except Exception as exc:  # noqa: BLE001 - keep the paid evaluation run alive
            ok, reason = False, f"call_error: {str(exc)[:300]}"
        bucket = by_op.setdefault(op, {"pass": 0, "total": 0})
        bucket["total"] += 1
        bucket["pass"] += int(ok)
        passed += int(ok)
        if not ok:
            misses.append(f"{case['id']} ({op}): {reason}"[:500])
    return {"pass": passed, "total": len(cases),
            "rate": passed / len(cases) if cases else 0.0,
            "by_op": by_op, "misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="кома-розділений список замість типового")
    ap.add_argument("--out", help="куди зберегти сирий JSON результатів")
    ap.add_argument("--dataset", help="JSONL з human-reviewed production-кейсами")
    args = ap.parse_args()
    # Windows-консоль часто cp1251 і падає на '¢'/кирилиці при друку — примусово UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    models = [m.strip() for m in args.models.split(",")] if args.models else list(DEFAULT_MODELS)
    production_cases = load_production_cases(args.dataset)
    _records.clear()
    set_usage_recorder(_records.append)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results: dict[str, dict] = {}
    for model in models:
        sys.stderr.write(f"[eval] {model}: класифікація…\n"); sys.stderr.flush()
        try:
            cls = run_classify(client, model)
        except Exception as exc:  # noqa: BLE001 — модель недоступна/несумісна
            results[model] = {"unavailable": str(exc)[:300]}
            sys.stderr.write(f"[eval] {model}: НЕДОСТУПНА → {str(exc)[:120]}\n")
            continue
        sys.stderr.write(f"[eval] {model}: генерація…\n"); sys.stderr.flush()
        try:
            gen = run_generate(client, model)
        except Exception as exc:  # noqa: BLE001
            gen = {"pass": 0, "total": 0, "rate": 0.0, "samples": [], "error": str(exc)[:200]}
        prod = run_production_cases(client, model, production_cases) if production_cases else None
        results[model] = {"classify_quality": cls, "generate_quality": gen,
                          "production_quality": prod, "cost": cost_summary(model)}

    # ЗБЕРІГАЄМО СИРІ ДАНІ ПЕРШИМИ — щоб збій друку не втратив платний прогін.
    total_spend = sum(r.cost_usd for r in _records)
    out = args.out or str(Path(os.environ.get("TEMP", ".")) / "model_eval_results.json")
    try:
        payload = {"results": results, "total_spend_usd": total_spend, "total_calls": len(_records)}
        Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        sys.stderr.write(f"[eval] сирий JSON збережено: {out}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[eval] не вдалося зберегти JSON: {exc}\n")

    try:
        _print_report(models, results)
    except Exception as exc:  # noqa: BLE001 — друк не має валити прогін; дані вже збережено
        sys.stderr.write(f"[eval] друк звіту впав ({exc}); дивись JSON вище\n")

    print(f"\nСУКУПНІ ВИТРАТИ ЦЬОГО ПРОГОНУ: ${total_spend:.4f} за {len(_records)} викликів")
    return 0


def _print_report(models: list[str], results: dict[str, dict]) -> None:
    live = [m for m in models if "unavailable" not in results.get(m, {})]
    dead = [m for m in models if "unavailable" in results.get(m, {})]

    print("\n" + "=" * 78)
    print("ПОРІВНЯННЯ МОДЕЛЕЙ — якість і вартість на реальних завданнях пайплайна")
    print("=" * 78)

    if dead:
        print("\nНедоступні / несумісні:")
        for m in dead:
            print(f"  ✗ {m}: {results[m]['unavailable']}")

    # базова вартість/виклик для індексу QPC
    base = results.get(BASELINE, {})
    base_cpc = base.get("cost", {}).get("classify", {}).get("cost_per_call") if "unavailable" not in base else None
    base_acc = base.get("classify_quality", {}).get("accuracy") if "unavailable" not in base else None
    base_qpc = (base_acc / (base_cpc * 100)) if (base_cpc and base_acc is not None) else None

    print("\nКЛАСИФІКАЦІЯ (14 новин, точність рубрики) + вартість:")
    print(f"{'модель':<16}{'точн.':>7}{'ток.in/out':>13}{'$/1000':>10}{'викл/1¢':>10}{'QPC':>8}{'×база':>8}{'мс':>7}")
    print("-" * 78)
    for m in live:
        r = results[m]
        acc = r["classify_quality"]["accuracy"]
        c = r["cost"]["classify"]
        cpc = c["cost_per_call"]
        per_cent = (0.01 / cpc) if cpc else float("inf")
        qpc = (acc / (cpc * 100)) if cpc else float("inf")
        idx = (qpc / base_qpc) if (base_qpc and qpc != float("inf")) else float("nan")
        tag = "  ← поточна" if m == BASELINE else ""
        print(f"{m:<16}{acc*100:>6.0f}%{c['avg_in']:>6.0f}/{c['avg_out']:<5.0f}"
              f"{cpc*1000:>10.4f}{per_cent:>10.0f}{qpc:>8.2f}{idx:>7.2f}x{c['avg_ms']:>7.0f}{tag}")

    print("\n  QPC = точність / (вартість одного виклику в центах) — «одиниць правильності за 1¢».")
    print("  ×база — у скільки разів більше якості-за-цент, ніж поточна gpt-4o-mini.")
    print("  викл/1¢ — скільки класифікацій купує 1 цент.")

    print("\nГЕНЕРАЦІЯ (3 фактбази-пастки, збереження суті/маркерів):")
    print(f"{'модель':<16}{'пройдено':>10}{'ставка':>9}{'$/пост':>10}{'мс':>8}")
    print("-" * 55)
    for m in live:
        g = results[m]["generate_quality"]
        gc = results[m]["cost"]["generate"]
        print(f"{m:<16}{g['pass']:>6}/{g['total']:<3}{g['rate']*100:>7.0f}%{gc['cost_per_call']*1000:>9.4f}‰{gc['avg_ms']:>8.0f}")
    print("  ($/пост показано ×1000 = за 1000 постів, у $)")

    print("\nПРОМАХИ КЛАСИФІКАЦІЇ (очікувано→отримано):")
    for m in live:
        misses = results[m]["classify_quality"]["misses"]
        print(f"  {m:<16}{', '.join(misses) if misses else '— (усе правильно)'}")

    if any(results[m].get("production_quality") for m in live):
        print("\nHUMAN-REVIEWED PRODUCTION CASES:")
        for m in live:
            p = results[m].get("production_quality")
            if not p:
                continue
            ops = ", ".join(f"{op}={v['pass']}/{v['total']}" for op, v in p["by_op"].items())
            print(f"  {m:<16}{p['pass']}/{p['total']} ({p['rate']*100:.1f}%)  {ops}")
            for miss in p["misses"][:10]:
                print(f"    ✗ {miss}")

    # якісні зразки генерації — надрукуємо для ручного читання
    print("\n" + "=" * 78)
    print("ЗРАЗКИ ГЕНЕРАЦІЇ (для ручної оцінки живості/точності)")
    print("=" * 78)
    for case_i, case in enumerate(GEN_CASES):
        print(f"\n### Кейс: {case.key}")
        for m in live:
            samples = results[m]["generate_quality"]["samples"]
            if case_i >= len(samples):
                continue
            key, headline, body, checks = samples[case_i]
            flags = " ".join(("✓" if ok else "✗") + label for label, ok in checks)
            print(f"\n[{m}]  {flags}")
            print(f"  ЗАГ: {headline}")
            for para in (body or "").split("\n"):
                if para.strip():
                    print(f"  {para.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
