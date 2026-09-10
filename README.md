# newsroom

Автономна мультиканальна новинна платформа для української аудиторії.
Опис: [CLAUDE.md](CLAUDE.md) · [docs/architecture.md](docs/architecture.md) · редакційні правила: [config/charter.md](config/charter.md).

**Поточний етап: 1а — фундамент** (модель даних, збирачі, стан джерел). Критерії — architecture §14.

## Стек
Python 3.11+, PostgreSQL 16 + pgvector, SQLAlchemy 2, feedparser + trafilatura (RSS),
Telethon (читання Telegram), Telegram Bot API (публікація), OpenAI (з етапу 1б), Docker, pytest.

## Локальний запуск

```bash
# 1. Залежності
python -m venv .venv && . .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Секрети
cp .env.example .env          # заповнити TELEGRAM_BOT_TOKEN тощо; .env не комітиться

# 3. База (Postgres 16 + pgvector)
docker compose up -d

# 4. Схема (idempotent, без Alembic — clean slate)
python -c "from newsroom.db import make_engine, init_db; init_db(make_engine())"

# 5. Збір
python -m newsroom.runner     # разовий прохід по всіх RSS (синхронізує джерела з конфіга)
python -m newsroom.service    # демон: RSS за poll_interval + Telegram realtime (за прапорцем)
```
Обидва нічого не публікують. `service` вмикає Telegram-збір лише за
`COLLECTOR_TELEGRAM_ENABLED=true` з кредами акаунта і каналами в `config/sources.yaml`.

## Тести
```bash
pytest                 # усе; pg-тести піднімають ефемерний Postgres (потрібен Docker)
pytest -m "not pg"     # лише офлайн-логіка, без Docker
```
Тести не ходять у мережу. Схему/pgvector перевіряють на ефемерному контейнері (testcontainers).

## Статус реалізації

| Мілстоун | Стан |
|---|---|
| M1 каркас (compose, pyproject, JSON-логи, .env) | ✅ |
| M2 схема §5 (усі таблиці) + `init_db` + pg-тести | ✅ |
| M3 `config/sources.yaml` (25 фідів, 3 official) + лоадер + тести | ✅ |
| M4 єдиний елемент (`RawItem`) + `content_hash`/`simhash` + дедуп | ✅ |
| M5 RSS-збирач + ідемпотентний upsert + моніторинг стану джерел | ✅ |
| M7a collect-only runner (`python -m newsroom.runner`) | ✅ |
| M6 Telegram-збирач: логіка (маппер, альбоми, forwarded_from, edits/deletes, backfill, FloodWait) + тести | ✅ |
| M7b жива петля: RSS-планувальник за `poll_interval` + оркестрація Telegram-збирача (backfill/handlers/album flush) + тести; демон `python -m newsroom.service` | ✅ |
| M7c перенос Telegram-паблішера з news_bot (gated) | ⏳ |

M6/M7b постачають усю **логіку** збору (RSS due-планувальник, Telegram backfill,
realtime-хендлери, склейка альбомів, FloodWait) — тестовано на фейкових клієнтах,
без мережі. Жива Telethon-петля (`TelegramCollector.start`) вмикається прапорцем
`COLLECTOR_TELEGRAM_ENABLED` разом із кредами акаунта і списком каналів у `sources.yaml`.

## Свідомі відхилення від architecture (за рішенням власника)
- **Без Alembic**: схема створюється `create_all` з чистого листа; міграції додамо, коли схема почне еволюціонувати між етапами (§11/§14 передбачали Alembic).
- **Публікація дозволена**: 1а не суто «тільки збір» — Telegram-паблішер (Bot API, той самий канал) переноситься з news_bot і під'єднується; фактична публікація почнеться, коли редакційний конвеєр (1б–1в) даватиме готові пости. Майстер-вимикач — `PUBLISH_ENABLED` у `.env`.
