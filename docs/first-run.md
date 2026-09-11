# Перший запуск

Покрокове введення в експлуатацію: спершу «тільки збір», потім поетапно вмикаємо
аналітику й, нарешті, публікацію в закритий тестовий канал. Кожна ризикова стадія
вмикається окремим прапорцем — доти нічого нікуди не постить.

Правило безпеки на весь час: **ніколи не комітимо** `.env`, файли сесій Telethon
(`*.session`), ключі API. Файл сесії Telethon = повний доступ до акаунта.

---

## 0. Передумови

- Docker (для PostgreSQL 16 + pgvector) і Python 3.11+.
- Віртуальне середовище й залежності:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install -e .
```

- Конфіг середовища: скопіювати приклад і заповнити його **у `.env`** (не в `.env.example`):

```bash
cp .env.example .env
```

За замовчуванням у `.env` уже правильний `NEWSROOM_DATABASE_URL` під docker-compose,
і **всі ризикові прапорці вимкнені** (`*_ENABLED=false`), `SHADOW_MODE=false`.

---

## 1. Підняти базу

```bash
docker compose up -d db
```

Перевірити, що вона готова (health — `healthy`):

```bash
docker compose ps
```

Схема створюється автоматично при першому підключенні (`init_db`: `CREATE EXTENSION
vector` + `create_all`). Окремих міграцій немає.

---

## 2. Разовий збір RSS (димовий тест)

Один прохід по всіх активних RSS-джерелах із `config/sources.yaml`. Синхронізує
джерела в базу, збирає, друкує звіт про стан джерел. Нічого не публікує.

```bash
.venv/Scripts/python -m newsroom.runner
```

Перевірити, що елементи з'явилися в базі:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select count(*) from items;"
```

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select s.name, count(i.id) from sources s left join items i on i.source_id=s.id group by 1 order by 2 desc;"
```

Якщо якесь джерело «нездорове» — у логах буде `source unhealthy` з причиною.

---

## 3. Сервіс у режимі «тільки збір»

Довготривалий демон: планувальник RSS у реальному часі (+ Telegram, якщо ввімкнено).
З типовим `.env` усі аналітичні стадії вимкнені — це чистий збір.

```bash
.venv/Scripts/python -m newsroom.service
```

У логах при старті видно, що кожна стадія `disabled (... off)`. Зупинка — `Ctrl+C`.

---

## 4. Додати збір із Telegram

1. У `.env` заповнити `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` (з https://my.telegram.org).
2. Разово, **інтерактивно**, залогінитися й підтягнути канали з підписок акаунта
   (створить файл сесії за шляхом `TELEGRAM_SESSION_PATH`):

```bash
.venv/Scripts/python scripts/import_telegram_channels.py
```

   Це «сухий» перегляд. Щоб дописати нові канали в `config/sources.yaml`:

```bash
.venv/Scripts/python scripts/import_telegram_channels.py --write
```

   Далі вручну переглянути `sources.yaml`: прибрати особисті канали, проставити
   `tier: official` / `origin: world`, де доречно.
3. Увімкнути збір: `COLLECTOR_TELEGRAM_ENABLED=true` у `.env`, перезапустити сервіс.

> Не запускати ту саму сесію Telethon локально й у хмарі одночасно — Telegram може
> її анулювати. Для розробки — окрема сесія/акаунт.

---

## 5. Поетапно вмикати аналітику

Потрібен `OPENAI_API_KEY` у `.env` (ембеддинги + LLM). Вмикати **по одному
прапорцю**, перезапускаючи сервіс і спостерігаючи логи/базу. Рекомендований порядок
збігається з конвеєром:

| Прапорець | Що вмикає |
|---|---|
| `VERIFY_ENABLED` | фільтр сигналу → класифікація → кластеризація → ворота ризику (події, статуси) |
| `FACTBASE_ENABLED` | спільна фактбаза події (одна точка зору) |
| `FACTCHECK_ENABLED` | твердження → докази → вердикт |
| `STORY_UPDATES_ENABLED` | тип оновлення сюжету + `current_summary`/`story_versions` |
| `EDITORIAL_ENABLED` | генератор + критик → **чернетки** постів |
| `REPUTATION_ENABLED` | журнал репутації джерел |

Після `EDITORIAL_ENABLED` над подіями народжуються чернетки. Перевірити:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select id,status,kind,left(headline,60) from publications order by id desc limit 20;"
```

Усе ще **нічого не публікується** — це рядки `status='draft'`. Кожне рішення видно
в журналі:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select stage,decision,reason,created_at from decisions order by id desc limit 30;"
```

---

## 6. Медіа

1. Встановити декодер зображень (опційна залежність):

```bash
.venv/Scripts/pip install -e ".[media]"
```

2. У `.env`: `MEDIA_DOWNLOAD_ENABLED=true`, `MEDIA_CHECK_ENABLED=true`,
   `MEDIA_MODERATION_ENABLED=true` (модерація потребує vision-моделі OpenAI).

Порядок: завантаження медіа (+pHash) → перевірка повтору → візуальна модерація.
Картинка додається до поста, **лише якщо пройшла і перевірку на повтор, і
модерацію**; інакше — текст без медіа.

---

## 7. Тіньовий режим (обов'язковий крок перед бойовим)

Публікація в **закритий тестовий канал** 1–2 тижні. Спершу створити приватний
канал, додати бота адміном, узяти його `chat_id`.

У `.env`:

- `TELEGRAM_BOT_TOKEN=...` (бот-адмін тестового каналу);
- `TELEGRAM_SHADOW_CHANNEL_CHAT_ID=...` (тестовий канал);
- `TELEGRAM_ADMIN_CHAT_ID=...` (службовий чат нагляду — сповіщення й кнопки);
- `SHADOW_MODE=true`;
- `PUBLISH_ENABLED=true`.

Перезапустити сервіс. Тепер критик-схвалені чернетки публікуються в **тестовий**
канал, кожна проходить ворота (стоп-кнопка, стоп-лист, ліміти, сплеск). Ризикові
пости й чутки дають сповіщення в службовий чат із кнопками **Відкликати** / **Стоп**.

**Стоп-кнопка** (повна зупинка публікацій), вручну:

```bash
.venv/Scripts/python -c "from dotenv import load_dotenv; load_dotenv(); from newsroom.db import make_engine, make_session_factory; from newsroom.publishers.gate import set_publishing_stopped; sf=make_session_factory(make_engine()); s=sf(); set_publishing_stopped(s, True, reason='manual'); s.commit(); print('stopped')"
```

(замінити `True` на `False`, щоб відновити).

**Звіт про критерії виходу** з тіньового режиму (обсяг, дні, порушення стоп-листа,
частка дублів; людський критерій «жодного пропущеного фейку» перевіряється окремо):

```bash
.venv/Scripts/python -c "from dotenv import load_dotenv; load_dotenv(); from newsroom.db import make_engine, make_session_factory; from newsroom.analyze.stoplist import load_stoplist; from newsroom.publishers.shadow import load_shadow_criteria, shadow_report; sf=make_session_factory(make_engine()); c=load_shadow_criteria('config/shadow.yaml'); r=shadow_report(sf, criteria=c, stoplist_rules=load_stoplist('config/stoplist.yaml')); print(r)"
```

Критерії налаштовуються в `config/shadow.yaml`.

---

## 8. Бойовий режим

Коли `auto_criteria_met=True` і людина підтвердила відсутність пропущених фейків:

- `SHADOW_MODE=false` (пости йдуть у `TELEGRAM_CHANNEL_CHAT_ID`);
- за потреби відкоригувати пороги (`config/risk.yaml`, `config/limits.yaml`).

Стоп-кнопка й службовий чат працюють так само. `PUBLISH_ENABLED=false` — миттєвий
жорсткий стоп усієї публікації.

---

## Метрики (опційно)

`METRICS_ENABLED=true` вмикає збір перегляди/реакції/пересилання постів і кількість
підписників каналу. Читаються **тим самим Telethon-акаунтом**, що й збір, тож
потрібні `COLLECTOR_TELEGRAM_ENABLED=true` і щоб цей акаунт був **підписаний на
канал публікації** (тестовий або бойовий). Працює і в тіньовому режимі.

Останні знімки метрик:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select publication_id, views, forwards, measured_at from publication_metrics order by id desc limit 20;"
```

---

## Швидка діагностика

Скільки чого в базі:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select 'items' t, count(*) from items union all select 'events', count(*) from events union all select 'stories', count(*) from stories union all select 'claims', count(*) from claims union all select 'publications', count(*) from publications;"
```

Останні опубліковані / заблоковані пости:

```bash
docker compose exec db psql -U newsroom -d newsroom -c "select entity_id,decision,reason,created_at from decisions where stage='publish' order by id desc limit 20;"
```

Прапорці, які зараз впливають на конвеєр, — у `.env`; їхній опис — там же, поряд із
кожним значенням.
