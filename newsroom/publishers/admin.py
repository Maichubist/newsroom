"""Admin analytics console (architecture §10) — read-only.

Text commands from the service chat (TELEGRAM_ADMIN_CHAT_ID) that report on the
pipeline and the competitor/demand data. Two kinds:

  * canned reports — /stats /pub /demand /sources /top /media /event
  * /sql (alias /script) — the admin's own SELECT, run READ-ONLY.

Trust model: only messages from the admin chat id are ever dispatched (enforced by
the bot). /sql is defence-in-depth read-only: the statement must be a single
SELECT/WITH, a function blocklist rejects the superuser file/program/network tricks
(pg_read_file, COPY … PROGRAM, dblink, lo_*), it runs in a READ ONLY transaction
(any write raises), under a statement timeout, and only the first N rows are
returned. It can read DB data (which the admin owns) but cannot modify anything,
run shell commands, or read server files. Nothing here publishes.

Parsing, the SQL guard and formatting are pure/offline-tested; the reports are pg-tested.
"""
from __future__ import annotations

import html
import logging
import re

log = logging.getLogger("newsroom.publishers.admin")

TELEGRAM_MAX = 4096
_REPORT_CAP = 3800          # leave room for <pre> tags under the Telegram limit
_SQL_MAX_ROWS = 50
_SQL_TIMEOUT_MS = 5000

# Functions/utilities a read-only transaction does NOT stop (they read files, run
# programs or reach the network) — rejected before execution. Word-boundary matched.
_SQL_BLOCKLIST = (
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_ls_waldir", "pg_ls_logdir",
    "pg_stat_file", "lo_import", "lo_export", "lo_get", "lo_put", "dblink", "copy",
    "pg_sleep", "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "set_config", "current_setting", "pg_read_server_files",
)
_BLOCK_RE = re.compile(r"\b(" + "|".join(_SQL_BLOCKLIST) + r")\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def _esc(v) -> str:
    s = "" if v is None else str(v)
    return html.escape(s.replace("\n", " ").replace("\r", " "))


def _pre(body: str) -> str:
    """Monospace block for Telegram, truncated under the message limit."""
    body = body if len(body) <= _REPORT_CAP else body[:_REPORT_CAP] + "\n…(обрізано)"
    return f"<pre>{body}</pre>"


def _table(headers: list[str], rows: list[list], *, max_cell: int = 26) -> str:
    """Fixed-width text table (HTML-escaped, for a <pre> block)."""
    def cell(v):
        s = "" if v is None else str(v).replace("\n", " ").replace("\r", " ")
        return s if len(s) <= max_cell else s[: max_cell - 1] + "…"

    cols = [cell(h) for h in headers]
    data = [[cell(v) for v in r] for r in rows]
    widths = [len(c) for c in cols]
    for r in data:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "  ".join("-" * widths[i] for i in range(len(cols)))
    out = [line, sep]
    for r in data:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return html.escape("\n".join(out))


# --------------------------------------------------------------------------- #
# read-only SQL
# --------------------------------------------------------------------------- #

def is_readonly_sql(query: str | None) -> tuple[bool, str]:
    """(ok, reason). A single SELECT/WITH statement with no blocklisted function."""
    q = (query or "").strip()
    if not q:
        return False, "порожній запит"
    body = q.rstrip(";").strip()
    if ";" in body:
        return False, "лише один запит (без ';')"
    if not re.match(r"(?is)^\s*(select|with)\b", body):
        return False, "дозволені лише SELECT/WITH"
    hit = _BLOCK_RE.search(body)
    if hit:
        return False, f"заборонена функція: {hit.group(1)}"
    return True, ""


def run_readonly_sql(session_factory, query: str, *, max_rows: int = _SQL_MAX_ROWS,
                     timeout_ms: int = _SQL_TIMEOUT_MS) -> str:
    """Run the admin's SELECT in a READ ONLY transaction and format the result."""
    from sqlalchemy import text

    ok, reason = is_readonly_sql(query)
    if not ok:
        return f"❌ {reason}"
    body = query.strip().rstrip(";").strip()
    with session_factory() as s:
        try:
            # first statements in the tx: make it read-only + bounded
            s.execute(text("SET TRANSACTION READ ONLY"))
            s.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
            result = s.execute(text(body))
            headers = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
        except Exception as exc:  # noqa: BLE001 — report the DB error to the admin
            s.rollback()
            return f"❌ помилка: {_first_line(str(exc))}"
        finally:
            s.rollback()          # read-only: nothing to commit, just end the tx
    truncated = len(rows) > max_rows
    rows = [list(r) for r in rows[:max_rows]]
    if not rows:
        return "∅ 0 рядків"
    table = _table(headers, rows)
    note = f"\n…(показано перші {max_rows})" if truncated else f"\n{len(rows)} рядків"
    return _pre(table + html.escape(note))


def _first_line(s: str) -> str:
    return (s.splitlines() or [""])[0][:300]


# --------------------------------------------------------------------------- #
# canned reports
# --------------------------------------------------------------------------- #

def report_stats(session_factory) -> str:
    import datetime as dt

    from sqlalchemy import func, select

    from newsroom.models import Event, Item, Publication

    now = dt.datetime.now(dt.timezone.utc)
    with session_factory() as s:
        items = dict(s.execute(select(Item.status, func.count()).group_by(Item.status)).all())
        events = dict(s.execute(select(Event.status, func.count()).group_by(Event.status)).all())
        drafts = s.scalar(select(func.count()).select_from(Publication).where(Publication.status == "draft")) or 0
        pub_24h = s.scalar(select(func.count()).select_from(Publication).where(
            Publication.status == "published", Publication.published_at >= now - dt.timedelta(hours=24))) or 0
        curated = dict(s.execute(select(Event.curated, func.count()).where(Event.curated.is_not(None))
                                 .group_by(Event.curated)).all())
    lines = ["📊 Стан пайплайна", "",
             "items: " + ", ".join(f"{k}={v}" for k, v in sorted(items.items())),
             "events: " + ", ".join(f"{k}={v}" for k, v in sorted(events.items())),
             "curated: " + (", ".join(f"{k}={v}" for k, v in sorted(curated.items())) or "—"),
             f"чернеток: {drafts}", f"опубліковано за 24г: {pub_24h}"]
    return _pre(html.escape("\n".join(lines)))


def report_publications(session_factory, *, limit: int = 10) -> str:
    import datetime as dt

    from sqlalchemy import select

    from newsroom.models import Publication

    with session_factory() as s:
        rows = s.execute(
            select(Publication.id, Publication.event_id, Publication.headline, Publication.published_at)
            .where(Publication.status == "published")
            .order_by(Publication.published_at.desc()).limit(limit)
        ).all()
    if not rows:
        return "Ще немає опублікованих постів."
    data = [[pid, eid, (h or "")[:40],
             (at.astimezone(dt.timezone.utc).strftime("%m-%d %H:%M") if at else "—")]
            for pid, eid, h, at in rows]
    return _pre(_table(["pub", "event", "headline", "utc"], data))


def report_topics(session_factory, *, limit: int = 25) -> str:
    from newsroom.analyze.topics import load_hot_topics_detail

    with session_factory() as s:
        topics = load_hot_topics_detail(s)
    if not topics:
        return ("Гарячих тем ще нема (потрібні події з ключовими словами; "
                "TOPICS_ENABLED + класифікатор).")
    rows = [[t.get("topic", ""), t.get("events", 0), t.get("posts", 0), f"{t.get('heat', 0):.2f}"]
            for t in topics[:limit]]
    return _pre("Гарячі теми (з постів конкурентів)\n\n"
                + _table(["тема", "події", "пости", "heat"], rows))


def report_demand(session_factory) -> str:
    from newsroom.analyze.demand import load_demand

    with session_factory() as s:
        demand = load_demand(s)
    if not demand:
        return ("Індексу попиту ще нема (потрібні зібрані метрики конкурентів; "
                "DEMAND_METRICS_ENABLED + Telethon).")
    rows = [[r, f"{v:.2f}"] for r, v in sorted(demand.items(), key=lambda kv: kv[1], reverse=True)]
    return _pre("Попит за рубриками (0..1)\n\n" + _table(["рубрика", "індекс"], rows))


def report_sources(session_factory, *, limit: int = 30) -> str:
    from sqlalchemy import func, select

    from newsroom.models import Source, SourceMetric

    with session_factory() as s:
        by_tier = dict(s.execute(
            select(Source.tier, func.count()).where(Source.kind == "telegram").group_by(Source.tier)
        ).all())
        # latest subscriber count per telegram source
        rows = s.execute(
            select(Source.name, Source.tier, SourceMetric.subscribers, func.max(SourceMetric.measured_at))
            .join(SourceMetric, SourceMetric.source_id == Source.id)
            .where(Source.kind == "telegram")
            .group_by(Source.name, Source.tier, SourceMetric.subscribers)
            .order_by(SourceMetric.subscribers.desc().nullslast()).limit(limit)
        ).all()
    tiers = ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())) or "—"
    head = f"Джерела (telegram): {tiers}"
    if not rows:
        return _pre(html.escape(head + "\n\nПідписники ще не зібрані (source_metrics порожня)."))
    seen, data = set(), []
    for name, tier, subs, _at in rows:
        if name in seen:
            continue
        seen.add(name)
        data.append([name[:24], tier, subs])
    return _pre(html.escape(head) + "\n" + _table(["канал", "тір", "підписники"], data))


def report_top(session_factory, *, days: int = 7, limit: int = 12) -> str:
    """Top competitor posts by reach-normalised engagement in the window."""
    import datetime as dt

    from sqlalchemy import select

    from newsroom.analyze.demand import engagement_rate
    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric
    from newsroom.publishers.metrics import MessageStats

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    with session_factory() as s:
        subs: dict[int, int] = {}
        for sid, n, _at in s.execute(
            select(SourceMetric.source_id, SourceMetric.subscribers, SourceMetric.measured_at)
            .order_by(SourceMetric.source_id, SourceMetric.measured_at.desc())
        ).all():
            if sid not in subs and n:
                subs[sid] = int(n)
        rows = s.execute(
            select(ItemMetric.item_id, ItemMetric.views, ItemMetric.reactions, ItemMetric.forwards,
                   ItemMetric.measured_at, Source.id, Source.name, Event.rubric, Item.title, Item.text)
            .join(Item, Item.id == ItemMetric.item_id)
            .join(Source, Source.id == Item.source_id)
            .join(EventItem, EventItem.item_id == Item.id)
            .join(Event, Event.id == EventItem.event_id)
            .where(ItemMetric.measured_at >= cutoff)
            .order_by(ItemMetric.item_id, ItemMetric.measured_at.desc())
        ).all()
    seen, scored = set(), []
    for item_id, views, reactions, forwards, _at, sid, sname, rubric, title, text in rows:
        if item_id in seen:
            continue
        seen.add(item_id)
        rate = engagement_rate(MessageStats(views=views, reactions=reactions, forwards=forwards), subs.get(sid))
        if rate is None:
            continue
        label = (title or (text or "").split("\n", 1)[0] or "")[:34]
        scored.append((rate, sname, rubric or "—", label))
    if not scored:
        return f"Нема виміряних постів конкурентів за {days}д (потрібні item_metrics + підписники)."
    scored.sort(reverse=True)
    data = [[f"{r:.4f}", sname[:16], rubric, label] for r, sname, rubric, label in scored[:limit]]
    return _pre(f"Топ постів конкурентів ({days}д, engagement/підписник)\n\n"
                + _table(["eng", "канал", "рубрика", "заголовок"], data))


def report_media(session_factory) -> str:
    from sqlalchemy import func, select

    from newsroom.models import Item, MediaAsset, Source

    with session_factory() as s:
        rows = s.execute(
            select(Source.kind, func.count(), func.count(MediaAsset.storage_key), func.count(MediaAsset.purged_at))
            .join(Item, Item.id == MediaAsset.item_id).join(Source, Source.id == Item.source_id)
            .group_by(Source.kind)
        ).all()
        tg_pending = s.scalar(select(func.count()).select_from(MediaAsset)
                              .join(Item, Item.id == MediaAsset.item_id).join(Source, Source.id == Item.source_id)
                              .where(Source.kind == "telegram", MediaAsset.source_ref.is_not(None),
                                     MediaAsset.storage_key.is_(None))) or 0
    data = [[k, total, dl, purged] for k, total, dl, purged in rows]
    return _pre("Медіа (assets / завантажено / видалено)\n\n"
                + _table(["джерело", "усього", "завантаж.", "видалено"], data)
                + html.escape(f"\n\nТГ-медіа очікує завантаження (Telethon): {tg_pending}"))


def report_event(session_factory, event_id: int) -> str:
    from sqlalchemy import select

    from newsroom.models import Decision, Event

    with session_factory() as s:
        ev = s.get(Event, event_id)
        if ev is None:
            return f"Подію {event_id} не знайдено."
        decisions = s.execute(
            select(Decision.stage, Decision.decision, Decision.reason, Decision.created_at)
            .where(Decision.entity_type == "event", Decision.entity_id == str(event_id))
            .order_by(Decision.id)
        ).all()
    head = (f"event {event_id}: {ev.status}/{ev.risk_level or '?'}/{ev.rubric or '?'} "
            f"sig={ev.significance if ev.significance is not None else '—'} "
            f"curated={ev.curated or '—'} dup_of={ev.duplicate_of or '—'}\n"
            f"«{(ev.title or '')[:80]}»")
    if not decisions:
        return _pre(html.escape(head + "\n\n(рішень нема)"))
    data = [[st, d, (r or "")[:40]] for st, d, r, _at in decisions]
    return _pre(html.escape(head) + "\n\n" + _table(["stage", "decision", "reason"], data))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

HELP = (
    "🛠 Команди адмін-консолі\n"
    "/stats — стан пайплайна\n"
    "/pub [N] — останні опубліковані пости\n"
    "/topics [N] — гарячі теми з постів конкурентів\n"
    "/demand — індекс попиту за рубриками\n"
    "/sources — джерела-конкуренти й підписники\n"
    "/top [N] — топ постів конкурентів за залученістю\n"
    "/media — стан медіа й ТГ-завантаження\n"
    "/event &lt;id&gt; — журнал рішень події\n"
    "/sql &lt;SELECT…&gt; — власний read-only запит (alias /script)\n"
    "/help — ця довідка"
)


def parse_command(text: str | None) -> tuple[str, str] | None:
    """('cmd', 'args') for a '/command …' message, else None. Strips a @botname suffix."""
    if not text:
        return None
    t = text.strip()
    if not t.startswith("/"):
        return None
    head, _, rest = t.partition(" ")
    cmd = head[1:].split("@", 1)[0].strip().lower()
    return (cmd, rest.strip()) if cmd else None


class AdminConsole:
    def __init__(self, session_factory):
        self.sf = session_factory

    def handle(self, text: str | None) -> str | None:
        """Dispatch a '/command'. Returns reply text, or None if `text` is not a command."""
        parsed = parse_command(text)
        if parsed is None:
            return None
        cmd, args = parsed
        try:
            return self._dispatch(cmd, args)
        except Exception as exc:  # noqa: BLE001 — never let a report crash the bot
            log.exception("admin command failed", extra={"cmd": cmd})
            return f"❌ помилка команди /{cmd}: {_first_line(str(exc))}"

    def _dispatch(self, cmd: str, args: str) -> str:
        if cmd in ("help", "start"):
            return HELP
        if cmd == "stats":
            return report_stats(self.sf)
        if cmd == "pub":
            return report_publications(self.sf, limit=_int(args, 10, lo=1, hi=30))
        if cmd == "topics":
            return report_topics(self.sf, limit=_int(args, 25, lo=1, hi=50))
        if cmd == "demand":
            return report_demand(self.sf)
        if cmd == "sources":
            return report_sources(self.sf)
        if cmd == "top":
            return report_top(self.sf, limit=_int(args, 12, lo=1, hi=30))
        if cmd == "media":
            return report_media(self.sf)
        if cmd == "event":
            eid = _int(args, 0, lo=0, hi=2**62)
            return report_event(self.sf, eid) if eid else "Вкажи id події: /event 123"
        if cmd in ("sql", "script"):
            return run_readonly_sql(self.sf, args) if args else "Напиши запит: /sql SELECT …"
        return f"Невідома команда /{cmd}. /help — список."


def _int(s: str, default: int, *, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int((s or "").strip().split()[0])))
    except (ValueError, IndexError):
        return default
