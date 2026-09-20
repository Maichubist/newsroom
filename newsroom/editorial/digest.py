"""Digests (§: fold routine, individually-minor events into one post per window).

Individual routine posts — shelling, daily enemy losses, crime blotter, local
incidents, economy briefs — are буденність and drown the channel. Instead, an event
that matches a digest CATEGORY (its rubric, plus an optional text marker that picks
out the routine subset) is reserved (not posted individually) and all such events in
a window are combined into ONE deterministic digest per category — "Обстріли за ніч:
…", "Втрати ворога за день: …". Requiring a marker keeps important items in a rubric
(a major terror case, a milestone) out of the digest; a category with no markers
folds the whole rubric.

Detection, composition and the window/clock logic are pure and offline-tested;
publishing reuses the normal Publisher (stop button, stop list, Telegram) so a digest
is not a bypass. Categories/markers/windows are versioned config, calibrated without code.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

log = logging.getLogger("newsroom.editorial.digest")

DIGEST_STATE_KEY = "digest_last"          # system_state: {window_name: "YYYY-MM-DD"}


class DigestConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DigestWindow:
    name: str                              # internal key for idempotency (e.g. "night")
    suffix: str                            # human tail appended to a category name ("за ніч")
    start_hour: int
    end_hour: int
    publish_hour: int
    publish_minute: int = 0


@dataclass(frozen=True)
class DigestCategory:
    name: str                              # human title ("Обстріли", "Втрати ворога")
    hashtag: str
    rubrics: frozenset[str]                # spine slugs whose events this category folds
    markers: tuple[re.Pattern, ...] = ()   # text patterns picking the routine subset; empty = whole rubric

    def matches(self, rubric: str | None, text: str | None) -> bool:
        if (rubric or "").lower() not in self.rubrics:
            return False
        if not self.markers:
            return True                    # no marker required: the whole rubric is digest-worthy
        blob = text or ""
        return any(p.search(blob) for p in self.markers)


@dataclass(frozen=True)
class DigestConfig:
    tz: ZoneInfo
    categories: tuple[DigestCategory, ...]
    windows: tuple[DigestWindow, ...]


def load_digest_config(path: str | Path) -> DigestConfig:
    path = Path(path)
    if not path.exists():
        raise DigestConfigError(f"digest config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        windows = tuple(
            DigestWindow(name=str(w["name"]), suffix=str(w.get("suffix", "")).strip(),
                         start_hour=int(w["start_hour"]), end_hour=int(w["end_hour"]),
                         publish_hour=int(w["publish_hour"]), publish_minute=int(w.get("publish_minute", 0)))
            for w in (data.get("windows") or [])
        )
        categories = tuple(
            DigestCategory(
                name=str(c["name"]), hashtag=str(c.get("hashtag", "")).strip(),
                rubrics=frozenset(str(r).strip().lower() for r in (c.get("rubrics") or []) if str(r).strip()),
                markers=tuple(re.compile(str(m), re.IGNORECASE)
                              for m in (c.get("markers") or []) if str(m).strip()),
            )
            for c in (data.get("categories") or [])
        )
        if not categories:
            raise DigestConfigError("digest.yaml must define at least one category")
        return DigestConfig(tz=ZoneInfo(str(data.get("timezone", "Europe/Kyiv"))),
                            categories=categories, windows=windows)
    except (KeyError, TypeError, ValueError) as exc:
        raise DigestConfigError(f"bad digest config: {exc}") from exc


def _attack_blob(title: str | None, fact_base: object) -> str:
    """Title plus the fact texts of an event — the text a category's markers scan.
    Alerts often carry the marker words only in the body, so the bare title is not
    enough."""
    parts: list[str] = [title or ""]
    if isinstance(fact_base, dict):
        for f in (fact_base.get("facts") or []):
            if isinstance(f, dict) and f.get("text"):
                parts.append(str(f["text"]))
    return "\n".join(parts)


def digest_category(rubric: str | None, text: str | None, config: DigestConfig) -> DigestCategory | None:
    """The FIRST digest category an event belongs to (config order = precedence), or
    None if it is not a routine digest-class event. Precedence matters: a war event
    with an attack marker is 'Обстріли' (listed first), a war event about losses is
    'Втрати ворога'."""
    for cat in config.categories:
        if cat.matches(rubric, text):
            return cat
    return None


def compose_digest(name: str, titles: list[str]) -> tuple[str, str]:
    """Deterministic digest: a headline with the count and a bullet list of the event
    headlines. `name` already includes the window suffix ("Обстріли за ніч"). Returns
    (headline, body) for the HTML renderer."""
    headline = f"{name} ({len(titles)})"
    body = "\n".join(f"• {t.strip()}" for t in titles if t and t.strip())
    return headline, body


def window_range(window: DigestWindow, publish_date: dt.date, tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    """The [start, end) datetimes a window covers for a given publish date. A window
    whose start_hour > end_hour crosses midnight (night: 22:00 prev day → 06:00)."""
    end = dt.datetime.combine(publish_date, dt.time(window.end_hour), tz)
    if window.start_hour < window.end_hour:
        start = dt.datetime.combine(publish_date, dt.time(window.start_hour), tz)
    else:
        start = dt.datetime.combine(publish_date - dt.timedelta(days=1), dt.time(window.start_hour), tz)
    return start, end


def due_windows(now: dt.datetime, last_map: dict[str, str],
                config: DigestConfig) -> list[tuple[DigestWindow, dt.datetime, dt.datetime, str]]:
    """Windows whose publish time has passed today and weren't published yet today."""
    today = now.date()
    out = []
    for w in config.windows:
        publish_moment = dt.datetime.combine(today, dt.time(w.publish_hour, w.publish_minute), config.tz)
        if now >= publish_moment and last_map.get(w.name) != today.isoformat():
            start, end = window_range(w, today, config.tz)
            out.append((w, start, end, today.isoformat()))
    return out


def reserve_digests(session_factory, config: DigestConfig, *, window_hours: int = 48,
                    limit: int = 200) -> dict[str, int]:
    """Reserve postable digest-class events (curated='digest'), so they are not posted
    individually. NOT significance-gated on purpose: routine events fall below the bar,
    but aggregated in a digest they are worth publishing. Records the matched category."""
    from sqlalchemy import select

    from newsroom.models import Decision, Event, Publication

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    with session_factory() as s:
        have_pub = select(Publication.event_id).where(Publication.event_id.is_not(None))
        rows = s.execute(
            select(Event.id, Event.rubric, Event.title, Event.fact_base)
            # scan the MOST RECENT events, not the oldest: at scale (~900 events/day) an
            # id-asc scan never reaches today's events before curation must-publishes them
            .where(Event.status.in_(("reported", "confirmed", "rumor")),
                   Event.curated.is_(None), Event.duplicate_of.is_(None),
                   Event.id.not_in(have_pub),        # already posted individually -> don't digest it too
                   Event.first_seen_at >= cutoff, Event.title.is_not(None))
            .order_by(Event.first_seen_at.desc()).limit(limit)
        ).all()

    reserved = 0
    with session_factory() as s:
        for eid, rubric, title, fact_base in rows:
            # match markers against title + fact text, not the bare title: monitoring
            # channels post headline-less alerts with the marker words only in the body.
            cat = digest_category(rubric, _attack_blob(title, fact_base), config)
            if cat is None:
                continue
            ev = s.get(Event, eid)
            if ev is None or ev.curated is not None:
                continue
            ev.curated = "digest"
            s.add(Decision(entity_type="event", entity_id=str(eid), stage="edit",
                           decision="curate_digest", reason=cat.name, details={"category": cat.name}))
            reserved += 1
        s.commit()
    return {"reserved": reserved}


# --- DB helpers ---------------------------------------------------------------

def _load_last(session) -> dict[str, str]:
    from newsroom.models import SystemState

    row = session.get(SystemState, DIGEST_STATE_KEY)
    if row and isinstance(row.value, dict):
        return {str(k): str(v) for k, v in row.value.items()}
    return {}


def _store_last(session, last_map: dict[str, str]) -> None:
    from newsroom.models import SystemState

    row = session.get(SystemState, DIGEST_STATE_KEY)
    if row is None:
        session.add(SystemState(key=DIGEST_STATE_KEY, value=last_map))
    else:
        row.value = dict(last_map)
    session.flush()


def publish_due_digests(session_factory, config: DigestConfig, publisher, *,
                        now: dt.datetime | None = None) -> dict[str, int]:
    """For each due window: gather the reserved events in its range, group them by
    category (deterministic first-match), publish ONE combined digest per non-empty
    category via the Publisher, mark those events digested. Idempotent per day."""
    from sqlalchemy import select

    from newsroom.models import Decision, Event, Publication

    now = now or dt.datetime.now(config.tz)
    with session_factory() as s:
        last_map = _load_last(s)
    due = due_windows(now, last_map, config)

    stats = {"digests": 0, "events": 0}
    for window, start, end, date_str in due:
        with session_factory() as s:
            rows = s.execute(
                select(Event.id, Event.title, Event.rubric, Event.fact_base)
                .where(Event.curated == "digest",
                       Event.first_seen_at >= start, Event.first_seen_at < end)
                .order_by(Event.first_seen_at)
            ).all()

        # group the reserved events by their category (config order = precedence)
        buckets: "OrderedDict[str, tuple[DigestCategory, list]]" = OrderedDict()
        for eid, title, rubric, fact_base in rows:
            cat = digest_category(rubric, _attack_blob(title, fact_base), config)
            if cat is None:
                continue
            buckets.setdefault(cat.name, (cat, []))[1].append((eid, title))

        for cat, ev_rows in buckets.values():
            name = f"{cat.name} {window.suffix}".strip()
            headline, body = compose_digest(name, [str(t) for _, t in ev_rows])
            render = {"headline": headline, "body": body, "watching": "",
                      "hashtags": [cat.hashtag] if cat.hashtag else [], "source_links": [],
                      "is_rumor": False, "reported": False}
            with session_factory() as s:
                pub = Publication(event_id=None, channel="telegram", kind="digest", status="draft",
                                  headline=headline, body=f"{headline}\n\n{body}",
                                  features={"critic_ok": True, "is_rumor": False, "render": render})
                s.add(pub)
                s.flush()
                pub_id = pub.id
                s.commit()

            outcome = publisher.publish_one(pub_id)
            if outcome.published:
                with session_factory() as s:
                    for eid, _ in ev_rows:
                        ev = s.get(Event, eid)
                        if ev is not None:
                            ev.curated = "digested"
                    s.add(Decision(entity_type="publication", entity_id=str(pub_id), stage="publish",
                                   decision="digest_published", reason=cat.name,
                                   details={"events": len(ev_rows), "window": window.name}))
                    s.commit()
                stats["digests"] += 1
                stats["events"] += len(ev_rows)
            else:
                log.warning("digest blocked", extra={"category": cat.name, "window": window.name,
                                                     "reasons": outcome.reasons})

        # mark the window done for today (all categories processed), so we don't re-run it all day
        with session_factory() as s:
            last_map = _load_last(s)
            last_map[window.name] = date_str
            _store_last(s, last_map)
            s.commit()
    return stats
