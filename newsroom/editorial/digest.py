"""Attacks digest (§: fold routine shelling/strike events into one post per window).

Individual drone/missile posts are буденність and drown the channel. Instead, an
event that looks like an attack is reserved (not posted individually) and all such
events in a window (overnight, daytime) are combined into ONE deterministic digest
post — "Обстріли за ніч: …". Detection, composition and the window/clock logic are
pure and offline-tested; publishing reuses the normal Publisher (stop button, stop
list, Telegram) so a digest is not a bypass.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

log = logging.getLogger("newsroom.editorial.digest")

DIGEST_STATE_KEY = "digest_last"          # system_state: {window_name: "YYYY-MM-DD"}


class DigestConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DigestWindow:
    name: str
    start_hour: int
    end_hour: int
    publish_hour: int
    publish_minute: int = 0


@dataclass(frozen=True)
class DigestConfig:
    tz: ZoneInfo
    rubrics: frozenset[str]
    markers: tuple[re.Pattern, ...]
    windows: tuple[DigestWindow, ...]


def load_digest_config(path: str | Path) -> DigestConfig:
    path = Path(path)
    if not path.exists():
        raise DigestConfigError(f"digest config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        windows = tuple(
            DigestWindow(name=str(w["name"]), start_hour=int(w["start_hour"]),
                         end_hour=int(w["end_hour"]), publish_hour=int(w["publish_hour"]),
                         publish_minute=int(w.get("publish_minute", 0)))
            for w in (data.get("windows") or [])
        )
        return DigestConfig(
            tz=ZoneInfo(str(data.get("timezone", "Europe/Kyiv"))),
            rubrics=frozenset(str(r).lower() for r in (data.get("rubrics") or [])),
            markers=tuple(re.compile(str(m), re.IGNORECASE) for m in (data.get("markers") or []) if str(m).strip()),
            windows=windows,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DigestConfigError(f"bad digest config: {exc}") from exc


def is_attack(rubric: str | None, text: str | None, config: DigestConfig) -> bool:
    """True when the event is a routine attack/shelling report: an attack-class
    rubric AND a marker in the text. Both required, so a policy piece about the war
    (rubric war, no attack marker) is not swept into the shelling digest."""
    if (rubric or "").lower() not in config.rubrics:
        return False
    blob = text or ""
    return any(p.search(blob) for p in config.markers)


def compose_digest(name: str, titles: list[str]) -> tuple[str, str]:
    """Deterministic digest: a bold-ready headline with the count and a bullet list
    of the event headlines. Returns (headline, body) for the HTML renderer."""
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


def reserve_attacks(session_factory, config: DigestConfig, *, window_hours: int = 48,
                    limit: int = 200) -> dict[str, int]:
    """Reserve postable attack events for the digest (curated='digest'), so they are
    not posted individually. NOT significance-gated on purpose: routine attacks fall
    below the significance bar, but aggregated in a digest they are worth publishing."""
    from sqlalchemy import select

    from newsroom.models import Decision, Event

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    with session_factory() as s:
        rows = s.execute(
            select(Event.id, Event.rubric, Event.title)
            .where(Event.status.in_(("reported", "confirmed", "rumor")),
                   Event.curated.is_(None), Event.duplicate_of.is_(None),
                   Event.first_seen_at >= cutoff, Event.title.is_not(None))
            .order_by(Event.id).limit(limit)
        ).all()

    reserved = 0
    with session_factory() as s:
        for eid, rubric, title in rows:
            if not is_attack(rubric, title, config):
                continue
            ev = s.get(Event, eid)
            if ev is None or ev.curated is not None:
                continue
            ev.curated = "digest"
            s.add(Decision(entity_type="event", entity_id=str(eid), stage="edit",
                           decision="curate_digest", reason="attack", details={}))
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
    """For each due window: gather the reserved attack events in its range, publish one
    combined digest via the Publisher, mark those events digested. Idempotent per day."""
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
                select(Event.id, Event.title)
                .where(Event.curated == "digest",
                       Event.first_seen_at >= start, Event.first_seen_at < end)
                .order_by(Event.first_seen_at)
            ).all()

        if rows:
            headline, body = compose_digest(window.name, [str(t) for _, t in rows])
            render = {"headline": headline, "body": body, "watching": "",
                      "hashtags": ["#обстріли"], "source_links": [], "is_rumor": False, "reported": False}
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
                    for eid, _ in rows:
                        ev = s.get(Event, eid)
                        if ev is not None:
                            ev.curated = "digested"
                    s.add(Decision(entity_type="publication", entity_id=str(pub_id), stage="publish",
                                   decision="digest_published", reason=window.name,
                                   details={"events": len(rows)}))
                    s.commit()
                stats["digests"] += 1
                stats["events"] += len(rows)
            else:
                log.warning("digest blocked", extra={"window": window.name, "reasons": outcome.reasons})

        # mark the window done for today either way, so we don't re-run it all day
        with session_factory() as s:
            last_map = _load_last(s)
            last_map[window.name] = date_str
            _store_last(s, last_map)
            s.commit()
    return stats
