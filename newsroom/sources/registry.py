"""Sync the sources config into the DB and report source health (architecture §12,
готовності §14: "стан кожного джерела видно"). Health fields are never reset by a
config sync — only collectors update them."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.config.sources import SourceConfig
from newsroom.models import Source


def sync_sources(session: Session, configs: list[SourceConfig]) -> dict[str, int]:
    """Upsert sources by (kind, handle_or_url). Descriptive fields are updated;
    health/state (last_success_at, consecutive_failures, …) is left untouched."""
    created = updated = 0
    for cfg in configs:
        src = session.execute(
            select(Source).where(Source.kind == cfg.kind, Source.handle_or_url == cfg.handle_or_url)
        ).scalar_one_or_none()
        if src is None:
            session.add(Source(
                kind=cfg.kind, handle_or_url=cfg.handle_or_url, name=cfg.name,
                origin=cfg.origin, lang=cfg.lang, region=cfg.region,
                tier=cfg.tier, is_official=cfg.is_official,
                active=cfg.active, poll_interval=cfg.poll_interval,
            ))
            created += 1
        else:
            src.name = cfg.name
            src.origin = cfg.origin
            src.lang = cfg.lang
            src.region = cfg.region
            src.tier = cfg.tier
            src.is_official = cfg.is_official
            src.active = cfg.active
            src.poll_interval = cfg.poll_interval
            updated += 1
    session.flush()
    return {"created": created, "updated": updated}


def health_report(session: Session) -> list[dict]:
    """One row per source, worst-health first (most consecutive failures)."""
    rows = session.execute(select(Source).order_by(Source.name)).scalars().all()
    report = [{
        "id": s.id,
        "name": s.name,
        "kind": s.kind,
        "tier": s.tier,
        "active": s.active,
        "last_success_at": s.last_success_at.isoformat() if s.last_success_at else None,
        "consecutive_failures": s.consecutive_failures,
        "last_error": s.last_error,
        "healthy": bool(s.last_success_at) and (s.consecutive_failures or 0) == 0,
    } for s in rows]
    report.sort(key=lambda r: (-(r["consecutive_failures"] or 0), r["name"]))
    return report
