"""DB maintenance — retention pruning for the high-volume `decisions` log.

Observe-mode dedup logs every clean/near-miss decision, which is what makes the data
useful for calibration but also what makes `decisions` grow fastest. This prunes only
those high-volume dedup stages beyond a retention window; the publish/verify audit trail
is left intact. Meant to be run periodically (a scheduled job), not inline in the pipeline.
"""
from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("newsroom.maintenance")

# the chatty stages safe to age out (dedup observability); NOT publish/verify/edit audit.
PRUNABLE_DEDUP_STAGES = ("predup", "ingest_dedup")


def prune_decisions(session_factory, *, older_than_days: int = 90,
                    stages: tuple[str, ...] = PRUNABLE_DEDUP_STAGES) -> int:
    """Delete decisions in `stages` older than `older_than_days`. Returns rows deleted.

    Only the named (high-volume, low-long-term-value) stages are touched — the default is
    the dedup observe logs. Keep a generous window (default 90d) so calibration still has
    history. Safe to run repeatedly."""
    from sqlalchemy import delete

    from newsroom.models import Decision

    if not stages:
        return 0
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=older_than_days)
    with session_factory() as s:
        result = s.execute(
            delete(Decision).where(Decision.stage.in_(tuple(stages)), Decision.created_at < cutoff)
        )
        s.commit()
        deleted = int(result.rowcount or 0)
    if deleted:
        log.info("pruned decisions", extra={"deleted": deleted, "older_than_days": older_than_days})
    return deleted
