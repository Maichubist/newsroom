"""Крок за кроком: як йшла кожна новина і чому була опублікована.

Збирає з журналу `decisions` (+ подія, джерела, публікація) ВЕЛИКУ детальну таблицю:
один рядок на КОЖНЕ рішення (новина + дія), з контекстом події та її фінальним статусом.
READ-ONLY — нічого не пише.

Приклади:
    python -m scripts.trace_events                      # усі опубліковані + їхній повний трейс → CSV
    python -m scripts.trace_events --all --limit 200    # усі події
    python -m scripts.trace_events --event 1234         # трейс однієї події у консоль
    python -m scripts.trace_events --out trace.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _db_url() -> str:
    url = os.environ.get("NEWSROOM_DATABASE_URL")
    if url:
        return url
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("NEWSROOM_DATABASE_URL="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("NEWSROOM_DATABASE_URL не задано")


_TIER_RANK = {"official": 0, "media": 1, "aggregator": 2, "leak": 3, "anonymous": 4}
_TIER_LABEL = {0: "official", 1: "media", 2: "aggregator", 3: "leak", 4: "anonymous", 5: "?"}

CSV_COLS = [
    "event_id", "title", "rubric", "risk", "status", "curated", "duplicate_of",
    "is_rumor", "is_first_source", "indep_sources", "src_tier", "sources",
    "published", "pub_kind", "pub_headline",
    "step", "decision_entity_type", "decision_entity_id", "step_time", "stage", "decision",
    "reason", "score", "ref_id", "model", "details",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, help="трейс однієї події (у консоль)")
    ap.add_argument("--all", action="store_true", help="усі події, не лише опубліковані")
    ap.add_argument("--limit", type=int, default=100, help="скільки подій (за спаданням id)")
    ap.add_argument("--out", help="куди зберегти CSV")
    args = ap.parse_args()

    from sqlalchemy import create_engine, text
    eng = create_engine(_db_url(), future=True)

    with eng.connect() as c:
        # which columns exist (score/ref_id may predate the migration)
        cols = {r[0] for r in c.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='decisions'")).all()}
        has_new = {"score", "ref_id"} <= cols
        sc = ("d.score, d.ref_id" if has_new
              else "NULL::float AS score, NULL::bigint AS ref_id")

        if args.event:
            event_ids = [args.event]
        elif args.all:
            event_ids = [r[0] for r in c.execute(text(
                "SELECT id FROM events ORDER BY id DESC LIMIT :l"), {"l": args.limit}).all()]
        else:  # published only
            event_ids = [r[0] for r in c.execute(text(
                "SELECT DISTINCT e.id FROM events e JOIN publications p ON p.event_id=e.id "
                "WHERE p.status='published' ORDER BY e.id DESC LIMIT :l"), {"l": args.limit}).all()]
        if not event_ids:
            print("Немає подій у вибірці."); return 0

        events = {r.id: r for r in c.execute(text("""
            SELECT id, title, rubric, risk_level, status, curated, duplicate_of,
                   is_rumor, is_first_source, independent_source_count
            FROM events WHERE id = ANY(:ids)"""), {"ids": event_ids}).all()}

        pubs: dict[int, tuple] = {}
        for r in c.execute(text("""SELECT event_id, kind, status, headline FROM publications
            WHERE event_id = ANY(:ids)"""), {"ids": event_ids}).all():
            # keep the "published" one if present, else any
            if r.event_id not in pubs or r.status == "published":
                pubs[r.event_id] = (r.kind, r.status, r.headline)

        tiers: dict[int, tuple] = {}
        for r in c.execute(text("""SELECT ei.event_id, s.tier, s.is_official, s.name
            FROM sources s JOIN items i ON i.source_id=s.id JOIN event_items ei ON ei.item_id=i.id
            WHERE ei.event_id = ANY(:ids)"""), {"ids": event_ids}).all():
            rank = 0 if r.is_official else _TIER_RANK.get((r.tier or "").lower(), 5)
            best, names = tiers.get(r.event_id, (5, set()))
            names.add(r.name or "?")
            tiers[r.event_id] = (min(best, rank), names)

        # Decision is deliberately polymorphic: entity_id is text and may point to an
        # item, event, publication or media_asset. Build every scope that belongs to the
        # requested Event so the trace does not stop before predup/media/publish.
        steps: dict[int, list] = {eid: [] for eid in event_ids}
        for r in c.execute(text(f"""
            WITH scopes AS (
                SELECT e.id AS event_id, 'event'::text AS entity_type, e.id::text AS entity_id
                FROM events e WHERE e.id = ANY(:ids)
                UNION ALL
                SELECT p.event_id, 'publication', p.id::text
                FROM publications p WHERE p.event_id = ANY(:ids)
                UNION ALL
                SELECT ei.event_id, 'item', ei.item_id::text
                FROM event_items ei WHERE ei.event_id = ANY(:ids)
                UNION ALL
                SELECT ei.event_id, 'media_asset', ma.id::text
                FROM event_items ei JOIN media_assets ma ON ma.item_id = ei.item_id
                WHERE ei.event_id = ANY(:ids)
            )
            SELECT scopes.event_id, d.entity_type, d.entity_id, d.stage, d.decision,
                   d.reason, {sc}, d.details, d.model, d.created_at
            FROM scopes JOIN decisions d
              ON d.entity_type = scopes.entity_type AND d.entity_id = scopes.entity_id
            ORDER BY scopes.event_id, d.created_at, d.id
        """), {"ids": event_ids}).all():
            steps.setdefault(int(r.event_id), []).append(r)

    rows_out = []
    for eid in event_ids:
        ev = events.get(eid)
        if ev is None:
            continue
        tier_rank, names = tiers.get(eid, (5, set()))
        pub = pubs.get(eid)
        base = {
            "event_id": eid, "title": (ev.title or "")[:80], "rubric": ev.rubric or "",
            "risk": ev.risk_level or "", "status": ev.status or "", "curated": ev.curated or "",
            "duplicate_of": ev.duplicate_of or "", "is_rumor": bool(ev.is_rumor),
            "is_first_source": bool(ev.is_first_source),
            "indep_sources": ev.independent_source_count or 0,
            "src_tier": _TIER_LABEL.get(tier_rank, "?"), "sources": ", ".join(sorted(names))[:80],
            "published": bool(pub and pub[1] == "published"),
            "pub_kind": pub[0] if pub else "", "pub_headline": (pub[2] or "")[:80] if pub else "",
        }
        evsteps = steps.get(eid, [])
        if not evsteps:
            rows_out.append({**base, "step": 0, "step_time": "", "stage": "(немає рішень)",
                             "decision_entity_type": "", "decision_entity_id": "",
                             "decision": "", "reason": "", "score": "", "ref_id": "",
                             "model": "", "details": ""})
        for i, st in enumerate(evsteps, 1):
            rows_out.append({**base, "step": i,
                             "decision_entity_type": st.entity_type,
                             "decision_entity_id": st.entity_id,
                             "step_time": st.created_at.isoformat() if st.created_at else "",
                             "stage": st.stage, "decision": st.decision, "reason": st.reason or "",
                             "score": "" if st.score is None else round(st.score, 4),
                             "ref_id": st.ref_id or "", "model": st.model or "",
                             "details": json.dumps(st.details, ensure_ascii=False) if st.details else ""})

    # single-event trace to console
    if args.event:
        _print_trace(args.event, rows_out)
        return 0

    out = args.out or str(Path(os.environ.get("TEMP", ".")) / "events_trace.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows_out)
    n_events = len({r["event_id"] for r in rows_out})
    print(f"Готово: {len(rows_out)} рядків (крок-рішень) по {n_events} подіях → {out}")
    if not has_new:
        print("УВАГА: колонки score/ref_id ще не в БД (запусти init_db); показано NULL.")
    return 0


def _print_trace(eid: int, rows_out: list) -> None:
    rows = [r for r in rows_out if r["event_id"] == eid]
    if not rows:
        print(f"Подія {eid} не знайдена."); return
    h = rows[0]
    print(f"\n=== ПОДІЯ {eid} — {h['title']} ===")
    print(f"рубрика={h['rubric']} ризик={h['risk']} статус={h['status']} курація={h['curated']} "
          f"tier={h['src_tier']} джерел={h['indep_sources']} чутка={h['is_rumor']} "
          f"першоджерело={h['is_first_source']} опубліковано={h['published']}")
    if h["duplicate_of"]:
        print(f"  ⤷ дублікат події {h['duplicate_of']}")
    if h["pub_headline"]:
        print(f"  публікація [{h['pub_kind']}/{'published' if h['published'] else '—'}]: {h['pub_headline']}")
    print(f"\n{'#':>2} {'час':<10}{'сутність':<20}{'стадія':<15}{'рішення':<18}"
          f"{'score':>8}{'ref':>7}  деталі/причина")
    print("-" * 125)
    for r in rows:
        t = (r["step_time"] or "")[11:19]
        extra = r["reason"] or r["details"]
        scope = f"{r['decision_entity_type']}:{r['decision_entity_id']}"
        print(f"{r['step']:>2} {t:<10}{scope[:19]:<20}{r['stage']:<15}{str(r['decision'])[:17]:<18}"
              f"{str(r['score']):>8}{str(r['ref_id']):>7}  {str(extra)[:48]}")


if __name__ == "__main__":
    raise SystemExit(main())
