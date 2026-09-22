"""Export recent real LLM calls into a JSONL annotation queue for model_eval.

The exporter deliberately marks every row ``reviewed=false``. A model's historical
answer is useful as a starting suggestion, never as ground truth. A human reviews a
balanced sample, corrects ``expected``, flips ``reviewed`` to true, and then runs:

    python -m scripts.model_eval --dataset eval_cases.jsonl --models gpt-4o-mini,gpt-5-nano
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


OPS = (
    "classify", "twin", "story_update", "factbase", "factcheck_claims",
    "factcheck_verdict", "curate", "generate", "dedup", "taxonomy_merge",
)

MAX_OUT = {
    "classify": 768, "twin": 256, "story_update": 512, "factbase": 2048,
    "factcheck_claims": 1536, "factcheck_verdict": 512, "curate": 2048,
    "generate": 2048, "dedup": 1024, "taxonomy_merge": 1024,
}

CRITICAL_KEYS = {
    "classify": ("is_event", "rubrics", "side", "is_first_source", "is_rumor"),
    "twin": ("decision",),
    "story_update": ("update_type", "significant", "position_changed"),
    "factcheck_verdict": ("verdict",),
}


def _prompt(request_text: str | None) -> str:
    text = (request_text or "").strip()
    return text[6:] if text.startswith("user: ") else text


def _suggested_expected(op: str, response_text: str | None) -> dict:
    try:
        obj = json.loads(response_text or "")
    except (TypeError, json.JSONDecodeError):
        return {"equals": {}}
    if not isinstance(obj, dict):
        return {"equals": {}}
    keys = CRITICAL_KEYS.get(op, ())
    return {"equals": {key: obj[key] for key in keys if key in obj}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", default="eval_cases.jsonl")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from sqlalchemy import select

    from newsroom.db import make_engine, make_session_factory
    from newsroom.models import LlmCall

    load_dotenv()
    sf = make_session_factory(make_engine())
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, args.days))
    with sf() as session:
        calls = list(session.execute(
            select(LlmCall)
            .where(LlmCall.created_at >= cutoff, LlmCall.op.in_(OPS),
                   LlmCall.request_text.is_not(None), LlmCall.response_text.is_not(None))
            .order_by(LlmCall.created_at.desc())
            .limit(max(1, args.limit * 3))
        ).scalars())

    rows: list[dict] = []
    seen: set[str] = set()
    for call in calls:
        prompt = _prompt(call.request_text)
        if not prompt:
            continue
        digest = hashlib.sha256(f"{call.op}\0{prompt}".encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows.append({
            "id": f"llm-{call.id}",
            "op": call.op,
            "prompt": prompt,
            "expected": _suggested_expected(call.op, call.response_text),
            "max_out": MAX_OUT.get(call.op, 1024),
            "reviewed": False,
            "source": {
                "llm_call_id": call.id,
                "event_id": call.event_id,
                "related_event_id": call.related_event_id,
                "created_at": call.created_at.isoformat() if call.created_at else None,
                "historical_model": call.model,
            },
        })
        if len(rows) >= args.limit:
            break

    out = Path(args.out)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                   encoding="utf-8")
    print(f"Exported {len(rows)} unreviewed production cases to {out}")
    print("Review expected values and set reviewed=true before model_eval uses a row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
