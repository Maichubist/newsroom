"""Print an LLM cost report from the llm_calls table.

Every LLM call (completion + embedding) is recorded with its token usage, computed USD
cost and the (capped) request/response text. This script aggregates that spend for
analysis: totals, and breakdowns by op, by model and by day.

Usage (project root, venv active):
    python -m scripts.llm_cost_report            # last 7 days
    python -m scripts.llm_cost_report 30         # last 30 days

For per-call detail, query the table directly, e.g. via the admin /sql console:
    SELECT created_at, op, model, cost_usd, prompt_tokens, completion_tokens, request_text
    FROM llm_calls ORDER BY cost_usd DESC LIMIT 50;
"""
from __future__ import annotations

import sys


def main() -> int:
    from dotenv import load_dotenv

    from newsroom.db.base import make_engine, make_session_factory
    from newsroom.llm_recorder import summarize_cost

    load_dotenv()
    days = 7
    if len(sys.argv) > 1:
        try:
            days = max(1, int(sys.argv[1]))
        except ValueError:
            print(f"bad days argument {sys.argv[1]!r}; using 7", file=sys.stderr)

    session_factory = make_session_factory(make_engine())
    s = summarize_cost(session_factory, days=days)

    if not s["calls"]:
        print(f"No LLM calls in the last {days} day(s) — the llm_calls table is empty.")
        return 0

    print(f"LLM spend, last {days} day(s)")
    print(f"  total: ${s['cost']:.4f}   calls: {s['calls']}   tokens: {s['tokens']:,}")
    call_share = (100.0 * s["attributed_calls"] / s["calls"]) if s["calls"] else 0.0
    cost_share = (100.0 * s["attributed_cost"] / s["cost"]) if s["cost"] else 0.0
    print(f"  attributed to event: {s['attributed_calls']}/{s['calls']} calls "
          f"({call_share:.1f}%), ${s['attributed_cost']:.4f} ({cost_share:.1f}% of spend)")

    print("\nby op:")
    print(f"  {'op':16} {'calls':>7} {'cost $':>12} {'tokens':>12}")
    for op, n, cost, toks in s["by_op"]:
        print(f"  {op[:16]:16} {n:>7} {cost:>12.4f} {toks:>12,}")

    print("\nby model:")
    print(f"  {'model':26} {'calls':>7} {'cost $':>12}")
    for model, n, cost in s["by_model"]:
        print(f"  {model[:26]:26} {n:>7} {cost:>12.4f}")

    print("\nby day:")
    print(f"  {'day':12} {'cost $':>12} {'calls':>7}")
    for day, cost, n in s["by_day"]:
        print(f"  {day:12} {cost:>12.4f} {n:>7}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
