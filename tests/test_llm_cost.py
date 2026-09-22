from __future__ import annotations

import datetime as dt
import types

import pytest
from sqlalchemy.orm import Session

from newsroom.llmutil import (
    UsageRecord,
    cost_for,
    llm_context,
    messages_text,
    record_completion,
    record_usage,
    set_usage_recorder,
)


@pytest.fixture(autouse=True)
def _reset_recorder():
    yield
    set_usage_recorder(None)          # never leak a recorder between tests


# --- pricing (offline) --------------------------------------------------------

def test_cost_for_input_and_output():
    # 1000 in @ .15/M + 500 out @ .60/M = .00015 + .0003
    assert cost_for("gpt-4o-mini", 1000, 500) == pytest.approx(0.00045)


def test_cost_for_discounts_cached_prompt_tokens():
    # prompt_tokens INCLUDES cached: 600 uncached @ .15/M + 400 cached @ .075/M
    assert cost_for("gpt-4o-mini", 1000, 0, 400) == pytest.approx(0.00012)


def test_cost_for_embedding_has_no_output():
    assert cost_for("text-embedding-3-small", 1000, 0) == pytest.approx(0.00002)


def test_cost_for_unknown_model_is_zero():
    assert cost_for("some-future-model", 1000, 500) == 0.0


# --- request text (offline) ---------------------------------------------------

def test_messages_text_flattens_and_caps():
    msgs = [{"role": "system", "content": "ти редактор"}, {"role": "user", "content": "текст"}]
    assert messages_text(msgs) == "system: ти редактор\nuser: текст"
    capped = messages_text([{"role": "user", "content": "HEAD" + "x" * 9999 + "TAIL"}], cap=100)
    assert len(capped) == 100
    assert capped.startswith("user: HEAD") and capped.endswith("TAIL")
    assert messages_text([{"role": "user", "content": "abcdef"}], cap=3) == "use"


# --- recorder plumbing (offline) ----------------------------------------------

def test_record_usage_calls_the_sink_when_set():
    seen: list[UsageRecord] = []
    set_usage_recorder(seen.append)
    rec = UsageRecord(op="classify", model="gpt-4o-mini", prompt_tokens=10, cost_usd=0.001)
    record_usage(rec)
    assert seen == [rec]


def test_record_usage_is_noop_without_a_sink():
    record_usage(UsageRecord(op="x", model="m"))     # no recorder set -> must not raise


def test_record_usage_swallows_a_failing_sink():
    def boom(_rec):
        raise RuntimeError("db down")
    set_usage_recorder(boom)
    record_usage(UsageRecord(op="x", model="m"))     # telemetry failure must not propagate


def test_record_completion_builds_record_from_response():
    seen: list[UsageRecord] = []
    set_usage_recorder(seen.append)
    usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                  prompt_tokens_details=types.SimpleNamespace(cached_tokens=200))
    resp = types.SimpleNamespace(usage=usage)
    record_completion("curate", "gpt-4o-mini", resp,
                      messages=[{"role": "user", "content": "рангуй"}], event_id=7,
                      duration_ms=42, content='{"ok": true}')
    assert len(seen) == 1
    r = seen[0]
    assert r.op == "curate" and r.prompt_tokens == 1000 and r.completion_tokens == 500
    assert r.cached_tokens == 200 and r.event_id == 7 and r.duration_ms == 42
    assert r.cost_usd == pytest.approx(cost_for("gpt-4o-mini", 1000, 500, 200))
    assert "рангуй" in r.request_text and r.response_text == '{"ok": true}'


def test_record_completion_inherits_business_context():
    seen: list[UsageRecord] = []
    set_usage_recorder(seen.append)
    resp = types.SimpleNamespace(usage=_usage_for_context())
    with llm_context(event_id=17, related_event_id=9, source_id=4, stage="twin"):
        record_completion("twin", "gpt-4o-mini", resp,
                          messages=[{"role": "user", "content": "pair"}])
    assert seen[0].event_id == 17 and seen[0].related_event_id == 9
    assert seen[0].context == {"source_id": 4, "stage": "twin"}


def _usage_for_context():
    return types.SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                                 prompt_tokens_details=types.SimpleNamespace(cached_tokens=0))


# --- DB recorder + summary (pg) -----------------------------------------------

@pytest.mark.pg
def test_db_recorder_inserts_a_row(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.llm_recorder import make_db_recorder
    from newsroom.models import LlmCall

    sf = make_session_factory(pg_engine)
    rec = make_db_recorder(sf, max_text=20)
    rec(UsageRecord(op="factbase", model="gpt-4o-mini", prompt_tokens=1200, completion_tokens=300,
                    cached_tokens=100, cost_usd=0.00042, event_id=5, related_event_id=3,
                    context={"source_id": 8}, duration_ms=88,
                    request_text="x" * 100, response_text="y" * 100))
    with Session(pg_engine) as s:
        row = s.query(LlmCall).one()
        assert row.op == "factbase" and row.prompt_tokens == 1200 and row.completion_tokens == 300
        assert row.cost_usd == pytest.approx(0.00042) and row.event_id == 5
        assert row.related_event_id == 3 and row.context == {"source_id": 8}
        assert len(row.request_text) == 20 and len(row.response_text) == 20   # capped


@pytest.mark.pg
def test_summarize_cost_aggregates_by_op_and_model(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.llm_recorder import summarize_cost
    from newsroom.models import LlmCall

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        s.add_all([
            LlmCall(op="classify", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=10,
                    cost_usd=0.01, created_at=now),
            LlmCall(op="classify", model="gpt-4o-mini", prompt_tokens=200, completion_tokens=20,
                    cost_usd=0.02, created_at=now),
            LlmCall(op="embed", model="text-embedding-3-small", prompt_tokens=500, completion_tokens=0,
                    cost_usd=0.001, created_at=now),
        ])
        s.commit()

    out = summarize_cost(sf, days=7)
    assert out["calls"] == 3 and out["cost"] == pytest.approx(0.031)
    assert out["tokens"] == 100 + 10 + 200 + 20 + 500
    assert out["attributed_calls"] == 0 and out["attributed_cost"] == 0.0
    by_op = dict((o, (n, c)) for o, n, c, _t in out["by_op"])
    assert by_op["classify"] == (2, pytest.approx(0.03))
    assert out["by_op"][0][0] == "classify"          # ordered by cost desc
