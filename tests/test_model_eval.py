from __future__ import annotations

import json

import pytest

from scripts.model_eval import (
    call_model,
    load_production_cases,
    run_production_cases,
    score_production_output,
)


def test_load_production_cases_uses_only_human_reviewed_rows(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    rows = [
        {"id": "keep", "op": "twin", "prompt": "pair", "expected": {
            "equals": {"decision": "duplicate"}}, "reviewed": True},
        {"id": "skip", "op": "twin", "prompt": "pair 2", "expected": {},
         "reviewed": False},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert [row["id"] for row in load_production_cases(str(dataset))] == ["keep"]


def test_load_production_cases_rejects_incomplete_reviewed_row(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"reviewed": True, "op": "twin"}), encoding="utf-8")

    with pytest.raises(ValueError, match="needs op, prompt and expected"):
        load_production_cases(str(dataset))


def test_load_production_cases_rejects_reviewed_row_without_assertions(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({
        "reviewed": True, "op": "generate", "prompt": "write", "expected": {"equals": {}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="needs at least one assertion"):
        load_production_cases(str(dataset))


def test_score_production_output_supports_subset_and_text_guards():
    case = {"expected": {
        "equals": {"decision": "duplicate"},
        "contains": ["same event"],
        "not_contains": ["different city"],
    }}
    raw = json.dumps({"decision": "duplicate", "confidence": 0.93,
                      "reason": "Same event from another source"})

    assert score_production_output(raw, case) == (True, "")


class _FakeClient:
    class _Completions:
        def create(self, **_kwargs):
            raise RuntimeError("temporary model failure")

    def __init__(self):
        self.chat = type("Chat", (), {"completions": self._Completions()})()


class _SchemaFallbackClient:
    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("response_format", {}).get("type") == "json_schema":
                raise ValueError("response_format json_schema is unsupported")
            message = type("Message", (), {"content": '{"decision":"separate"}'})()
            usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2,
                                        "prompt_tokens_details": None})()
            return type("Response", (), {
                "choices": [type("Choice", (), {"message": message})()], "usage": usage})()

    def __init__(self):
        self.completions = self._Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_production_eval_records_call_error_instead_of_aborting():
    cases = [{"id": "pair-1", "op": "twin", "prompt": "pair", "expected": {
        "equals": {"decision": "duplicate"}}, "reviewed": True}]

    result = run_production_cases(_FakeClient(), "gpt-4o-mini", cases)

    assert result["pass"] == 0 and result["total"] == 1
    assert "call_error" in result["misses"][0]


def test_eval_falls_back_from_schema_to_json_mode():
    client = _SchemaFallbackClient()

    raw = call_model(client, model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                     op="twin", max_out=64)

    assert raw == '{"decision":"separate"}'
    assert [call["response_format"]["type"] for call in client.completions.calls] == [
        "json_schema", "json_object",
    ]
