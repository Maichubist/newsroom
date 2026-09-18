from __future__ import annotations

import types

from newsroom.llmutil import _cached_tokens, chat_json, log_usage


def _usage(prompt=100, completion=20, cached=None):
    details = types.SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                 prompt_tokens_details=details)


class FakeClient:
    """Records create() kwargs and returns a canned response."""

    def __init__(self, content='{"ok": true}', usage=None):
        self.calls: list[dict] = []
        self._content = content
        self._usage = usage
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=self._usage)


def test_chat_json_caps_output_and_sets_json_format():
    client = FakeClient(content='{"decision": "duplicate"}', usage=_usage())
    out = chat_json(client, model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    op="twin", max_tokens=256, temperature=0.0)
    assert out == '{"decision": "duplicate"}'
    kw = client.calls[0]
    assert kw["max_tokens"] == 256                      # hard output cap applied
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["temperature"] == 0.0 and kw["model"] == "gpt-4o-mini"


def test_chat_json_passes_temperature_through():
    client = FakeClient(usage=_usage())
    chat_json(client, model="m", messages=[{"role": "user", "content": "x"}],
              op="generate", max_tokens=2048, temperature=0.4)
    assert client.calls[0]["temperature"] == 0.4


def test_chat_json_survives_missing_usage():
    client = FakeClient(content='{"a":1}', usage=None)   # some responses carry no usage
    assert chat_json(client, model="m", messages=[{"role": "user", "content": "x"}],
                     op="op", max_tokens=100) == '{"a":1}'


def test_log_usage_does_not_raise_without_usage():
    log_usage("op", "m", types.SimpleNamespace(usage=None))   # no-op, no crash


def test_cached_tokens_defensive():
    assert _cached_tokens(_usage(cached=7)) == 7
    assert _cached_tokens(_usage(cached=None)) is None       # no details -> None
    assert _cached_tokens(types.SimpleNamespace(prompt_tokens_details=None)) is None
