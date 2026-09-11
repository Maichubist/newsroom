from __future__ import annotations

import pytest

from newsroom.analyze.classifier import DEFAULT_PROMPT as CLASSIFY_PROMPT
from newsroom.editorial.generator import DEFAULT_PROMPT as GEN_PROMPT
from newsroom.editorial.updates import DEFAULT_UPDATE_PROMPT
from newsroom.factbase.builder import DEFAULT_FACT_PROMPT
from newsroom.factcheck.claims import DEFAULT_CLAIM_PROMPT
from newsroom.factcheck.verdict import DEFAULT_VERDICT_PROMPT
from newsroom.promptutil import fill_prompt


def test_fill_prompt_leaves_json_braces_alone():
    tpl = 'Поверни JSON: {"claims": ["..."]}\nМатеріал:\n{news_text}'
    out = fill_prompt(tpl, news_text="ТЕКСТ")
    assert '{"claims": ["..."]}' in out and "ТЕКСТ" in out and "{news_text}" not in out


def test_str_format_would_break_json_prompts():
    # documents WHY we use fill_prompt: str.format raises on the JSON braces
    with pytest.raises(KeyError):
        '{"claims": []}\n{news_text}'.format(news_text="x")


# Every prompt filled with its real placeholders must not raise, must substitute
# the value, and (for JSON prompts) must keep the JSON example intact.
PROMPTS = [
    (DEFAULT_CLAIM_PROMPT, {"news_text": "ЗРАЗОК"}, '{"claims"'),
    (DEFAULT_FACT_PROMPT, {"news_text": "ЗРАЗОК"}, '{"facts"'),
    (DEFAULT_VERDICT_PROMPT, {"claim": "ЗРАЗОК", "evidence": "E"}, '{"verdict"'),
    (DEFAULT_UPDATE_PROMPT, {"summary": "ЗРАЗОК", "event": "E"}, '{"update_type"'),
    (GEN_PROMPT, {"feedback": "", "material": "ЗРАЗОК"}, '{"headline"'),
    (CLASSIFY_PROMPT, {"news_text": "ЗРАЗОК"}, None),   # classifier prompt has no JSON braces
]


@pytest.mark.parametrize("prompt,kwargs,json_marker", PROMPTS)
def test_prompt_fills_cleanly(prompt, kwargs, json_marker):
    out = fill_prompt(prompt, **kwargs)
    assert "ЗРАЗОК" in out                      # the placeholder was substituted
    for key in kwargs:
        assert "{" + key + "}" not in out        # no leftover placeholder
    if json_marker:
        assert json_marker in out                # the JSON example survived intact
