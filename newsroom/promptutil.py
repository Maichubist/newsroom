"""Fill a prompt template's {placeholders} without str.format().

The LLM prompts embed literal JSON examples like {"claims": [...]}. str.format()
reads those braces as format fields and raises KeyError('"claims"') before the
model is ever called — silently turning extraction into "no result". Substituting
placeholders with str.replace leaves JSON braces untouched, so this is the only
safe way to fill these templates.
"""
from __future__ import annotations


def fill_prompt(template: str, /, **values: str) -> str:
    """Replace each {key} in the template with its value. JSON braces in the
    template are left alone (unlike str.format)."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out
