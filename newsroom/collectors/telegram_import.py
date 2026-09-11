"""Turn discovered Telegram channels into config/sources.yaml lines.

Pure and offline (no Telethon, no network): the network discovery lives in
scripts/import_telegram_channels.py; this module only renders/deduplicates
source entries so the logic is unit-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DiscoveredChannel:
    username: str | None
    title: str
    channel_id: int


def render_source_lines(
    discovered: Iterable[DiscoveredChannel],
    existing_handles: Iterable[str],
    *,
    origin: str = "ua",
    tier: str = "media",
) -> list[str]:
    """One `sources.yaml` list line per NEW public channel (has a @username and
    is not already present). Titles are JSON-encoded, which is valid YAML and
    escapes quotes/colons/braces safely. Dedup is case-insensitive by handle."""
    existing = {(h or "").strip().lower() for h in existing_handles}
    seen: set[str] = set()
    lines: list[str] = []
    for ch in discovered:
        uname = (ch.username or "").strip().lstrip("@")
        if not uname:
            continue  # private channels (no username) are skipped for now
        handle = f"@{uname}"
        key = handle.lower()
        if key in existing or key in seen:
            continue
        seen.add(key)
        name = json.dumps(ch.title or handle, ensure_ascii=False)
        lines.append(
            f'  - {{name: {name}, kind: telegram, handle_or_url: "{handle}", '
            f"origin: {origin}, tier: {tier}}}"
        )
    return lines
