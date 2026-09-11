"""Discover the account's public broadcast-channel subscriptions and merge them
into config/sources.yaml as telegram sources.

First run is INTERACTIVE: Telethon asks for the phone number, the login code
Telegram sends you, and (if set) your 2FA password. That creates the session
file at TELEGRAM_SESSION_PATH. Later runs reuse it silently.

    python scripts/import_telegram_channels.py            # dry run: list only
    python scripts/import_telegram_channels.py --write     # append new channels

Only public channels (with a @username) and broadcast channels (not groups) are
imported. Review the result: remove personal channels and set tier: official /
origin: world where appropriate.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "sources.yaml"


async def _discover() -> list:
    from telethon import TelegramClient

    from newsroom.collectors.telegram_import import DiscoveredChannel

    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        raise SystemExit("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first.")
    session = os.environ.get("TELEGRAM_SESSION_PATH", "./secrets/newsroom.session")
    Path(session).parent.mkdir(parents=True, exist_ok=True)

    found: list = []
    async with TelegramClient(session, int(api_id), api_hash) as client:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            if getattr(entity, "broadcast", False):  # broadcast channel, not a group
                found.append(DiscoveredChannel(
                    username=getattr(entity, "username", None),
                    title=getattr(entity, "title", None) or (getattr(entity, "username", None) or "channel"),
                    channel_id=int(getattr(entity, "id", 0)),
                ))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="append new channels to config/sources.yaml")
    ap.add_argument("--origin", default="ua", choices=["ua", "world"])
    ap.add_argument("--tier", default="media", choices=["official", "media", "aggregator", "leak", "anonymous"])
    args = ap.parse_args()

    from dotenv import load_dotenv

    from newsroom.collectors.telegram_import import render_source_lines
    from newsroom.config.sources import load_sources

    load_dotenv()
    discovered = asyncio.run(_discover())
    public = [d for d in discovered if d.username]
    existing = [s.handle_or_url for s in load_sources(SOURCES_PATH) if s.kind == "telegram"]
    lines = render_source_lines(public, existing, origin=args.origin, tier=args.tier)

    print(f"Discovered {len(discovered)} broadcast channels "
          f"({len(public)} public, {len(discovered) - len(public)} private/skipped); "
          f"{len(lines)} new to add.")
    for line in lines:
        print(line)

    if not lines:
        print("Nothing new to add.")
        return
    if not args.write:
        print("\nDry run. Re-run with --write to append these to config/sources.yaml, "
              "then review tiers/origin before enabling COLLECTOR_TELEGRAM_ENABLED.")
        return

    with SOURCES_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n  # --- Telegram channels (auto-imported from subscriptions) ---\n")
        fh.write("\n".join(lines) + "\n")
    print(f"\nAppended {len(lines)} channels to {SOURCES_PATH}. Review before enabling collection.")


if __name__ == "__main__":
    main()
