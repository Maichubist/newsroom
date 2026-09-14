"""Delete ALL messages from the target Telegram channel — a full reset of the
published feed, to start over from zero.

This is destructive and irreversible: it permanently deletes every message in the
channel named by TELEGRAM_CHANNEL_CHAT_ID. Run it yourself, deliberately.

Safety:
  * dry run by default — it only counts messages and prints the channel title;
    pass --yes to actually delete.
  * uses the SAME Telethon user session as the collector (TELEGRAM_SESSION_PATH).
    Telegram invalidates a session used in two places at once, so STOP the newsroom
    app before running this (it must not be collecting while this runs).
  * the session account must be the channel owner/admin with "delete messages"
    rights (bot posts are deleted by the admin account, not the bot).

Usage (from the project root, with the venv active):
    python -m scripts.wipe_channel            # dry run: how many messages
    python -m scripts.wipe_channel --yes      # actually delete everything
"""
from __future__ import annotations

import asyncio
import os
import sys


async def _resolve_channel(client, channel_ref):
    """Resolve the channel entity. get_entity by numeric id only works once the channel
    is in the session's entity cache, so warm it with get_dialogs() first, then fall back
    to matching by marked id across the account's dialogs. Returns None if the account
    cannot see the channel at all (not a member/admin)."""
    from telethon.utils import get_peer_id

    # 1. direct resolve (works for @username, or an already-cached id)
    try:
        return await client.get_entity(channel_ref)
    except (ValueError, TypeError):
        pass

    # 2. warm the dialog cache, then retry the direct resolve
    await client.get_dialogs()
    try:
        return await client.get_entity(channel_ref)
    except (ValueError, TypeError):
        pass

    # 3. last resort: scan dialogs and match by marked id (-100…)
    if isinstance(channel_ref, int):
        async for d in client.iter_dialogs():
            try:
                if get_peer_id(d.entity) == channel_ref:
                    return d.entity
            except (ValueError, TypeError):
                continue
    return None


async def _run(confirm: bool) -> int:
    from dotenv import load_dotenv
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError

    load_dotenv()
    try:
        api_id = int(os.environ["TELEGRAM_API_ID"])
        api_hash = os.environ["TELEGRAM_API_HASH"]
        session = os.environ["TELEGRAM_SESSION_PATH"]
        channel_raw = os.environ["TELEGRAM_CHANNEL_CHAT_ID"]
    except KeyError as exc:
        print(f"missing env var: {exc}", file=sys.stderr)
        return 2

    # channel id is a numeric -100... peer; fall back to a raw int if needed
    try:
        channel_ref: object = int(channel_raw)
    except ValueError:
        channel_ref = channel_raw  # allow @username too

    client = TelegramClient(session, api_id, api_hash)
    await client.start()  # uses the existing session; no login prompt if already authorized
    try:
        entity = await _resolve_channel(client, channel_ref)
        if entity is None:
            print(
                f"could not access channel {channel_raw}. The session account is likely "
                f"not a member/admin of it (only the bot posts there). Add this account "
                f"to the channel as an admin with delete rights, or delete via the bot.",
                file=sys.stderr,
            )
            return 3
        title = getattr(entity, "title", None) or getattr(entity, "username", None) or str(channel_ref)

        # collect every message id first (cheap; ids only)
        ids: list[int] = [m.id async for m in client.iter_messages(entity)]
        print(f"channel: {title}")
        print(f"messages found: {len(ids)}")

        if not ids:
            print("nothing to delete.")
            return 0
        if not confirm:
            print("\nDRY RUN — nothing deleted. Re-run with --yes to delete all of the above.")
            return 0

        deleted = 0
        # delete_messages accepts up to 100 ids per call for a channel
        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            while True:
                try:
                    await client.delete_messages(entity, batch, revoke=True)
                    break
                except FloodWaitError as fw:
                    print(f"  flood wait: sleeping {fw.seconds}s")
                    await asyncio.sleep(fw.seconds + 1)
            deleted += len(batch)
            print(f"  deleted {deleted}/{len(ids)}")

        print(f"done — deleted {deleted} messages from {title}.")
        return 0
    finally:
        await client.disconnect()


def main() -> int:
    confirm = "--yes" in sys.argv[1:]
    return asyncio.run(_run(confirm))


if __name__ == "__main__":
    raise SystemExit(main())
