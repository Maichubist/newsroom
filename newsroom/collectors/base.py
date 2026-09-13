"""Unified item format + dedup hashing. Pure and offline (no DB, no network)."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field

_WS_RE = re.compile(r"\s+")
# Word tokens across Latin, Cyrillic (incl. Ukrainian і ї є ґ) and digits.
_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯіІїЇєЄґҐ']+")

# We collect news, not an archive: every source ingests only items published within
# this window. Overridable via COLLECT_MAX_AGE_HOURS.
DEFAULT_MAX_ITEM_AGE_HOURS = 24


def collect_max_age_hours() -> int:
    import os

    try:
        return int(os.getenv("COLLECT_MAX_AGE_HOURS", str(DEFAULT_MAX_ITEM_AGE_HOURS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ITEM_AGE_HOURS


def is_recent(published_at: "dt.datetime | None", now: "dt.datetime", max_age_hours: int) -> bool:
    """Keep an item only if it was published within the window. An item with no
    known date is kept (a feed omitting dates is usually serving current items — we
    don't drop possibly-fresh news on a missing timestamp). max_age_hours <= 0
    disables the window (keep everything)."""
    if max_age_hours <= 0 or published_at is None:
        return True
    return published_at >= now - dt.timedelta(hours=max_age_hours)


def normalize_text(value: str | None) -> str:
    """Lowercase, collapse whitespace, strip — for stable exact-dedup hashing."""
    if not value:
        return ""
    return _WS_RE.sub(" ", value).strip().lower()


def compute_content_hash(title: str | None, text: str | None) -> str:
    """Exact cross-source dedup key: sha256 over normalized title + text."""
    payload = normalize_text(title) + "\n" + normalize_text(text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def compute_simhash(title: str | None, text: str | None) -> int:
    """64-bit SimHash over tokens — near-duplicate (rewritten copy) detection.

    Rewrites of the same story share most tokens, so their SimHashes differ in
    only a few bits (small Hamming distance); unrelated texts differ in ~32 bits.
    """
    tokens = _TOKEN_RE.findall(f"{title or ''} {text or ''}".lower())
    if not tokens:
        return 0
    bit_counts = [0] * 64
    for token in tokens:
        h = _token_hash64(token)
        for i in range(64):
            bit_counts[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if bit_counts[i] > 0:
            out |= 1 << i
    # Map the unsigned 64-bit pattern into signed range so it fits a SQL BIGINT
    # (Postgres has no unsigned bigint). The bit pattern is preserved, so
    # hamming_distance (which masks to 64 bits) is unaffected.
    if out >= (1 << 63):
        out -= 1 << 64
    return out


def hamming_distance(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


@dataclass(frozen=True)
class RawMedia:
    kind: str                 # image | video | embed
    url: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    source_ref: str | None = None    # Telegram message id (url-less media fetched via Telethon)


@dataclass
class RawItem:
    """What every collector emits before persistence — the single item format."""
    source_id: int
    external_id: str
    url: str | None = None
    title: str | None = None
    text: str | None = None
    lang: str | None = None
    published_at: dt.datetime | None = None
    fetched_at: dt.datetime | None = None
    forwarded_from: str | None = None
    grouped_id: int | None = None
    media: list[RawMedia] = field(default_factory=list)
    raw_payload: dict | None = None

    @property
    def content_hash(self) -> str:
        return compute_content_hash(self.title, self.text)

    @property
    def simhash(self) -> int:
        return compute_simhash(self.title, self.text)
