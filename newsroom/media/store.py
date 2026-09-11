"""Media storage behind an abstraction (architecture §11).

Local folder in development, S3-compatible object storage in the cloud — the same
`MediaStore` interface either way, so the downloader never hard-codes a backend.
Only LocalMediaStore ships now; the S3 variant is a later, drop-in implementation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class MediaStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...

    def exists(self, key: str) -> bool: ...


def media_key(item_id: int, media_id: int, url: str) -> str:
    """A stable, collision-resistant storage key for one media asset."""
    digest = hashlib.sha256(f"{item_id}:{media_id}:{url}".encode("utf-8")).hexdigest()
    return f"{digest[:2]}/{digest}"


class LocalMediaStore:
    """Writes media under a base directory (dev). storage_key is the relative path."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def put(self, key: str, data: bytes) -> str:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()
