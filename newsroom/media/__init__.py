from newsroom.media.phash import phash_bytes, phash_from_pixels
from newsroom.media.store import LocalMediaStore, MediaStore, media_key
from newsroom.media.decode import Decoder, PillowDecoder
from newsroom.media.download import DownloadResult, MediaDownloader, persist_media_bytes
from newsroom.media.purge import purge_event_media, purge_stale_media
from newsroom.media.tgdownload import TelegramMediaDownloader, select_pending_tg_media
from newsroom.media.ogimage import OgImageResolver, OgResult, extract_og_image, resolve_pending
from newsroom.media.moderation import (
    ImageModerator,
    ImageVerdict,
    ModerationResult,
    OpenAIImageModerator,
    moderate_event_media,
    moderate_pending,
    parse_image_verdict,
)

__all__ = [
    "phash_from_pixels", "phash_bytes",
    "MediaStore", "LocalMediaStore", "media_key",
    "Decoder", "PillowDecoder",
    "MediaDownloader", "DownloadResult", "persist_media_bytes", "purge_event_media", "purge_stale_media",
    "TelegramMediaDownloader", "select_pending_tg_media",
    # og:image resolution (§13) — media for feeds that carry none in RSS
    "OgImageResolver", "OgResult", "extract_og_image", "resolve_pending",
    # image moderation (charter §4, §9.5)
    "ImageVerdict", "ImageModerator", "OpenAIImageModerator", "parse_image_verdict",
    "moderate_event_media", "moderate_pending", "ModerationResult",
]
