"""Embedding provider (pluggable).

The clusterer is embedder-agnostic — it takes vectors — so tests inject vectors
directly and never need this module. Production uses OpenAIEmbedder. The model
and its dimensionality are an open decision (architecture §15); the default
matches db.base.EMBEDDING_DIM.
"""
from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from newsroom.db.base import EMBEDDING_DIM

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims == EMBEDDING_DIM

# The embedding models cap input at 8192 tokens. A BPE token is always >= 1
# character, so capping the character count is a hard upper bound on the token
# count — 8000 chars can never exceed 8000 tokens. That's plenty of topical
# signal for clustering, and it means a long full-text article never 400s.
MAX_EMBED_CHARS = 8000

# Boilerplate that pulls a cross-source twin's vector toward the CHANNEL's template
# instead of the story (measured: it drags real twins' cosine below the story-link bar).
# We strip footers/emoji/URLs before embedding — NOT linguistic reduction (lemmatizing or
# keeping only nouns HURTS a transformer embedding and destroys negation/modality), and NO
# lede truncation (double-edged: it cut shared facts). Raw text is untouched; this only
# feeds the embedder. The biggest win is the Telegram side (RSS is already trafilatura-clean).
_URL_RE = re.compile(r"https?://\S+")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U00002190-\U000021FF]+",
    flags=re.UNICODE,
)
# subscribe/donate/social footer phrases — matched only on SHORT lines or pipe/🛑 menus,
# so content words (програми ПІДТРИМКИ, Зеленський ПІДПИСАВ) are never dropped.
_FOOTER_RE = re.compile(
    r"(підписат|підписуй|підтримати нас|підтримати$|наш канал|читайте також|instagram|"
    r"facebook|youtube|threads|tiktok|тikтok|t\.me/|написати нам|зсу\s*help)",
    re.IGNORECASE,
)


def clean_for_embedding(text: str | None) -> str:
    """Strip channel boilerplate (subscribe/social footers, emoji, URLs, pipe/🛑 menus)
    from text before embedding. Never linguistic reduction, never lede truncation. Falls
    back to the raw stripped text if cleaning would empty it (an all-footer post)."""
    raw = (text or "").replace("\xa0", " ")
    stripped_url = _URL_RE.sub(" ", raw)
    out: list[str] = []
    for line in stripped_url.split("\n"):
        no_emoji = _EMOJI_RE.sub("", line)
        stripped = no_emoji.strip(" |—-•·🛑").strip()
        if not stripped:
            continue
        is_menu = line.count("|") >= 2 or line.count("🛑") >= 2
        is_footer = _FOOTER_RE.search(no_emoji) and len(stripped) < 70
        if is_menu or is_footer:
            continue
        out.append(stripped)
    cleaned = re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", "\n".join(out))).strip()
    return cleaned or raw.strip()


def clip_for_embedding(text: str | None, *, max_chars: int = MAX_EMBED_CHARS) -> str:
    """Clean channel boilerplate, then trim to a safe length (never empty)."""
    trimmed = clean_for_embedding(text)[:max_chars]
    return trimmed or " "


class Embedder(Protocol):
    model: str

    def embed(self, text: str) -> np.ndarray: ...


class OpenAIEmbedder:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_EMBEDDING_MODEL):
        import os

        self.model = model
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> np.ndarray:
        import time

        from newsroom.llmutil import (
            MAX_LOG_TEXT,
            UsageRecord,
            cost_for,
            current_llm_context,
            record_usage,
        )

        inp = clip_for_embedding(text)
        t0 = time.monotonic()
        resp = self._ensure_client().embeddings.create(model=self.model, input=inp)
        duration_ms = int((time.monotonic() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        ptok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        context = current_llm_context()
        event_id = context.pop("event_id", None)
        related_event_id = context.pop("related_event_id", None)
        record_usage(UsageRecord(op="embed", model=self.model, prompt_tokens=ptok,
                                 cost_usd=cost_for(self.model, ptok, 0, 0), duration_ms=duration_ms,
                                 event_id=event_id, related_event_id=related_event_id,
                                 context=context or None,
                                 request_text=inp[:MAX_LOG_TEXT]))
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"embedding dim {vec.shape[0]} != expected {EMBEDDING_DIM}")
        return vec
