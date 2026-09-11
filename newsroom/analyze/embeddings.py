"""Embedding provider (pluggable).

The clusterer is embedder-agnostic — it takes vectors — so tests inject vectors
directly and never need this module. Production uses OpenAIEmbedder. The model
and its dimensionality are an open decision (architecture §15); the default
matches db.base.EMBEDDING_DIM.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from newsroom.db.base import EMBEDDING_DIM

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims == EMBEDDING_DIM


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
        resp = self._ensure_client().embeddings.create(model=self.model, input=text or " ")
        vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
        if vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"embedding dim {vec.shape[0]} != expected {EMBEDDING_DIM}")
        return vec
