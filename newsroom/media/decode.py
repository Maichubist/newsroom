"""Image decoding (pluggable).

Decoding arbitrary JPEG/PNG/WebP bytes to pixels needs an image library; keeping
it behind a Protocol lets the pHash core stay pure numpy and the tests stay
offline (a fake decoder returns an array). Production uses Pillow — an optional
dependency (`pip install "newsroom[media]"`); it is imported lazily so the rest of
the system runs without it.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Decoder(Protocol):
    def to_grayscale(self, data: bytes, size: int) -> np.ndarray | None: ...


class PillowDecoder:  # pragma: no cover - needs Pillow + real image bytes
    def to_grayscale(self, data: bytes, size: int) -> np.ndarray | None:
        import io

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('image decoding needs Pillow: pip install "newsroom[media]"') from exc

        try:
            img = Image.open(io.BytesIO(data)).convert("L").resize((size, size), Image.LANCZOS)
        except Exception:
            return None
        return np.asarray(img, dtype=np.float64)
