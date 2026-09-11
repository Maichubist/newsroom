"""Perceptual hash of an image (architecture §5.1, §9.4).

pHash reduces an image to a 64-bit fingerprint that survives rescaling and
re-encoding, so recycled imagery is caught by small Hamming distance (the
factcheck media check consumes these hashes). Computed from a 32x32 grayscale
array via a 2D DCT, keeping the low-frequency 8x8 block and thresholding at its
median. Pure numpy — decoding bytes to pixels is a separate, pluggable step, so
this is fully offline-tested. The output is 16 hex chars, matching the format
factcheck.media.phash_to_int expects.
"""
from __future__ import annotations

import numpy as np

HASH_SIZE = 8            # 8x8 low-frequency block -> 64 bits
HIGHFREQ_FACTOR = 4      # work on a 32x32 image
_IMG_SIZE = HASH_SIZE * HIGHFREQ_FACTOR


def _dct_matrix(n: int) -> np.ndarray:
    idx = np.arange(n)
    k = idx.reshape(-1, 1)
    return np.cos(np.pi / n * (idx + 0.5) * k)   # unnormalised DCT-II (median threshold is scale-free)


_DCT = _dct_matrix(_IMG_SIZE)


def phash_from_pixels(gray: np.ndarray) -> str:
    """pHash of a 32x32 grayscale array, as 16 hex chars (64 bits)."""
    img = np.asarray(gray, dtype=np.float64)
    if img.shape != (_IMG_SIZE, _IMG_SIZE):
        raise ValueError(f"expected {_IMG_SIZE}x{_IMG_SIZE} grayscale, got {img.shape}")
    dct = _DCT @ img @ _DCT.T
    low = dct[:HASH_SIZE, :HASH_SIZE].flatten()
    median = float(np.median(low))
    value = 0
    for coef in low:
        value = (value << 1) | (1 if coef > median else 0)
    return f"{value:016x}"


def phash_bytes(data: bytes, decoder) -> str | None:
    """Decode image bytes via the given decoder and hash them. None if the bytes
    cannot be decoded (never guess a hash)."""
    try:
        gray = decoder.to_grayscale(data, _IMG_SIZE)
    except Exception:  # noqa: BLE001 - a bad image must not crash the pipeline
        return None
    if gray is None:
        return None
    return phash_from_pixels(gray)
