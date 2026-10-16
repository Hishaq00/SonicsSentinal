"""Acoustic fingerprinting for near-duplicate detection.

Exact duplicates are found with a SHA-256 hash of the file bytes. That misses
copies that were re-encoded (WAV -> MP3), trimmed or had their volume changed,
so we also compute a Philips-style binary fingerprint:

* resample to 8 kHz, STFT with 128 ms windows and an 8 ms hop,
* 17 log-spaced bands between 300 Hz and 3 kHz,
* 16 bits per frame: the sign of the band-energy difference, differenced in time.

The bits depend on energy *differences*, so a volume change does not alter them
and lossy re-encoding flips only a few bits. Two recordings are compared by
sliding one over the other (FFT cross-correlation) and taking the lowest bit
error rate (BER). Trimmed copies still align at some offset.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .loader import resample

FP_SR = 8000
FP_NFFT = 1024
FP_HOP = 64  # 1/16 of the window: large overlap keeps bits stable when a copy is trimmed at any offset
FP_BANDS = 17
FP_FMIN = 300.0
FP_FMAX = 3000.0
NEAR_DUPLICATE_BER = 0.25


def _band_energies(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != FP_SR:
        y = resample(y, sr, FP_SR)
    if y.size < FP_NFFT:
        y = np.pad(y, (0, FP_NFFT - y.size))
    n = 1 + (y.size - FP_NFFT) // FP_HOP
    idx = np.arange(FP_NFFT)[None, :] + FP_HOP * np.arange(n)[:, None]
    frames = y[idx] * np.hanning(FP_NFFT)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FP_NFFT, 1.0 / FP_SR)
    edges = np.geomspace(FP_FMIN, FP_FMAX, FP_BANDS + 1)
    bands = np.empty((n, FP_BANDS), dtype=np.float64)
    for b in range(FP_BANDS):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        bands[:, b] = spec[:, sel].sum(axis=1)
    return np.log10(bands + 1e-10)


def compute_fingerprint(y: np.ndarray, sr: int) -> tuple[bytes, bytes]:
    """Return (packed fingerprint bits, descriptor vector) as bytes for DB storage."""
    energy = _band_energies(np.asarray(y, dtype=np.float32), sr)
    diff = energy[:, :-1] - energy[:, 1:]  # (frames, 16)
    bits = (diff[1:] - diff[:-1]) > 0  # (frames-1, 16)
    packed = np.packbits(bits.astype(np.uint8), axis=1)  # 2 bytes per frame
    # Volume-invariant spectral descriptor used to shortlist candidates quickly.
    rel = energy - energy.mean(axis=1, keepdims=True)
    desc = np.concatenate([rel.mean(axis=0), rel.std(axis=0)]).astype(np.float32)
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm
    return packed.tobytes(), desc.tobytes()


def unpack_fingerprint(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(-1, 2)
    return np.unpackbits(arr, axis=1)[:, :16].astype(np.float32)


def descriptor_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def compare_fingerprints(fp_a: bytes, fp_b: bytes, min_overlap_ratio: float = 0.8) -> tuple[float, int]:
    """Lowest bit error rate between two fingerprints over all alignments.

    Only alignments where the overlap covers at least ``min_overlap_ratio`` of
    the shorter recording are considered, so a short clip must be (mostly)
    contained in the longer one.
    """
    a = unpack_fingerprint(fp_a) * 2 - 1
    b = unpack_fingerprint(fp_b) * 2 - 1
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0, 0
    agree = np.zeros(na + nb - 1)
    for c in range(a.shape[1]):
        agree += fftconvolve(a[:, c], b[::-1, c], mode="full")
    lags = np.arange(-(nb - 1), na)
    overlap = np.minimum(na, lags + nb) - np.maximum(0, lags)
    need = max(4, int(min(na, nb) * min_overlap_ratio))
    valid = overlap >= need
    if not valid.any():
        return 1.0, 0
    ber = 0.5 * (1.0 - agree[valid] / (a.shape[1] * overlap[valid]))
    best = int(np.argmin(ber))
    return float(ber[best]), int(lags[valid][best])
