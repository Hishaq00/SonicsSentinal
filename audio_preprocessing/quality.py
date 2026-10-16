"""Audio quality analysis.

Checks: silence, clipping, excessive background noise, low signal strength,
unsuitable duration, encoding problems and missing audio frames (drop-outs).
Quality is measured on the *original* decoded signal (before normalisation),
because normalisation would hide low levels and clipping.

Scoring: start at 100 and subtract a penalty per issue.
    >= 80 Good, >= 60 Acceptable, >= 35 Poor, otherwise Unusable.
A fatal issue (silence, undecodable content) is always Unusable.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "silence_dbfs": -55.0,
    "low_signal_dbfs": -40.0,
    "clipping_warning_ratio": 0.001,
    "clipping_severe_ratio": 0.01,
    "noise_floor_limit_dbfs": -25.0,
}


def _frames(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if y.size < frame:
        return np.pad(y, (0, frame - y.size))[None, :]
    n = 1 + (y.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def _db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-10))


def clipping_ratio(y: np.ndarray, level: float = 0.995, min_run: int = 3) -> float:
    """Fraction of samples that sit in flat runs at full scale (true clipping)."""
    if y.size == 0:
        return 0.0
    hot = np.abs(y) >= level
    if not hot.any():
        return 0.0
    padded = np.concatenate(([0], hot.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    lengths = ends - starts
    clipped = int(lengths[lengths >= min_run].sum())
    return clipped / float(y.size)


def dropout_count(y: np.ndarray, sr: int, min_ms: float = 10.0) -> int:
    """Runs of exact digital zeros *inside* the signal (missing frames / buffer drop-outs)."""
    if y.size == 0:
        return 0
    nz = np.flatnonzero(y != 0)
    if nz.size == 0:
        return 0
    inner = y[nz[0]:nz[-1] + 1]
    zero = (inner == 0).astype(np.int8)
    padded = np.concatenate(([0], zero, [0]))
    diff = np.diff(padded)
    lengths = np.flatnonzero(diff == -1) - np.flatnonzero(diff == 1)
    return int(np.sum(lengths >= int(sr * min_ms / 1000.0)))


def spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness (1 = white noise, 0 = pure tone)."""
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1)
    return geo / arith


def analyze_quality(
    y: np.ndarray,
    sr: int,
    settings: dict | None = None,
    decode_warnings: list[str] | None = None,
    expected_frames: int | None = None,
    decoded_frames: int | None = None,
    bit_depth: int | None = None,
    compact: bool = False,
) -> dict:
    """Return quality label, score, measured metrics and a list of issues.

    ``compact=True`` skips the more expensive checks (used per segment).
    """
    cfg = dict(DEFAULTS)
    cfg.update({k: float(v) for k, v in (settings or {}).items() if k in DEFAULTS})
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    issues: list[dict] = []
    fatal = False

    duration = y.size / float(sr) if sr else 0.0
    frame = max(256, int(0.05 * sr))  # 50 ms frames
    hop = frame // 2
    frames = _frames(y, frame, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    rms_db = _db(rms)
    nonzero = rms_db > -150  # ignore digital-silence frames when estimating noise
    rms_db_active = rms_db[nonzero] if nonzero.any() else rms_db

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    loudest = float(rms_db.max()) if rms_db.size else -200.0
    overall = float(_db(np.sqrt(np.mean(y.astype(np.float64) ** 2)))) if y.size else -200.0
    noise_floor = float(np.percentile(rms_db_active, 10)) if rms_db_active.size else -200.0
    # 99th percentile: brief events (gunshot, glass) must count as signal.
    signal_level = float(np.percentile(rms_db_active, 99)) if rms_db_active.size else -200.0
    snr = signal_level - noise_floor
    silence_ratio = float(np.mean(rms_db < cfg["silence_dbfs"])) if rms_db.size else 1.0
    clip = clipping_ratio(y)

    # Broadband-ness of the quietest frames: distinguishes noise from stationary tones.
    quiet_idx = np.argsort(rms)[: max(1, len(rms) // 5)]
    flat = float(np.mean(spectral_flatness(frames[quiet_idx]))) if frames.size else 0.0

    metrics = {
        "duration_s": round(duration, 3),
        "peak_dbfs": round(float(_db(peak)), 1),
        "rms_dbfs": round(overall, 1),
        "loudest_frame_dbfs": round(loudest, 1),
        "noise_floor_dbfs": round(noise_floor, 1),
        "signal_level_dbfs": round(signal_level, 1),
        "snr_estimate_db": round(snr, 1),
        "silence_ratio": round(silence_ratio, 3),
        "clipping_ratio": round(clip, 5),
        "noise_flatness": round(flat, 3),
    }

    def issue(code: str, label: str, penalty: int, detail: str, fatal_issue: bool = False):
        nonlocal fatal
        issues.append({"code": code, "label": label, "penalty": penalty, "detail": detail})
        if fatal_issue:
            fatal = True

    # Silence / low level
    if loudest < cfg["silence_dbfs"]:
        issue("silence", "Silent", 100, f"No usable sound (loudest part {loudest:.0f} dBFS).", True)
    else:
        if silence_ratio > 0.9:
            issue("mostly_silent", "Mostly silent", 35, f"{silence_ratio * 100:.0f}% of the audio is silent.")
        elif silence_ratio > 0.7:
            issue("mostly_silent", "Long silences", 15, f"{silence_ratio * 100:.0f}% of the audio is silent.")
        if loudest < cfg["low_signal_dbfs"]:
            issue("low_signal", "Low signal strength", 30,
                  f"Loudest part is only {loudest:.0f} dBFS; the source may be far away or the gain too low.")
        elif loudest < cfg["low_signal_dbfs"] + 10:
            issue("low_signal", "Weak signal", 10, f"Loudest part is {loudest:.0f} dBFS.")

    # Clipping
    if clip >= cfg["clipping_severe_ratio"]:
        issue("clipping", "Severe clipping", 40, f"{clip * 100:.2f}% of samples are clipped (distorted).")
    elif clip >= cfg["clipping_warning_ratio"]:
        issue("clipping", "Clipping", 15, f"{clip * 100:.2f}% of samples are clipped.")

    # Background noise
    if not fatal:
        if snr < 6 and flat > 0.35 and noise_floor > -45:
            issue("noise", "Excessive background noise", 30,
                  f"Broadband noise dominates (SNR about {snr:.0f} dB, noise floor {noise_floor:.0f} dBFS).")
        elif snr < 10 and flat > 0.3 and noise_floor > -50:
            issue("noise", "Noticeable background noise", 15, f"Estimated SNR about {snr:.0f} dB.")
    metrics["noise_level_exceeded"] = bool(noise_floor > cfg["noise_floor_limit_dbfs"])

    # Duration
    if duration < 0.3:
        issue("duration", "Too short", 100, f"Only {duration:.2f} s of audio.", True)
    elif duration < 1.0 and not compact:
        issue("duration", "Very short", 10, f"{duration:.2f} s is shorter than one analysis window.")

    if not compact:
        # Encoding problems
        if decode_warnings:
            issue("encoding", "Encoding problems", 15, "; ".join(decode_warnings)[:200])
        if sr < 16000:
            issue("encoding", "Narrow-band audio", 10, f"Sampling rate {sr} Hz removes high frequencies.")
        if bit_depth is not None and bit_depth <= 8:
            issue("encoding", "Low bit depth", 10, f"{bit_depth}-bit audio adds quantisation noise.")
        dc = float(np.mean(y)) if y.size else 0.0
        metrics["dc_offset"] = round(dc, 4)
        if abs(dc) > 0.05:
            issue("encoding", "DC offset", 5, f"Signal is offset by {dc:+.3f}.")
        # Missing frames
        drops = dropout_count(y, sr)
        metrics["dropouts"] = drops
        if drops:
            issue("missing_frames", "Missing audio frames", min(30, 10 * drops),
                  f"{drops} gap(s) of digital silence inside the signal (buffer drop-outs).")
        if expected_frames and decoded_frames is not None and decoded_frames < expected_frames * 0.98:
            missing = 1.0 - decoded_frames / float(expected_frames)
            metrics["missing_frame_ratio"] = round(missing, 3)
            issue("missing_frames", "Truncated stream", 20, f"{missing * 100:.1f}% of the expected frames are missing.")

    score = max(0, 100 - sum(i["penalty"] for i in issues))
    if fatal or score < 35:
        label = "Unusable"
    elif score < 60:
        label = "Poor"
    elif score < 80:
        label = "Acceptable"
    else:
        label = "Good"
    if fatal:
        score = 0
    return {"label": label, "score": int(score), "metrics": metrics, "issues": issues}
