"""Render a dataset recording the way the live microphone path delivers it.

``src/services/live_service.py`` receives a short window from the browser and classifies it on its own:

    getUserMedia (automatic gain control on; echo cancellation and noise suppression off)
      -> Web Audio at the device rate -> resampled to 22,050 Hz -> WAV window (``static/js/live.js``)
      -> preprocess_audio(window, trim=False)  [mono, DC, noise reduction, peak normalise **per window**]
      -> one feature vector for the whole window.

Training on the upload rendering alone is therefore not enough: an uploaded file is trimmed and
peak-normalised as a whole, while a live window is normalised on its own and carries the microphone's
band limit, self-noise and the browser's gain control. ``simulate_microphone`` adds that chain to a dataset
recording and ``live_windows`` cuts it into windows exactly as the live path does, so the same recordings
can be learned and evaluated in both renderings.

Everything here is deterministic for a given ``seed`` (the recording's Audio ID), so a rebuild of the
features produces the same rows.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import signal

from audio_preprocessing import PreprocessConfig, preprocess_audio, segment_audio
from audio_preprocessing.loader import resample
from audio_preprocessing.preprocess import to_mono

LIVE_SAMPLE_RATE = 22050  # static/js/live.js TARGET_SR

# Band limits and self-noise of the microphones people use with the app.
MICROPHONES = {
    "laptop-builtin": {"band": (120.0, 7500.0), "noise_db": -62.0, "tilt_db": 3.0},
    "phone-handheld": {"band": (90.0, 9000.0), "noise_db": -66.0, "tilt_db": 4.0},
    "usb-headset": {"band": (80.0, 8500.0), "noise_db": -70.0, "tilt_db": 2.0},
    "webcam": {"band": (200.0, 6500.0), "noise_db": -58.0, "tilt_db": 5.0},
}


def seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def _agc(y: np.ndarray, sr: int, target: float = 0.2, attack: float = 0.05, release: float = 0.6) -> np.ndarray:
    """The browser's automatic gain control: slow level following, limited to +-20 dB."""
    frame = max(1, int(0.01 * sr))
    n = len(y) // frame
    if n < 2:
        return y
    env = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9
    gain = np.ones(n)
    g = 1.0
    a_up, a_dn = np.exp(-frame / (attack * sr)), np.exp(-frame / (release * sr))
    for i, e in enumerate(env):
        want = float(np.clip(target / e, 0.1, 10.0))
        g = want + (g - want) * (a_up if want < g else a_dn)  # fast down, slow up
        gain[i] = g
    curve = np.interp(np.arange(len(y)), np.arange(n) * frame + frame / 2, gain,
                      left=float(gain[0]), right=float(gain[-1]))
    return np.tanh(y * curve * 1.1) * 0.9  # the capture chain also limits


def simulate_microphone(y: np.ndarray, sr: int, seed: int, mic: str | None = None) -> tuple[np.ndarray, str]:
    """Recording through a microphone into the browser: band limit, presence tilt, self-noise, gain control."""
    rng = np.random.default_rng(seed)
    mic = mic or list(MICROPHONES)[int(rng.integers(0, len(MICROPHONES)))]
    cfg = MICROPHONES[mic]
    y = to_mono(np.nan_to_num(np.asarray(y, dtype=np.float32)))
    lo, hi = cfg["band"]
    hi = min(hi, sr / 2 * 0.95)
    if lo < hi:
        y = signal.sosfilt(signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos"), y)
    # Small presence peak, as cheap microphones have around 3 kHz.
    f0 = float(rng.uniform(2200, 4000))
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2.0
    a = 10 ** (cfg["tilt_db"] * rng.uniform(0.5, 1.0) / 40.0)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    y = signal.lfilter(b / den[0], den / den[0], y)
    y = y * 10 ** (rng.uniform(-14.0, -2.0) / 20)  # distance from the microphone
    y = y + 10 ** (cfg["noise_db"] / 20) * rng.standard_normal(len(y)).astype(np.float32)
    y = _agc(y.astype(np.float32), sr)
    return np.clip(y, -1.0, 1.0).astype(np.float32), mic


def live_windows(samples: np.ndarray, sr: int, seed: int, cfg: PreprocessConfig, window_s: float, hop_s: float,
                 mic: str | None = None):
    """(list of pre-processed windows, microphone name) exactly as ``live_service.process_window`` sees them."""
    y, mic = simulate_microphone(samples, sr, seed, mic)
    # The browser only resamples; mono, DC removal, noise reduction and normalisation happen on the server
    # inside preprocess_audio, once per window.
    y22 = resample(y, sr, LIVE_SAMPLE_RATE) if sr != LIVE_SAMPLE_RATE else y
    out = []
    for seg in segment_audio(y22, LIVE_SAMPLE_RATE, window_s, hop_s):
        pre = preprocess_audio(seg.audio, LIVE_SAMPLE_RATE, cfg, trim=False)  # the live path never trims
        out.append((seg, pre.y))
    return out, mic
