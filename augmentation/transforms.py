"""Training-time audio augmentations.

Each transform takes ``(y, sr, rng)`` and returns ``(y_augmented, description)``.
They are applied only to *training* recordings; augmented copies keep the
parent's split and are never counted as unique original clips.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _peak(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if y.size else 0.0


def _limit(y: np.ndarray) -> np.ndarray:
    p = _peak(y)
    return y / p * 0.98 if p > 0.98 else y


def _active_power(y: np.ndarray) -> float:
    frame = 1024
    n = len(y) // frame
    if n < 2:
        return float(np.mean(y ** 2)) + 1e-12
    e = np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1)
    act = e[e >= 0.01 * e.max()]
    return float(np.mean(act)) + 1e-12


def add_background_noise(y, sr, rng, noise_bank: list[np.ndarray] | None = None, snr_range=(3.0, 25.0)):
    """Mix real background recordings (training split only) or coloured noise at a random SNR."""
    snr = float(rng.uniform(*snr_range))
    if noise_bank:
        noise = noise_bank[int(rng.integers(0, len(noise_bank)))]
        if len(noise) < len(y):
            noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
        start = int(rng.integers(0, len(noise) - len(y) + 1))
        noise = noise[start:start + len(y)]
        kind = "recorded background"
    else:
        exponent = float(rng.choice([0.0, 1.0, 2.0]))
        spec = np.fft.rfft(rng.standard_normal(len(y)))
        f = np.fft.rfftfreq(len(y))
        f[0] = f[1]
        noise = np.fft.irfft(spec / f ** (exponent / 2), len(y))
        kind = {0.0: "white", 1.0: "pink", 2.0: "brown"}[exponent] + " noise"
    gain = np.sqrt(_active_power(y) / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr / 10))
    return _limit(y + gain * noise), f"{kind} at {snr:.0f} dB SNR"


def time_shift(y, sr, rng, max_seconds: float = 0.5):
    # At most a quarter of the clip, so a short clip is never shifted out of itself.
    limit = min(max_seconds, 0.25 * len(y) / sr)
    shift = int(rng.uniform(-limit, limit) * sr)
    out = np.zeros_like(y)
    if shift >= 0:
        out[shift:] = y[: len(y) - shift]
    else:
        out[:shift] = y[-shift:]
    return out, f"time shift {shift / sr:+.2f} s"


def pitch_shift(y, sr, rng, semitones=(-2.0, 2.0)):
    import librosa

    steps = float(rng.uniform(*semitones))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps).astype(np.float32), f"pitch {steps:+.1f} st"


def time_stretch(y, sr, rng, rates=(0.85, 1.15)):
    import librosa

    rate = float(rng.uniform(*rates))
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32), f"time stretch x{rate:.2f}"


def volume(y, sr, rng, gain_db=(-12.0, 6.0)):
    g = float(rng.uniform(*gain_db))
    return _limit(y * 10 ** (g / 20)), f"volume {g:+.0f} dB"


def reverb(y, sr, rng, rt60=(0.2, 0.8)):
    """Limited reverberation with a synthetic exponentially-decaying impulse response."""
    t60 = float(rng.uniform(*rt60))
    n = int(sr * min(1.2, t60 * 1.2))
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / t60)
    h[: int(0.005 * sr)] = 0
    h = 0.3 * h / (np.max(np.abs(h)) + 1e-9)
    h[0] = 1.0
    wet = signal.fftconvolve(y, h)[: len(y)]
    return _limit(wet / (_peak(wet) + 1e-9) * _peak(y)), f"reverb RT60 {t60:.2f} s"


def distance_simulation(y, sr, rng, distance=(2.0, 25.0)):
    """Farther source: quieter, duller (air absorption) and more reverberant."""
    d = float(rng.uniform(*distance))
    cutoff = float(np.clip(11000 * (2.0 / d) ** 0.5, 2000, sr / 2 * 0.95))
    sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
    out = signal.sosfilt(sos, y) / max(1.0, d / 2.0)
    out, _ = reverb(out, sr, rng, rt60=(0.2, 0.3 + 0.03 * d))
    return out.astype(np.float32), f"distance {d:.0f} m"


DEVICE_PROFILES = {
    "phone": (150.0, 7000.0),
    "laptop": (200.0, 6500.0),
    "cctv": (300.0, 4500.0),
    "headset": (120.0, 8000.0),
}


def device_simulation(y, sr, rng):
    name = list(DEVICE_PROFILES)[int(rng.integers(0, len(DEVICE_PROFILES)))]
    lo, hi = DEVICE_PROFILES[name]
    sos = signal.butter(3, [lo, min(hi, sr / 2 * 0.95)], btype="band", fs=sr, output="sos")
    out = signal.sosfilt(sos, y)
    out = np.tanh(2.0 * out / (_peak(out) + 1e-9)) * 0.8  # small-speaker / AGC saturation
    out = out + 10 ** (-60 / 20) * rng.standard_normal(len(out))
    return out.astype(np.float32), f"{name} microphone"


TRANSFORMS = {
    "noise": add_background_noise,
    "shift": time_shift,
    "pitch": pitch_shift,
    "stretch": time_stretch,
    "volume": volume,
    "reverb": reverb,
    "distance": distance_simulation,
    "device": device_simulation,
}


def random_chain(y, sr, rng, noise_bank=None, n_transforms: tuple[int, int] = (2, 3), protect_pitch: bool = False):
    """Apply 2-3 random transforms. ``protect_pitch`` avoids pitch shifts (e.g. for sirens)."""
    names = [n for n in TRANSFORMS if not (protect_pitch and n == "pitch")]
    k = int(rng.integers(n_transforms[0], n_transforms[1] + 1))
    chosen = list(rng.choice(names, size=k, replace=False))
    applied = []
    out = y.astype(np.float32)
    for name in chosen:
        if name == "noise":
            out, desc = add_background_noise(out, sr, rng, noise_bank)
        else:
            out, desc = TRANSFORMS[name](out, sr, rng)
        applied.append(desc)
    return _limit(np.nan_to_num(out)).astype(np.float32), applied
