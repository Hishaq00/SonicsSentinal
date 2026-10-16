"""Signal-processing building blocks for the synthetic dataset generator.

Everything here is plain NumPy/SciPy so that each sound is explainable:
oscillators, noise colours, filters, envelopes, formant synthesis, room
impulse responses, distance and recording-device simulation.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 22050


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def n_samples(seconds: float, sr: int = SR) -> int:
    return max(1, int(round(seconds * sr)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 1e-9 else x


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def fade(x: np.ndarray, fade_in: float = 0.005, fade_out: float = 0.01, sr: int = SR) -> np.ndarray:
    y = x.copy()
    fi = min(len(y), n_samples(fade_in, sr))
    fo = min(len(y), n_samples(fade_out, sr))
    if fi > 1:
        y[:fi] *= np.linspace(0.0, 1.0, fi)
    if fo > 1:
        y[-fo:] *= np.linspace(1.0, 0.0, fo)
    return y


def place(canvas: np.ndarray, x: np.ndarray, start: int, gain: float = 1.0) -> np.ndarray:
    """Add ``x`` into ``canvas`` at sample ``start`` (clipped to the canvas)."""
    if start >= len(canvas) or start + len(x) <= 0:
        return canvas
    s0 = max(0, start)
    x0 = s0 - start
    e0 = min(len(canvas), start + len(x))
    canvas[s0:e0] += gain * x[x0:x0 + (e0 - s0)]
    return canvas


def exp_env(n: int, tau: float, sr: int = SR) -> np.ndarray:
    return np.exp(-np.arange(n) / max(1e-6, tau * sr))


def adsr(n: int, attack: float, decay: float, sustain: float, release: float, sr: int = SR) -> np.ndarray:
    a, d, r = n_samples(attack, sr), n_samples(decay, sr), n_samples(release, sr)
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def smooth_random(rng: np.random.Generator, n: int, rate_hz: float, sr: int = SR) -> np.ndarray:
    """Band-limited random curve (values roughly in [-1, 1]) for jitter / drift."""
    knots = max(2, int(n / sr * rate_hz) + 2)
    pts = rng.uniform(-1, 1, knots)
    return np.interp(np.linspace(0, knots - 1, n), np.arange(knots), pts)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def colored_noise(rng: np.random.Generator, n: int, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent power spectrum: 0 = white, 1 = pink, 2 = brown."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec = spec / (f ** (exponent / 2.0))
    y = np.fft.irfft(spec, n)
    return y / (np.std(y) + 1e-12)


def white(rng, n):
    return rng.standard_normal(n)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def _clip_freq(f: float, sr: int) -> float:
    return float(min(max(f, 10.0), sr / 2 * 0.98))


def lowpass(x, fc, sr=SR, order=4):
    sos = signal.butter(order, _clip_freq(fc, sr), btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def highpass(x, fc, sr=SR, order=4):
    sos = signal.butter(order, _clip_freq(fc, sr), btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def bandpass(x, lo, hi, sr=SR, order=4):
    lo, hi = _clip_freq(lo, sr), _clip_freq(hi, sr)
    if hi <= lo * 1.05:
        hi = min(sr / 2 * 0.98, lo * 1.2)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def peaking_eq(x, f0, gain_db, q=1.0, sr=SR):
    """RBJ cookbook peaking filter."""
    f0 = _clip_freq(f0, sr)
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return signal.lfilter(b / den[0], den / den[0], x)


def resonator(x, freq, bandwidth, sr=SR):
    """Two-pole resonator (formant / body resonance), unity gain at the peak."""
    freq = _clip_freq(freq, sr)
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2 * np.pi * freq / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return signal.lfilter(b, a, x)


# --------------------------------------------------------------------------- #
# Oscillators and voice synthesis
# --------------------------------------------------------------------------- #
def phase_from_freq(freq: np.ndarray, sr: int = SR, phase0: float = 0.0) -> np.ndarray:
    return phase0 + 2 * np.pi * np.cumsum(freq) / sr


def harmonic_tone(freq: np.ndarray, sr: int = SR, n_harm: int = 10, tilt_db: float = -6.0,
                  odd_only: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    """Band-limited harmonic oscillator following a frequency curve.

    ``tilt_db`` is the level change per octave (e.g. -12 dB = dull, -3 dB = bright).
    Harmonics above 0.45*sr are muted sample-by-sample, so there is no aliasing.
    """
    freq = np.asarray(freq, dtype=np.float64)
    phase = phase_from_freq(freq, sr, rng.uniform(0, 2 * np.pi) if rng is not None else 0.0)
    out = np.zeros_like(freq)
    limit = 0.45 * sr
    for k in range(1, n_harm + 1):
        if odd_only and k % 2 == 0:
            continue
        amp = 10 ** (tilt_db * np.log2(k) / 20.0)
        mask = (k * freq) < limit
        if not mask.any():
            break
        out += amp * np.sin(k * phase) * mask
    return out


def glottal_source(f0: np.ndarray, rng: np.random.Generator, sr: int = SR, tilt_db: float = -9.0,
                   jitter: float = 0.01, shimmer: float = 0.05, breath: float = 0.05) -> np.ndarray:
    """Voiced excitation: harmonic series with jitter, shimmer and breath noise."""
    n = len(f0)
    f0 = f0 * (1.0 + jitter * smooth_random(rng, n, 60.0, sr))
    n_harm = int(min(80, 0.45 * sr / max(40.0, float(np.min(f0)))))
    src = harmonic_tone(f0, sr, n_harm=n_harm, tilt_db=tilt_db, rng=rng)
    src *= 1.0 + shimmer * smooth_random(rng, n, 40.0, sr)
    if breath > 0:
        src += breath * np.std(src) * highpass(white(rng, n), 1500, sr, 2)
    return src


VOWELS = {
    # F1, F2, F3 in Hz (adult averages)
    "a": (800, 1250, 2600),
    "e": (500, 1850, 2600),
    "i": (320, 2300, 3000),
    "o": (500, 900, 2500),
    "u": (350, 800, 2400),
    "ae": (700, 1700, 2600),
    "uh": (600, 1200, 2500),
}


def formant_filter(src: np.ndarray, formants, bandwidths=(90, 110, 170), sr: int = SR,
                   gains=(1.0, 0.7, 0.4)) -> np.ndarray:
    out = np.zeros_like(src)
    for f, bw, g in zip(formants, bandwidths, gains):
        out += g * resonator(src, f, bw, sr)
    return out


def vowel_sequence(src: np.ndarray, vowels: list[str], sr: int = SR, shift: float = 1.0,
                   bw_scale: float = 1.0) -> np.ndarray:
    """Filter ``src`` through a sequence of vowels with cross-fades between them."""
    n = len(src)
    k = max(1, len(vowels))
    edges = np.linspace(0, n, k + 1).astype(int)
    out = np.zeros(n)
    xf = n_samples(0.03, sr)
    for i, v in enumerate(vowels):
        f = [x * shift for x in VOWELS.get(v, VOWELS["a"])]
        a, b = max(0, edges[i] - xf), min(n, edges[i + 1] + xf)
        seg = formant_filter(src[a:b], f, [bw * bw_scale for bw in (90, 110, 170)], sr)
        w = np.ones(b - a)
        if i > 0:
            w[: min(2 * xf, len(w))] = np.linspace(0, 1, min(2 * xf, len(w)))
        if i < k - 1:
            w[-min(2 * xf, len(w)):] = np.linspace(1, 0, min(2 * xf, len(w)))
        out[a:b] += seg * w
    return out


# --------------------------------------------------------------------------- #
# Impacts
# --------------------------------------------------------------------------- #
def modal_impact(rng: np.random.Generator, freqs, decays, amps=None, dur: float = 0.3, sr: int = SR) -> np.ndarray:
    """Sum of exponentially decaying partials (metal, glass, wood strikes)."""
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    amps = amps if amps is not None else [1.0 / (i + 1) for i in range(len(freqs))]
    for f, d, a in zip(freqs, decays, amps):
        if f < sr / 2 * 0.95:
            out += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / d)
    click = n_samples(0.002, sr)
    out[:click] += 0.3 * rng.standard_normal(click) * np.linspace(1, 0, click)
    return out


def thud(rng: np.random.Generator, freq: float = 90.0, decay: float = 0.06, dur: float = 0.25, sr: int = SR) -> np.ndarray:
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    f = freq * (1.0 + 0.6 * np.exp(-t / 0.01))  # pitch drops right after impact
    body = np.sin(phase_from_freq(f, sr)) * np.exp(-t / decay)
    noise = lowpass(rng.standard_normal(n), 600, sr) * np.exp(-t / (decay * 0.5))
    return body + 0.5 * noise / (np.std(noise) + 1e-9) * 0.3


# --------------------------------------------------------------------------- #
# Environment, distance and device simulation
# --------------------------------------------------------------------------- #
ENVIRONMENTS = {
    "indoor-small-room": {"indoor": True, "rt60": (0.25, 0.5), "early": (3, 6), "echo": 0},
    "indoor-hall": {"indoor": True, "rt60": (0.8, 1.6), "early": (4, 8), "echo": 0},
    "indoor-factory": {"indoor": True, "rt60": (1.0, 2.0), "early": (5, 9), "echo": 0},
    "outdoor-street": {"indoor": False, "rt60": (0.15, 0.4), "early": (0, 2), "echo": 3},
    "outdoor-open": {"indoor": False, "rt60": (0.05, 0.15), "early": (0, 1), "echo": 1},
}


def room_impulse_response(rng: np.random.Generator, rt60: float, n_early: int, sr: int = SR) -> np.ndarray:
    """Synthetic RIR: direct path + early reflections + exponentially decaying diffuse tail."""
    length = n_samples(min(2.5, rt60 * 1.2) + 0.05, sr)
    t = np.arange(length) / sr
    tail = rng.standard_normal(length) * np.exp(-6.9 * t / max(rt60, 0.05))  # -60 dB at rt60
    tail = lowpass(tail, rng.uniform(3500, 8000), sr, 2)
    onset = n_samples(rng.uniform(0.005, 0.02), sr)
    tail[:onset] = 0
    h = 0.35 * tail / (np.max(np.abs(tail)) + 1e-9)
    h[0] = 1.0
    for _ in range(n_early):
        d = n_samples(rng.uniform(0.004, 0.04), sr)
        if d < length:
            h[d] += rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
    return h


def apply_environment(x: np.ndarray, env: str, distance_m: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Reverberation / echoes whose share grows with distance, plus air absorption."""
    cfg = ENVIRONMENTS[env]
    rt60 = rng.uniform(*cfg["rt60"])
    h = room_impulse_response(rng, rt60, int(rng.integers(cfg["early"][0], cfg["early"][1] + 1)), sr)
    wet = signal.fftconvolve(x, h)[: len(x)]
    wet_only = wet - x  # everything except the direct path
    # Direct-to-reverberant ratio falls with distance (critical distance ~2-4 m indoors).
    critical = 3.0 if cfg["indoor"] else 12.0
    reverb_gain = min(2.5, (distance_m / critical) ** 0.9) * (1.0 if cfg["indoor"] else 0.5)
    y = x + reverb_gain * wet_only
    # Discrete echoes from buildings (outdoor)
    for _ in range(cfg["echo"] if cfg["echo"] == 0 else int(rng.integers(0, cfg["echo"] + 1))):
        delay = n_samples(rng.uniform(0.06, 0.45), sr)
        if delay < len(y):
            echo = lowpass(x, rng.uniform(2000, 5000), sr, 2) * db_to_gain(-rng.uniform(8, 20))
            y[delay:] += echo[: len(y) - delay]
    # Air absorption: farther sources lose high frequencies.
    if distance_m > 3:
        cutoff = float(np.clip(11000 * (3.0 / distance_m) ** 0.45, 2500, 10500))
        y = lowpass(y, cutoff, sr, 2)
    return y


DEVICES = {
    "studio-recorder": {"hp": 30, "lp": 10500, "noise_db": -82, "comp": 0.0, "eq": None, "mulaw": False},
    "smartphone": {"hp": 120, "lp": 9000, "noise_db": -68, "comp": 0.4, "eq": (3000, 3.0), "mulaw": False},
    "laptop-mic": {"hp": 200, "lp": 7500, "noise_db": -62, "comp": 0.3, "eq": (2500, 5.0), "mulaw": False},
    "cctv-camera": {"hp": 300, "lp": 5000, "noise_db": -55, "comp": 0.6, "eq": (1800, 4.0), "mulaw": True},
    "body-camera": {"hp": 150, "lp": 7000, "noise_db": -60, "comp": 0.5, "eq": (2200, 3.0), "mulaw": False},
    "headset": {"hp": 180, "lp": 8000, "noise_db": -70, "comp": 0.2, "eq": (1500, 2.0), "mulaw": False},
}


def compress(x: np.ndarray, amount: float, sr: int = SR) -> np.ndarray:
    """Simple AGC-style compressor (envelope follower + gain curve)."""
    if amount <= 0:
        return x
    env = np.abs(signal.hilbert(x)) if len(x) < 400000 else np.abs(x)
    env = lowpass(env, 15, sr, 1) + 1e-6
    target = np.mean(env)
    gain = (target / env) ** (0.5 * amount)
    gain = np.clip(gain, 0.25, 4.0)
    return x * gain


def mulaw_roundtrip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """8-bit mu-law companding, as used by many cheap IP camera audio codecs."""
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127) / 127
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)


def apply_device(x: np.ndarray, device: str, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    cfg = DEVICES[device]
    y = highpass(x, cfg["hp"], sr, 2)
    y = lowpass(y, cfg["lp"], sr, 4)
    if cfg["eq"]:
        y = peaking_eq(y, cfg["eq"][0], cfg["eq"][1], 1.2, sr)
    y = compress(y, cfg["comp"], sr)
    y = peak_normalize(y, 0.7)
    y = y + db_to_gain(cfg["noise_db"]) * colored_noise(rng, len(y), 0.6)
    if cfg["mulaw"]:
        y = mulaw_roundtrip(y)
    return y


def mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that the *active* part of ``sig`` sits ``snr_db`` above it."""
    frame = 1024
    n = len(sig) // frame
    if n >= 2:
        fr = sig[: n * frame].reshape(n, frame)
        e = np.sqrt(np.mean(fr ** 2, axis=1))
        active = e[e >= 0.1 * e.max()]
        ps = float(np.mean(active ** 2)) if active.size else rms(sig) ** 2
    else:
        ps = rms(sig) ** 2
    pn = rms(noise) ** 2 + 1e-12
    gain = np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    return sig + gain * noise[: len(sig)]
