"""Python reference implementation of the Google Teachable Machine (GTM) audio
input representation.

GTM audio models are TensorFlow.js "speech-commands" (BROWSER_FFT) models.
In the browser they read spectrogram frames from a Web Audio ``AnalyserNode``:

* AudioContext sample rate 44 100 Hz,
* ``analyser.fftSize = 2048`` (2 x the speech-commands fftSize of 1024),
  Blackman window, no smoothing,
* a new frame every 1024 samples (23.2 ms),
* each frame = ``getFloatFrequencyData`` = 20*log10(|X[k]| / 2048), first 232 bins (0-5 kHz),
* 43 frames = one example of about 1 second, shape (43, 232, 1),
* the example is z-score normalised before ``model.predict``.

The web application computes exactly this in JavaScript (``static/js/gtm.js``)
for uploaded files and live microphone windows. This module mirrors it
number-for-number (same dither sequence, same window choice, same rounding and
padding), so the GTM training script sees exactly the input the browser gives
the model. ``tests/test_gtm.py`` checks both implementations agree.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

GTM_SAMPLE_RATE = 44100
GTM_HOP = 1024  # speech-commands fftSize (frame period)
GTM_FFT = 2048  # AnalyserNode.fftSize
GTM_FRAMES = 43
GTM_BINS = 232
GTM_INPUT_SAMPLES = GTM_FFT + (GTM_FRAMES - 1) * GTM_HOP  # 45 056 samples (~1.02 s)
GTM_DITHER = 3.4e-5  # peak-to-peak of the browser's uniform dither
GTM_DITHER_SEED = 7
GTM_EPSILON = 1e-7


@lru_cache(maxsize=1)
def blackman(n: int = GTM_FFT) -> np.ndarray:
    """Web Audio AnalyserNode window (alpha = 0.16), stored as float32 like gtm.js."""
    k = np.arange(n)
    w = (0.42 - 0.5 * np.cos(2 * np.pi * k / n) + 0.08 * np.cos(4 * np.pi * k / n)).astype(np.float32).astype(np.float64)
    w.flags.writeable = False  # cached and shared
    return w


@lru_cache(maxsize=4)
def browser_dither(frames: int = GTM_FRAMES) -> np.ndarray:
    """The (frames, 2048) dither gtm.js adds: an LCG (seed 7) consumed frame by frame.

    A real microphone never produces exact zeros; the dither keeps log(0) out of
    the spectrogram of zero-padded or digitally silent audio.
    """
    s = GTM_DITHER_SEED
    out = np.empty(frames * GTM_FFT, dtype=np.float64)
    for i in range(out.size):
        s = (1664525 * s + 1013904223) % 4294967296
        out[i] = (s / 4294967296 - 0.5) * GTM_DITHER
    out.flags.writeable = False
    return out.reshape(frames, GTM_FFT)


def gtm_spectrogram(x44k: np.ndarray, dither: bool | np.ndarray = True, frames: int = GTM_FRAMES) -> np.ndarray:
    """(frames, 232) dB spectrogram exactly as gtm.js ``spectrogram()`` computes it.

    Samples beyond the end of ``x44k`` count as zeros (as in the browser).
    ``dither`` may also be a (frames, 2048) array of per-frame dither values.
    """
    x = np.asarray(x44k, dtype=np.float32).astype(np.float64)
    need = GTM_FFT + (frames - 1) * GTM_HOP
    if x.size < need:
        x = np.pad(x, (0, need - x.size))
    idx = np.arange(GTM_FFT)[None, :] + GTM_HOP * np.arange(frames)[:, None]
    frames_ = x[idx]
    if isinstance(dither, np.ndarray):
        frames_ = frames_ + dither
    elif dither:
        frames_ = frames_ + browser_dither(frames)
    frames_ = frames_ * blackman()[None, :]
    mag = np.abs(np.fft.rfft(frames_, n=GTM_FFT, axis=1)[:, :GTM_BINS]) / GTM_FFT
    return (20.0 * np.log10(np.maximum(mag, 1e-12))).astype(np.float32)


def normalize(spec: np.ndarray) -> np.ndarray:
    """speech-commands ``normalize()``: (x - mean) / (std + epsilon) over the whole example."""
    s = np.asarray(spec, dtype=np.float64)
    return ((s - s.mean()) / (s.std() + GTM_EPSILON)).astype(np.float32)


def best_window_start(y: np.ndarray, sr: int = GTM_SAMPLE_RATE,
                      window_seconds: float = GTM_INPUT_SAMPLES / GTM_SAMPLE_RATE) -> int:
    """Start sample of the ~1 s window centred on the loudest 50 ms frame.

    GTM's native input is about one second. For longer segments we classify
    the second that contains the most energy (where the event is). Same rule
    and rounding (JavaScript ``Math.round``) as ``bestWindowStart`` in gtm.js.
    """
    win = int(round(window_seconds * sr))
    y = np.asarray(y, dtype=np.float32)
    if y.size <= win:
        return 0
    frame = max(1, int(np.floor(0.05 * sr)))
    n = y.size // frame
    energy = np.square(y[: n * frame].astype(np.float64)).reshape(n, frame).sum(axis=1)
    centre = (int(np.argmax(energy)) + 0.5) * frame
    start = int(np.floor(centre - win / 2 + 0.5))
    return int(min(max(0, start), y.size - win))


def fit_input(y: np.ndarray) -> np.ndarray:
    """Exactly GTM_INPUT_SAMPLES samples: shorter clips are centred in zeros (gtm.js ``fitInput``)."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == GTM_INPUT_SAMPLES:
        return y
    if y.size > GTM_INPUT_SAMPLES:
        return y[:GTM_INPUT_SAMPLES].copy()
    out = np.zeros(GTM_INPUT_SAMPLES, dtype=np.float32)
    off = (GTM_INPUT_SAMPLES - y.size) // 2
    out[off: off + y.size] = y
    return out


def segment_bounds(n_samples: int, start_s: float, end_s: float) -> tuple[int, int]:
    """Sample range gtm.js ``classifySegment`` cuts for a segment given in seconds."""
    s0 = max(0, int(np.floor(start_s * GTM_SAMPLE_RATE)))
    s1 = min(n_samples, max(s0 + 1, int(np.ceil(end_s * GTM_SAMPLE_RATE))))
    return s0, s1


def segment_window(y44k: np.ndarray, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """The ~1 s window gtm.js ``classifySegment`` takes from one analysis segment.

    Returns (window samples, window start in samples of ``y44k``). The window is
    not yet fitted: pass it through ``gtm_input``.
    """
    s0, s1 = segment_bounds(len(y44k), start_s, end_s)
    seg = y44k[s0:s1]
    ws = best_window_start(seg)
    return seg[ws: ws + GTM_INPUT_SAMPLES], s0 + ws


def gtm_input(window: np.ndarray, dither: bool = True) -> np.ndarray:
    """(43, 232) normalised model input for one window (gtm.js ``classifyWindow``)."""
    return normalize(gtm_spectrogram(fit_input(window), dither=dither))
