"""Audio -> the exact GTM model inputs the web application produces.

Upload path (``src/services/pipeline.py`` + ``static/js/gtm.js``):
    decode -> ``preprocess_audio`` (settings) -> processed.wav (16-bit PCM,
    22.05 kHz) -> 2 s segments every 1 s. The browser decodes processed.wav at
    44.1 kHz and, for every non-silent segment, classifies the loudest ~1 s
    window of that segment.

Live path (``static/js/live.js``): the raw 2 s microphone window is resampled
to 44.1 kHz, quiet windows are boosted, and the loudest ~1 s is classified
(no noise reduction or silence trimming).

The GTM training script, the comparison report and the tests all use these
functions, so the model is trained and evaluated on exactly what the browser
feeds it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_preprocessing import PreprocessConfig, decode_audio, preprocess_audio, segment_audio, select_event_segments
from audio_preprocessing.loader import resample
from audio_preprocessing.preprocess import Segment, remove_dc, to_mono
from config import load_default_settings
from feature_extraction.gtm_spectrogram import (
    GTM_FRAMES,
    GTM_HOP,
    GTM_INPUT_SAMPLES,
    GTM_SAMPLE_RATE,
    best_window_start,
    browser_dither,
    fit_input,
    gtm_input,
    gtm_spectrogram,
    segment_bounds,
)

SILENT_SEGMENT_DBFS = -60.0  # src/services/pipeline.py: segments at or below this are not classified
EVENT_WITHIN_DB = 15.0  # src/services/pipeline.py: event segments for the recording-level result
MAX_TRAINING_SEGMENTS = 8  # feature_extraction/build_features.py
BACKGROUND_CLASS = "Background Noise"


@dataclass
class GTMAudio:
    """One recording as the browser sees it."""

    y44: np.ndarray  # 44.1 kHz samples the GTM windows are cut from
    segments: list[Segment]  # analysis segments (times on the same timeline as y44)
    live: bool = False  # live-monitor path (quiet windows are boosted)


def pcm16_roundtrip(y: np.ndarray) -> np.ndarray:
    """processed.wav is written as 16-bit PCM; the browser decodes those samples."""
    q = np.rint(np.clip(np.nan_to_num(y), -1.0, 1.0) * 32767.0)
    return (q / 32768.0).astype(np.float32)


def segmentation_settings(settings: dict | None = None) -> tuple[PreprocessConfig, float, float]:
    settings = settings or load_default_settings()
    seg = settings["segmentation"]
    return (PreprocessConfig.from_settings(settings), float(seg["segment_seconds"]),
            float(seg["segment_hop_seconds"]))


def upload_audio(samples: np.ndarray, sr: int, settings: dict | None = None) -> GTMAudio:
    """The upload path: pre-processing, processed.wav round trip, segmentation."""
    pre_cfg, seg_s, hop_s = segmentation_settings(settings)
    pre = preprocess_audio(samples, sr, pre_cfg)
    y44 = resample(pcm16_roundtrip(pre.y), pre.sr, GTM_SAMPLE_RATE)
    return GTMAudio(y44=y44, segments=segment_audio(pre.y, pre.sr, seg_s, hop_s))


def live_audio(samples: np.ndarray, sr: int, settings: dict | None = None) -> GTMAudio:
    """A recording cut into live-monitor windows (raw audio, no noise reduction or trimming)."""
    _, seg_s, hop_s = segmentation_settings(settings)
    y = to_mono(np.nan_to_num(samples))
    y22 = resample(remove_dc(y)[0], sr, 22050)
    return GTMAudio(y44=resample(y, sr, GTM_SAMPLE_RATE), segments=segment_audio(y22, 22050, seg_s, hop_s), live=True)


def load(path: str | Path, path_kind: str = "upload", settings: dict | None = None) -> GTMAudio:
    dec = decode_audio(path)
    if path_kind == "live":
        return live_audio(dec.samples, dec.sample_rate, settings)
    return upload_audio(dec.samples, dec.sample_rate, settings)


def app_segments(audio: GTMAudio) -> list[tuple[Segment, bool]]:
    """(segment, is_event) for every segment the upload page sends to GTM (non-silent ones)."""
    if not audio.segments:
        return []
    loudest = max(s.rms_dbfs for s in audio.segments)
    return [(s, s.rms_dbfs >= loudest - EVENT_WITHIN_DB) for s in audio.segments if s.rms_dbfs > SILENT_SEGMENT_DBFS]


def training_segments(audio: GTMAudio, class_label: str) -> list[Segment]:
    """Segments used for training and evaluation - the rule of feature_extraction/build_features.py."""
    segs = audio.segments
    if class_label != BACKGROUND_CLASS:
        segs = select_event_segments(segs, keep_within_db=EVENT_WITHIN_DB)
    if len(segs) > MAX_TRAINING_SEGMENTS:
        segs = sorted(sorted(segs, key=lambda s: -s.rms_dbfs)[:MAX_TRAINING_SEGMENTS], key=lambda s: s.index)
    return segs


def _live_gain(y: np.ndarray) -> float:
    """live.js boosts quiet microphone windows before classifying them (same constants)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return min(15.0, 0.85 / peak) if 0.0001 < peak < 0.40 else 1.0


def _window(audio: GTMAudio, seg: Segment) -> tuple[np.ndarray, int, float]:
    """(window samples, window start sample, gain) exactly as the browser picks them."""
    s0, s1 = segment_bounds(len(audio.y44), round(seg.start, 3), round(seg.end, 3))
    y = audio.y44[s0:s1]
    gain = _live_gain(y) if audio.live else 1.0
    if gain != 1.0:
        y = np.clip(y * gain, -1.0, 1.0).astype(np.float32)
    ws = best_window_start(y)
    return y[ws: ws + GTM_INPUT_SAMPLES], s0 + ws, gain


def segment_input(audio: GTMAudio, seg: Segment) -> tuple[np.ndarray, float]:
    """(43, 232) model input for one segment and the window start in seconds."""
    window, ws, _ = _window(audio, seg)
    return gtm_input(window), ws / GTM_SAMPLE_RATE


def context_spectrogram(audio: GTMAudio, seg: Segment, extra_frames: int) -> np.ndarray:
    """dB spectrogram of the browser window plus ``extra_frames`` frames on each side.

    Rows ``extra_frames .. extra_frames + 42`` are the window the browser
    classifies; the extra rows allow time-shift augmentation during training.
    """
    extra = extra_frames * GTM_HOP
    window, ws, gain = _window(audio, seg)
    if window.size < GTM_INPUT_SAMPLES:  # the whole clip is shorter than one window: centred in zeros
        ctx = np.concatenate([np.zeros(extra, np.float32), fit_input(window), np.zeros(extra, np.float32)])
    else:
        lo, hi = ws - extra, ws + GTM_INPUT_SAMPLES + extra
        ctx = np.zeros(hi - lo, dtype=np.float32)
        a, b = max(lo, 0), min(hi, len(audio.y44))
        y = audio.y44[a:b]
        ctx[a - lo: b - lo] = np.clip(y * gain, -1.0, 1.0) if gain != 1.0 else y  # the browser only clips boosted audio
    return gtm_spectrogram(ctx, dither=_context_dither(extra_frames), frames=GTM_FRAMES + 2 * extra_frames)


def _context_dither(extra_frames: int) -> np.ndarray:
    """Dither for a context spectrogram whose centre rows get the browser's own dither frames 0..42."""
    d = browser_dither(GTM_FRAMES + 2 * extra_frames)
    order = np.r_[GTM_FRAMES:GTM_FRAMES + extra_frames, 0:GTM_FRAMES,
                  GTM_FRAMES + extra_frames:GTM_FRAMES + 2 * extra_frames]
    return d[order]
