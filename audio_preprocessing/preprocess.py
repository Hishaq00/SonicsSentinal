"""Audio pre-processing and fixed-duration segmentation.

The same functions are used for dataset preparation, model training and live
inference, so training and prediction always see identically prepared audio.

Pipeline order:
    mono -> DC removal -> resample -> noise reduction -> silence trim -> peak normalise
Segmentation then cuts fixed-length windows (with padding / truncation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loader import resample
from .noise_reduction import spectral_gate


@dataclass
class PreprocessConfig:
    target_sr: int = 22050
    trim_silence: bool = True
    trim_top_db: float = 40.0
    noise_reduction: bool = True
    nr_max_db: float = 12.0
    nr_min_dynamic_range_db: float = 12.0
    normalize_peak: float = 0.95

    @classmethod
    def from_settings(cls, settings: dict) -> "PreprocessConfig":
        p = settings.get("preprocessing", {})
        return cls(
            target_sr=int(p.get("target_sample_rate", 22050)),
            trim_silence=bool(p.get("trim_silence", True)),
            trim_top_db=float(p.get("trim_top_db", 40)),
            noise_reduction=bool(p.get("noise_reduction", True)),
            nr_max_db=float(p.get("noise_reduction_max_db", 12)),
            nr_min_dynamic_range_db=float(p.get("noise_reduction_min_dynamic_range_db", 12)),
            normalize_peak=float(p.get("normalize_peak", 0.95)),
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PreprocessResult:
    y: np.ndarray
    sr: int
    steps: list[dict] = field(default_factory=list)
    trim_offset: float = 0.0  # seconds removed from the start of the original audio
    original_duration: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.y) / float(self.sr) if self.sr else 0.0


@dataclass
class Segment:
    index: int
    start: float  # seconds, in the pre-processed timeline
    end: float
    audio: np.ndarray  # exactly segment_seconds long (padded / truncated)
    padded: bool
    rms_dbfs: float


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32)
    return samples.mean(axis=1).astype(np.float32)


def remove_dc(y: np.ndarray) -> tuple[np.ndarray, float]:
    offset = float(np.mean(y)) if y.size else 0.0
    return (y - offset).astype(np.float32), offset


def frame_rms(y: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    if y.size < frame:
        return np.array([np.sqrt(np.mean(y ** 2))]) if y.size else np.array([0.0])
    n = 1 + (y.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(y[idx] ** 2, axis=1))


def trim_silence(y: np.ndarray, top_db: float = 40.0, frame: int = 2048, hop: int = 512) -> tuple[np.ndarray, int, int]:
    """Remove leading / trailing audio that is ``top_db`` below the loudest frame."""
    rms = frame_rms(y, frame, hop)
    peak = rms.max() if rms.size else 0.0
    if peak <= 0:
        return y, 0, len(y)
    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / peak)
    active = np.flatnonzero(db > -top_db)
    if active.size == 0:
        return y, 0, len(y)
    start = int(active[0] * hop)
    end = int(min(len(y), active[-1] * hop + frame))
    return y[start:end], start, end


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> tuple[np.ndarray, float]:
    current = float(np.max(np.abs(y))) if y.size else 0.0
    if current < 1e-6:
        return y, 1.0
    gain = peak / current
    return (y * gain).astype(np.float32), gain


def fix_length(y: np.ndarray, length: int, mode: str = "center") -> tuple[np.ndarray, bool]:
    """Pad by tiling/looping or truncate to exactly ``length`` samples.
    
    Using tiling prevents digital zero silence padding which distorts
    acoustic features (crest factor, pitch, envelope, MFCCs) on short transient events
    like gunshots, glass breaks, screams, and dog barks.
    """
    n = len(y)
    if n == length:
        return y, False
    if n > length:
        start = (n - length) // 2 if mode == "center" else 0
        return y[start:start + length], False
    if n == 0:
        return np.zeros(length, dtype=np.float32), True
    repeats = int(np.ceil(length / n))
    tiled = np.tile(y, repeats)[:length]
    return tiled.astype(np.float32), True


def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -200.0
    return float(20.0 * np.log10(max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-10)))


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_audio(
    samples: np.ndarray,
    sr: int,
    cfg: PreprocessConfig | None = None,
    trim: bool | None = None,
) -> PreprocessResult:
    """Run the full pre-processing chain and log every step."""
    cfg = cfg or PreprocessConfig()
    steps: list[dict] = []
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    original_duration = (samples.shape[0] / float(sr)) if sr else 0.0

    y = to_mono(np.nan_to_num(samples))
    steps.append({"step": "Stereo-to-mono conversion",
                  "detail": "Already mono." if channels == 1 else f"{channels} channels averaged to mono."})

    y, offset = remove_dc(y)
    steps.append({"step": "DC offset removal", "detail": f"Offset {offset:+.4f} removed."})

    if sr != cfg.target_sr:
        y = resample(y, sr, cfg.target_sr)
        steps.append({"step": "Resampling", "detail": f"{sr} Hz -> {cfg.target_sr} Hz."})
        sr = cfg.target_sr
    else:
        steps.append({"step": "Resampling", "detail": f"Already {sr} Hz."})

    if cfg.noise_reduction:
        y, nr = spectral_gate(y, sr, max_reduction_db=cfg.nr_max_db,
                              min_dynamic_range_db=cfg.nr_min_dynamic_range_db)
        if nr.get("applied"):
            detail = (f"Spectral gating: noise floor {nr['noise_floor_db']} dB, up to "
                      f"{nr['max_reduction_db']:g} dB attenuation on {nr['attenuated_fraction'] * 100:.0f}% of bins.")
        else:
            detail = f"Skipped: {nr.get('reason', 'not needed')}."
        steps.append({"step": "Noise reduction", "detail": detail})
    else:
        steps.append({"step": "Noise reduction", "detail": "Disabled in settings."})

    trim_offset = 0.0
    do_trim = cfg.trim_silence if trim is None else trim
    if do_trim:
        trimmed, start, end = trim_silence(y, cfg.trim_top_db)
        if len(trimmed) >= int(0.25 * sr) and (start > 0 or end < len(y)):
            trim_offset = start / float(sr)
            removed = (len(y) - len(trimmed)) / float(sr)
            y = trimmed
            steps.append({"step": "Silence trimming",
                          "detail": f"Removed {removed:.2f} s of leading/trailing silence (threshold {cfg.trim_top_db:g} dB)."})
        else:
            steps.append({"step": "Silence trimming", "detail": "No leading/trailing silence found."})
    else:
        steps.append({"step": "Silence trimming", "detail": "Not applied (fixed live window)."})

    y, gain = peak_normalize(y, cfg.normalize_peak)
    steps.append({"step": "Amplitude normalization",
                  "detail": f"Peak normalised to {cfg.normalize_peak:g} (gain {20 * np.log10(max(gain, 1e-9)):+.1f} dB)."})

    return PreprocessResult(y=y.astype(np.float32), sr=sr, steps=steps,
                            trim_offset=trim_offset, original_duration=original_duration)


def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    min_tail_seconds: float = 0.25,
) -> list[Segment]:
    """Cut fixed-duration, possibly overlapping segments.

    * Clips shorter than one segment become one centred, zero-padded segment.
    * A final end-aligned segment covers any tail longer than ``min_tail_seconds``.
    """
    length = max(1, int(round(segment_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    segments: list[Segment] = []
    if len(y) <= length:
        audio, padded = fix_length(y, length, "center")
        return [Segment(0, 0.0, len(y) / sr, audio, padded, dbfs(y))]

    starts = list(range(0, len(y) - length + 1, hop))
    tail = len(y) - (starts[-1] + length)
    if tail > int(min_tail_seconds * sr):
        starts.append(len(y) - length)
    for i, s in enumerate(starts):
        chunk = y[s:s + length]
        segments.append(Segment(i, s / sr, (s + length) / sr, chunk.astype(np.float32), False, dbfs(chunk)))
    return segments


def select_event_segments(segments: list[Segment], keep_within_db: float = 15.0,
                          silence_dbfs: float = -60.0) -> list[Segment]:
    """Keep only segments that actually contain the event (used for training).

    Event classes (gunshot, glass...) often occupy a small part of a clip.
    Labelling every segment with the event class would teach the model that
    near-silence is a gunshot, so we keep segments within ``keep_within_db``
    of the loudest segment.
    """
    if not segments:
        return []
    loudest = max(s.rms_dbfs for s in segments)
    kept = [s for s in segments if s.rms_dbfs >= loudest - keep_within_db and s.rms_dbfs > silence_dbfs]
    return kept or [max(segments, key=lambda s: s.rms_dbfs)]
