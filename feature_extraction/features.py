"""Acoustic feature extraction for the Python classifier.

One fixed-length segment (default 2 s at 22 050 Hz) becomes one feature vector.
Every frame-level feature is computed from a single STFT and summarised with
statistics (mean / standard deviation, etc.). The *same* function is used for
training and for live prediction, and the configuration is saved inside the
model file so inference always matches training.

Feature groups (193 values with the default config):
    MFCC (20) mean + std ............................. 40
    Delta-MFCC (20) std .............................. 20
    Log-mel spectrogram, 32 summary bands, mean + std  64
    Chroma (12) mean + std ........................... 24
    Spectral contrast (7 bands) mean + std ........... 14
    Centroid, bandwidth, roll-off, flatness, ZCR ..... 10  (mean + std)
    RMS energy: mean, std, max, dynamic range, crest .  5
    Onset strength: mean, std, max, onset rate ........  4
    Tempo .............................................  1
    Pitch (YIN) on loud frames: median, p10, p90,
        std in semitones, stability .................  5
    Harmonic / percussive energy ratio (HPSS on mel) .  2
    Temporal envelope: peak position, attack time,
        decay slope, temporal centroid ...............  4
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

FEATURE_SET_VERSION = "ssf-1.0"


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20
    n_mels: int = 64
    mel_summary_bands: int = 32
    fmin: float = 20.0
    f0_min: float = 60.0
    f0_max: float = 2000.0
    segment_seconds: float = 2.0
    version: str = FEATURE_SET_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureConfig":
        if not d:
            return cls()
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


DEFAULT_CONFIG = FeatureConfig()


def feature_names(cfg: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    names: list[str] = []
    names += [f"mfcc{i + 1}_mean" for i in range(cfg.n_mfcc)]
    names += [f"mfcc{i + 1}_std" for i in range(cfg.n_mfcc)]
    names += [f"dmfcc{i + 1}_std" for i in range(cfg.n_mfcc)]
    names += [f"mel{i + 1}_mean" for i in range(cfg.mel_summary_bands)]
    names += [f"mel{i + 1}_std" for i in range(cfg.mel_summary_bands)]
    names += [f"chroma{i + 1}_mean" for i in range(12)]
    names += [f"chroma{i + 1}_std" for i in range(12)]
    names += [f"contrast{i + 1}_mean" for i in range(7)]
    names += [f"contrast{i + 1}_std" for i in range(7)]
    for base in ("centroid", "bandwidth", "rolloff", "flatness", "zcr"):
        names += [f"{base}_mean", f"{base}_std"]
    names += ["rms_mean", "rms_std", "rms_max", "rms_dynamic_range_db", "rms_crest"]
    names += ["onset_mean", "onset_std", "onset_max", "onset_rate"]
    names += ["tempo"]
    names += ["f0_median_log2", "f0_p10_log2", "f0_p90_log2", "f0_std_semitones", "f0_stability"]
    names += ["harmonic_ratio", "percussive_ratio"]
    names += ["peak_position", "attack_time", "decay_slope_db_s", "temporal_centroid"]
    return names


def feature_groups(cfg: FeatureConfig = DEFAULT_CONFIG) -> dict[str, list[str]]:
    """Human-readable grouping used by documentation and the Models page."""
    groups: dict[str, list[str]] = {}
    for name in feature_names(cfg):
        if name.startswith("mfcc"):
            key = "MFCC"
        elif name.startswith("dmfcc"):
            key = "Delta MFCC"
        elif name.startswith("mel"):
            key = "Mel spectrogram"
        elif name.startswith("chroma"):
            key = "Chroma"
        elif name.startswith("contrast"):
            key = "Spectral contrast"
        elif name.startswith(("centroid", "bandwidth", "rolloff", "flatness")):
            key = "Spectral shape"
        elif name.startswith("zcr"):
            key = "Zero-crossing rate"
        elif name.startswith("rms"):
            key = "RMS energy"
        elif name.startswith("onset"):
            key = "Onset strength"
        elif name == "tempo":
            key = "Tempo"
        elif name.startswith("f0"):
            key = "Pitch (YIN)"
        elif name.endswith("_ratio"):
            key = "Harmonic / percussive"
        else:
            key = "Temporal envelope"
        groups.setdefault(key, []).append(name)
    return groups


def _mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] == 0:
        z = np.zeros(x.shape[0])
        return z, z
    return x.mean(axis=1), x.std(axis=1)


def extract_features(y: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Compute the feature vector for one pre-processed segment sampled at ``cfg.sr``."""
    import librosa  # imported lazily: librosa takes several seconds to import

    sr = cfg.sr
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if y.size < cfg.n_fft:
        y = np.pad(y, (0, cfg.n_fft - y.size))
    # A whisper of dither keeps log / pitch computations stable on digital silence.
    rng = np.random.default_rng(12345)
    y = y + rng.normal(0.0, 1e-6, y.size).astype(np.float32)

    hop = cfg.hop_length
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = librosa.stft(y, n_fft=cfg.n_fft, hop_length=hop, window="hann", center=True)
        S = np.abs(D)
        P = S ** 2
        mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=sr / 2)
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg.n_mfcc)
        frames = mfcc.shape[1]
        if frames >= 3:
            width = min(9, frames if frames % 2 == 1 else frames - 1)
            d_mfcc = librosa.feature.delta(mfcc, width=max(3, width), mode="interp")
        else:
            d_mfcc = np.zeros_like(mfcc)
        chroma = librosa.feature.chroma_stft(S=P, sr=sr, n_fft=cfg.n_fft)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=150.0)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=hop, center=True)[0]
        rms = librosa.feature.rms(S=S, frame_length=cfg.n_fft)[0]
        onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
        try:
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
        except Exception:
            onsets = np.array([])
        try:
            tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop))[0])
        except Exception:
            tempo = 0.0
        f0 = librosa.yin(y, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=sr, frame_length=cfg.n_fft, hop_length=hop)
        # HPSS on the 64-band mel power spectrogram: ~25x faster than on the full
        # STFT and enough to measure how tonal vs. impulsive the segment is.
        H, Pm = librosa.decompose.hpss(mel, kernel_size=(17, 9))

    duration = y.size / float(sr)
    feats: list[np.ndarray | list[float]] = []

    m_mean, m_std = _mean_std(mfcc)
    feats += [m_mean, m_std, _mean_std(d_mfcc)[1]]

    # Summarise the 64 mel bands into 32 by averaging neighbouring bands.
    groups = np.array_split(np.arange(log_mel.shape[0]), cfg.mel_summary_bands)
    mel_summary = np.stack([log_mel[g].mean(axis=0) for g in groups])
    mel_mean, mel_std = _mean_std(mel_summary)
    feats += [mel_mean, mel_std]

    c_mean, c_std = _mean_std(chroma)
    feats += [c_mean, c_std]
    k_mean, k_std = _mean_std(contrast)
    feats += [k_mean, k_std]

    for arr in (centroid, bandwidth, rolloff, flatness, zcr):
        feats.append([float(np.mean(arr)), float(np.std(arr))])

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    rms_max = float(np.max(rms))
    rms_mean = float(np.mean(rms))
    feats.append([
        rms_mean,
        float(np.std(rms)),
        rms_max,
        float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
        rms_max / (rms_mean + 1e-9),
    ])

    feats.append([
        float(np.mean(onset_env)),
        float(np.std(onset_env)),
        float(np.max(onset_env)) if onset_env.size else 0.0,
        len(onsets) / duration if duration > 0 else 0.0,
    ])
    feats.append([tempo])

    # Pitch statistics on frames within 30 dB of the loudest frame.
    n = min(len(f0), len(rms_db))
    loud = rms_db[:n] >= (rms_db[:n].max() - 30.0)
    f0_loud = f0[:n][loud]
    if f0_loud.size >= 2:
        semis = 12.0 * np.log2(f0_loud)
        steps = np.abs(np.diff(semis))
        feats.append([
            float(np.log2(np.median(f0_loud))),
            float(np.log2(np.percentile(f0_loud, 10))),
            float(np.log2(np.percentile(f0_loud, 90))),
            float(np.std(semis)),
            float(np.mean(steps < 1.0)),
        ])
    else:
        feats.append([0.0, 0.0, 0.0, 0.0, 0.0])

    total = float(np.sum(mel)) + 1e-12
    feats.append([float(np.sum(H)) / total, float(np.sum(Pm)) / total])

    peak_idx = int(np.argmax(rms))
    frame_t = hop / float(sr)
    above = np.flatnonzero(rms >= 0.1 * rms_max)
    attack = (peak_idx - int(above[0])) * frame_t if above.size else 0.0
    tail = rms_db[peak_idx: peak_idx + max(3, int(0.5 / frame_t))]
    if tail.size >= 3:
        t = np.arange(tail.size) * frame_t
        slope = float(np.polyfit(t, tail, 1)[0])
    else:
        slope = 0.0
    times = np.arange(len(rms)) * frame_t
    centroid_t = float(np.sum(times * rms) / (np.sum(rms) + 1e-12)) / max(duration, 1e-6)
    feats.append([peak_idx / max(1, len(rms) - 1), attack, slope, centroid_t])

    vec = np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in feats])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_many(segments: list[np.ndarray], cfg: FeatureConfig = DEFAULT_CONFIG, n_jobs: int = 1) -> np.ndarray:
    """Feature matrix for many segments (optionally in parallel)."""
    if not segments:
        return np.zeros((0, len(feature_names(cfg))), dtype=np.float32)
    if n_jobs == 1 or len(segments) < 8:
        return np.vstack([extract_features(s, cfg) for s in segments])
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(extract_features)(s, cfg) for s in segments)
    return np.vstack(rows)
