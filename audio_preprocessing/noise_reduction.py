"""Adaptive spectral-gating noise reduction.

Idea (explainable in one paragraph):
1. Compute the STFT and the energy of every frame.
2. If the recording has little dynamic range (a stationary sound such as a
   siren, machine hum or ambient noise) we *skip* reduction, because the
   "quiet" frames would contain the sound we want to classify.
3. Otherwise the quietest 15 % of frames are treated as the noise profile.
   Each time-frequency bin that is not clearly above that profile is
   attenuated by at most ``max_reduction_db`` (default 12 dB). A soft mask
   (smoothed over time and frequency) avoids "musical noise" artefacts.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft


def frame_energy_db(mag: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.mean(mag ** 2, axis=0) + 1e-12)


def spectral_gate(
    y: np.ndarray,
    sr: int,
    max_reduction_db: float = 12.0,
    min_dynamic_range_db: float = 12.0,
    noise_percentile: float = 15.0,
    threshold_std: float = 1.5,
    n_fft: int = 1024,
    hop: int = 256,
) -> tuple[np.ndarray, dict]:
    """Return (denoised signal, info dict)."""
    info: dict = {"applied": False}
    if y.size < n_fft * 2:
        info["reason"] = "clip too short for a noise profile"
        return y, info

    _, _, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    mag = np.abs(Z)
    energy = frame_energy_db(mag)
    # 99.5th percentile so that short impulsive events (a gunshot is ~3 % of a
    # clip) still count as "signal" above the noise floor.
    dynamic_range = float(np.percentile(energy, 99.5) - np.percentile(energy, 10))
    info["dynamic_range_db"] = round(dynamic_range, 1)
    if dynamic_range < min_dynamic_range_db:
        info["reason"] = "stationary signal (low dynamic range); reduction skipped to protect the sound"
        return y, info

    quiet = energy <= np.percentile(energy, noise_percentile)
    if quiet.sum() < 3:
        info["reason"] = "not enough quiet frames for a noise profile"
        return y, info

    noise_mean = mag[:, quiet].mean(axis=1, keepdims=True)
    noise_std = mag[:, quiet].std(axis=1, keepdims=True)
    threshold = noise_mean + threshold_std * noise_std

    mask = (mag > threshold).astype(np.float32)
    mask = uniform_filter(mask, size=(3, 5))  # smooth over 3 bins x 5 frames
    floor = 10.0 ** (-max_reduction_db / 20.0)
    gain = floor + (1.0 - floor) * mask

    _, y_out = istft(Z * gain, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    y_out = y_out[: y.size]
    if y_out.size < y.size:
        y_out = np.pad(y_out, (0, y.size - y_out.size))

    noise_floor_db = float(np.percentile(energy, noise_percentile))
    info.update(
        applied=True,
        noise_floor_db=round(noise_floor_db, 1),
        max_reduction_db=max_reduction_db,
        attenuated_fraction=round(float(1.0 - mask.mean()), 3),
    )
    return y_out.astype(np.float32), info
