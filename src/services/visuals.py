"""Waveform and spectrogram generation.

* Waveform: min/max peaks per pixel column (JSON), drawn by the browser on a
  canvas so it follows the colour theme and shows the play-head.
* Spectrogram: a grey-scale log-mel image (loud = dark, like a printed
  sonagram). The browser inverts it in dark mode and draws the axes.
* Report figures: matplotlib versions with axes for the PDF report.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audio_preprocessing.loader import resample

SPEC_SR = 22050
SPEC_MELS = 128
SPEC_NFFT = 1024
SPEC_HOP = 256
MAX_WIDTH = 1600
TICKS_HZ = [100, 250, 500, 1000, 2000, 4000, 8000]


def waveform_peaks(y: np.ndarray, sr: int, columns: int = 1200) -> dict:
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    columns = max(1, min(columns, n))
    edges = np.linspace(0, n, columns + 1).astype(int)
    peaks = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = y[a:max(b, a + 1)]
        peaks.append([round(float(chunk.min()), 4), round(float(chunk.max()), 4)])
    peak = float(np.max(np.abs(y))) if n else 0.0
    return {"duration": n / float(sr) if sr else 0.0, "peak": peak, "peaks": peaks}


def mel_image(y: np.ndarray, sr: int) -> tuple[Image.Image, list[dict]]:
    import librosa

    if sr != SPEC_SR:
        y = resample(y, sr, SPEC_SR)
    y = np.asarray(y, dtype=np.float32)
    if y.size < SPEC_NFFT:
        y = np.pad(y, (0, SPEC_NFFT - y.size))
    mel = librosa.feature.melspectrogram(y=y, sr=SPEC_SR, n_fft=SPEC_NFFT, hop_length=SPEC_HOP,
                                         n_mels=SPEC_MELS, fmin=20, fmax=SPEC_SR / 2)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    v = (db + 80.0) / 80.0  # 0 quiet .. 1 loud
    pix = (255.0 * (1.0 - np.clip(v, 0, 1)) ** 1.0).astype(np.uint8)[::-1]  # low frequencies at the bottom
    img = Image.fromarray(np.ascontiguousarray(pix))  # 2-D uint8 -> 8-bit grey-scale
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, img.height), Image.Resampling.BOX)
    mel_max = librosa.hz_to_mel(SPEC_SR / 2)
    mel_min = librosa.hz_to_mel(20)
    ticks = [{"hz": hz, "pos": float((librosa.hz_to_mel(hz) - mel_min) / (mel_max - mel_min))}
             for hz in TICKS_HZ if hz < SPEC_SR / 2]
    return img, ticks


def write_visuals(y: np.ndarray, sr: int, folder: Path) -> tuple[Path, Path]:
    """Create waveform.json and spectrogram.png for a recording."""
    folder.mkdir(parents=True, exist_ok=True)
    data = waveform_peaks(y, sr)
    img, ticks = mel_image(y, sr)
    data["spectrogram_ticks"] = ticks
    data["spectrogram_fmax"] = SPEC_SR / 2
    wave_path = folder / "waveform.json"
    wave_path.write_text(json.dumps(data), encoding="utf-8")
    spec_path = folder / "spectrogram.png"
    img.save(spec_path, optimize=True)
    return wave_path, spec_path


def report_figure(y: np.ndarray, sr: int, events: list[dict] | None = None) -> tuple[bytes, bytes]:
    """PNG bytes of (waveform, spectrogram) with axes for the PDF report."""
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid, paper = "#1d1c1a", "#d9d5cc", "#ffffff"
    t = np.arange(len(y)) / float(sr)
    fig, ax = plt.subplots(figsize=(7.2, 1.7), dpi=150)
    fig.patch.set_facecolor(paper)
    ax.plot(t, y, color=ink, linewidth=0.4)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.tick_params(labelsize=6, colors="#57534b")
    for s in ax.spines.values():
        s.set_color(grid)
    for ev in events or []:
        if ev.get("category") not in ("Background Noise",):
            ax.axvspan(ev["start"], ev["end"], color="#e4572e", alpha=0.12, linewidth=0)
    fig.tight_layout()
    wave_buf = io.BytesIO()
    fig.savefig(wave_buf, format="png", facecolor=paper)
    plt.close(fig)

    ys = resample(y, sr, SPEC_SR) if sr != SPEC_SR else y
    mel = librosa.feature.melspectrogram(y=np.asarray(ys, dtype=np.float32), sr=SPEC_SR, n_fft=SPEC_NFFT,
                                         hop_length=SPEC_HOP, n_mels=SPEC_MELS)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=150)
    fig.patch.set_facecolor(paper)
    import librosa.display

    librosa.display.specshow(db, sr=SPEC_SR, hop_length=SPEC_HOP, x_axis="time", y_axis="mel", ax=ax, cmap="gray_r")
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Frequency (Hz, mel)", fontsize=7)
    ax.tick_params(labelsize=6, colors="#57534b")
    fig.tight_layout()
    spec_buf = io.BytesIO()
    fig.savefig(spec_buf, format="png", facecolor=paper)
    plt.close(fig)
    return wave_buf.getvalue(), spec_buf.getvalue()
