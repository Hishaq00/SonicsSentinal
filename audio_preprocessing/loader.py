"""Audio decoding, metadata extraction and WAV writing.

Decoding order:
1. ``soundfile`` (libsndfile) handles WAV, FLAC, OGG/Vorbis/Opus and MP3.
2. FFmpeg handles everything else (M4A/AAC, unusual MP3s). We use the system
   ``ffmpeg`` when it is on PATH, otherwise the binary bundled with the
   ``imageio-ffmpeg`` package, so no manual FFmpeg install is required.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "m4a")

# Bit depth for the PCM subtypes libsndfile reports. Lossy codecs have no bit depth.
_SUBTYPE_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ALAC_16": 16,
    "ALAC_20": 20,
    "ALAC_24": 24,
    "ALAC_32": 32,
}


class AudioDecodeError(Exception):
    """Raised when a file cannot be decoded as audio."""


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, shape (frames, channels), range roughly [-1, 1]
    sample_rate: int
    channels: int
    format: str  # container, e.g. WAV, FLAC, MP3, OGG, M4A
    codec: str  # subtype / codec name, e.g. PCM_16, VORBIS, AAC
    bit_depth: int | None
    decoder: str  # "soundfile" or "ffmpeg"
    expected_frames: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    def mono(self) -> np.ndarray:
        if self.samples.ndim == 1:
            return self.samples.astype(np.float32)
        return self.samples.mean(axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Container sniffing (magic bytes)
# --------------------------------------------------------------------------- #
def sniff_container(head: bytes) -> str | None:
    """Identify the audio container from the first bytes of a file."""
    if len(head) < 12:
        return None
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:3] == b"ID3":
        return "mp3"
    # MPEG audio frame sync: 11 set bits
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    return None


def sniff_file(path: str | os.PathLike) -> str | None:
    with open(path, "rb") as fh:
        return sniff_container(fh.read(64))


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# FFmpeg helpers
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on environment
        return None


_STREAM_RE = re.compile(r"Audio:\s*([A-Za-z0-9_]+)[^,]*,\s*(\d+)\s*Hz,\s*([^,]+)")


def _probe_with_ffmpeg(ffmpeg: str, path: str) -> dict:
    """Parse codec / rate / layout from ``ffmpeg -i`` stderr (no ffprobe needed)."""
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    info: dict = {}
    match = _STREAM_RE.search(proc.stderr)
    if match:
        info["codec"] = match.group(1).upper()
        info["sample_rate"] = int(match.group(2))
        info["layout"] = match.group(3).strip()
    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if dur:
        h, m, s = dur.groups()
        info["duration"] = int(h) * 3600 + int(m) * 60 + float(s)
    return info


def _decode_with_ffmpeg(path: str, container: str | None) -> DecodedAudio:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AudioDecodeError(
            "This format needs FFmpeg to decode. Install FFmpeg or the 'imageio-ffmpeg' package."
        )
    info = _probe_with_ffmpeg(ffmpeg, path)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="ss_decode_")
    os.close(fd)
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-v", "error", "-y", "-i", path,
             "-vn", "-acodec", "pcm_f32le", "-f", "wav", tmp_path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
        )
        if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 44:
            raise AudioDecodeError(f"FFmpeg could not decode the file: {proc.stderr.strip()[:300]}")
        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        warnings = []
        if proc.stderr.strip():
            warnings.append("Decoder reported problems: " + proc.stderr.strip().splitlines()[0][:200])
        expected = None
        if info.get("duration"):
            expected = int(round(info["duration"] * sr))
        return DecodedAudio(
            samples=data,
            sample_rate=int(sr),
            channels=int(data.shape[1]),
            format=(container or Path(path).suffix.lstrip(".")).upper(),
            codec=info.get("codec", "UNKNOWN"),
            bit_depth=None,
            decoder="ffmpeg",
            expected_frames=expected,
            warnings=warnings,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _wav_declared_frames(path: str) -> int | None:
    """Frame count promised by a RIFF/WAVE header's ``data`` chunk.

    libsndfile clamps ``info.frames`` to the bytes actually present, so a
    truncated WAV looks complete; reading the header size exposes it.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            block_align = None
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    return None
                cid, size = hdr[:4], int.from_bytes(hdr[4:8], "little")
                if cid == b"fmt ":
                    fmt = fh.read(size + (size & 1))
                    if len(fmt) >= 14:
                        block_align = int.from_bytes(fmt[12:14], "little")
                elif cid == b"data":
                    if not block_align or size in (0, 0xFFFFFFFF):  # streaming writers leave 0 / -1
                        return None
                    return size // block_align
                else:
                    fh.seek(size + (size & 1), 1)
    except OSError:
        return None


def decode_audio(path: str | os.PathLike) -> DecodedAudio:
    """Decode any supported file into float32 samples (frames, channels)."""
    path = str(path)
    if not os.path.exists(path):
        raise AudioDecodeError("File not found.")
    if os.path.getsize(path) == 0:
        raise AudioDecodeError("The file is empty.")
    container = sniff_file(path)

    if container != "m4a":
        try:
            info = sf.info(path)
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            warnings: list[str] = []
            expected = int(info.frames) if info.frames and info.frames > 0 else None
            declared = _wav_declared_frames(path) if container == "wav" else None
            if declared and (expected is None or declared > expected):
                expected = declared
            if expected and data.shape[0] < expected * 0.98:
                warnings.append(
                    f"Only {data.shape[0]} of {expected} frames could be read (truncated file)."
                )
            fmt = (info.format or container or "").upper()
            return DecodedAudio(
                samples=data,
                sample_rate=int(sr),
                channels=int(data.shape[1]),
                format=fmt,
                codec=(info.subtype or "UNKNOWN").upper(),
                bit_depth=_SUBTYPE_BITS.get((info.subtype or "").upper()),
                decoder="soundfile",
                expected_frames=expected,
                warnings=warnings,
            )
        except AudioDecodeError:
            raise
        except Exception:
            # Fall through to FFmpeg (unusual MP3 variants, OGG/Opus on old libsndfile, etc.)
            pass
    try:
        return _decode_with_ffmpeg(path, container)
    except AudioDecodeError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise AudioDecodeError(f"Audio decoding failed: {exc}") from exc


def decode_bytes(data: bytes, suffix: str = ".wav") -> DecodedAudio:
    """Decode audio held in memory (used for live microphone windows)."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ss_mem_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return decode_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resampling (soxr, falling back to polyphase filtering)."""
    if sr_in == sr_out or y.size == 0:
        return y.astype(np.float32)
    try:
        import soxr  # type: ignore

        return soxr.resample(y.astype(np.float32), sr_in, sr_out, quality="HQ").astype(np.float32)
    except Exception:  # pragma: no cover - soxr is a librosa dependency
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def load_mono(path: str | os.PathLike, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode, down-mix to mono and optionally resample."""
    dec = decode_audio(path)
    y = dec.mono()
    sr = dec.sample_rate
    if target_sr and sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr
    return np.nan_to_num(y).astype(np.float32), sr


def write_wav(path: str | os.PathLike, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.nan_to_num(y), -1.0, 1.0)
    sf.write(str(path), y, int(sr), subtype=subtype)


def convert_with_ffmpeg(src: str | os.PathLike, dst: str | os.PathLike, extra_args: list[str] | None = None) -> None:
    """Generic FFmpeg conversion, e.g. WAV -> MP3 / M4A (used by tests and robustness checks)."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AudioDecodeError("FFmpeg is not available.")
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-y", "-i", str(src)] + (extra_args or []) + [str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=300)
    if proc.returncode != 0:
        raise AudioDecodeError(f"FFmpeg conversion failed: {proc.stderr.strip()[:300]}")
