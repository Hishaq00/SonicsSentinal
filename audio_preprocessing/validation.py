"""Audio file validation (format, size, duration, sampling rate, channels,
integrity and presence of an audio signal).

Every check is recorded so the UI can show exactly why a file was accepted or
rejected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .loader import AudioDecodeError, DecodedAudio, decode_audio, sniff_file

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class ValidationResult:
    checks: list[dict] = field(default_factory=list)
    decoded: DecodedAudio | None = None

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def ok(self) -> bool:
        return not any(c["status"] == FAIL for c in self.checks)

    @property
    def errors(self) -> list[str]:
        return [c["detail"] for c in self.checks if c["status"] == FAIL]

    @property
    def warnings(self) -> list[str]:
        return [c["detail"] for c in self.checks if c["status"] == WARN]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": self.checks, "errors": self.errors, "warnings": self.warnings}


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-10)))


def loudest_frame_dbfs(y: np.ndarray, sr: int, frame_seconds: float = 0.1) -> float:
    """RMS level (dBFS) of the loudest 100 ms frame — robust signal-presence test."""
    if y.size == 0:
        return -200.0
    frame = max(1, int(sr * frame_seconds))
    n = len(y) // frame
    if n == 0:
        return _db(np.sqrt(np.mean(y ** 2)))
    frames = y[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return _db(rms.max())


def validate_audio_file(
    path: str | os.PathLike,
    original_filename: str,
    rules: dict,
    silence_dbfs: float = -55.0,
) -> ValidationResult:
    """Validate an uploaded file. ``rules`` is the ``validation`` settings section."""
    result = ValidationResult()
    allowed = [f.lower() for f in rules.get("allowed_formats", ["wav", "mp3", "flac", "ogg", "m4a"])]
    ext = Path(original_filename).suffix.lower().lstrip(".")

    # 1. Format (extension + actual content)
    if ext not in allowed:
        result.add("Format", FAIL, f"Unsupported format '.{ext or '?'}'. Allowed: {', '.join(allowed).upper()}.")
        return result
    sniffed = sniff_file(path)
    if sniffed is None:
        result.add("Format", FAIL, "The file content is not a recognised audio format.")
        return result
    if sniffed not in allowed:
        result.add("Format", FAIL, f"The file contains {sniffed.upper()} data, which is not an allowed format.")
        return result
    if sniffed != ext and not (ext in ("m4a",) and sniffed == "m4a"):
        result.add("Format", WARN, f"Extension .{ext} does not match the content ({sniffed.upper()}); decoded as {sniffed.upper()}.")
    else:
        result.add("Format", PASS, f"{ext.upper()} file.")

    # 2. Size
    size = os.path.getsize(path)
    max_bytes = float(rules.get("max_file_mb", 25)) * 1024 * 1024
    if size == 0:
        result.add("File size", FAIL, "The file is empty.")
        return result
    if size > max_bytes:
        result.add("File size", FAIL, f"File is {size / 1048576:.1f} MB; the limit is {rules.get('max_file_mb', 25)} MB.")
        return result
    result.add("File size", PASS, f"{size / 1024:.0f} KB.")

    # 3. Integrity (decoding)
    try:
        decoded = decode_audio(path)
    except AudioDecodeError as exc:
        result.add("File integrity", FAIL, f"Audio decoding failed: {exc}")
        return result
    samples = decoded.samples
    if samples.size == 0:
        result.add("File integrity", FAIL, "The file decoded to zero audio samples (damaged or empty stream).")
        return result
    if not np.all(np.isfinite(samples)):
        bad = int(np.sum(~np.isfinite(samples)))
        result.add("File integrity", WARN, f"{bad} invalid sample values were replaced with silence.")
        decoded.samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    elif decoded.warnings:
        result.add("File integrity", WARN, "; ".join(decoded.warnings))
    else:
        result.add("File integrity", PASS, f"Decoded with {decoded.decoder} ({decoded.codec}).")
    result.decoded = decoded

    # 4. Duration
    duration = decoded.duration
    min_d = float(rules.get("min_duration_seconds", 0.5))
    max_d = float(rules.get("max_duration_seconds", 600))
    if duration < min_d:
        result.add("Duration", FAIL, f"Recording is too short ({duration:.2f} s). Minimum is {min_d:g} s.")
    elif duration > max_d:
        result.add("Duration", FAIL, f"Recording is too long ({duration:.0f} s). Maximum is {max_d:g} s.")
    else:
        result.add("Duration", PASS, f"{duration:.2f} s.")

    # 5. Sampling rate
    sr = decoded.sample_rate
    if sr < int(rules.get("min_sample_rate", 8000)):
        result.add("Sampling rate", FAIL, f"Sampling rate {sr} Hz is below the minimum of {rules.get('min_sample_rate', 8000)} Hz.")
    elif sr > int(rules.get("max_sample_rate", 192000)):
        result.add("Sampling rate", FAIL, f"Sampling rate {sr} Hz is above the maximum of {rules.get('max_sample_rate', 192000)} Hz.")
    else:
        result.add("Sampling rate", PASS, f"{sr} Hz.")

    # 6. Channels
    ch = decoded.channels
    if ch < 1 or ch > int(rules.get("max_channels", 8)):
        result.add("Channels", FAIL, f"{ch} channels is not supported (maximum {rules.get('max_channels', 8)}).")
    else:
        result.add("Channels", PASS, "Mono." if ch == 1 else f"{ch} channels (will be mixed to mono).")

    # 7. Presence of an audio signal
    loudest = loudest_frame_dbfs(decoded.mono(), sr)
    if loudest < silence_dbfs:
        result.add("Audio signal", FAIL, f"The recording is silent or near-silent (loudest part {loudest:.0f} dBFS).")
    else:
        result.add("Audio signal", PASS, f"Signal present (loudest part {loudest:.0f} dBFS).")
    return result


def validate_live_window(decoded: DecodedAudio, expected_seconds: float, silence_dbfs: float = -80.0) -> ValidationResult:
    """Lighter validation for microphone windows (format is always WAV from the browser)."""
    result = ValidationResult(decoded=decoded)
    if decoded.samples.size == 0:
        result.add("Window", FAIL, "Empty microphone window.")
        return result
    if decoded.duration < expected_seconds * 0.5:
        result.add("Duration", FAIL, f"Window too short ({decoded.duration:.2f} s).")
    else:
        result.add("Duration", PASS, f"{decoded.duration:.2f} s.")
    if decoded.sample_rate < 8000:
        result.add("Sampling rate", FAIL, f"{decoded.sample_rate} Hz is too low.")
    else:
        result.add("Sampling rate", PASS, f"{decoded.sample_rate} Hz.")
    loudest = loudest_frame_dbfs(decoded.mono(), decoded.sample_rate)
    if loudest < silence_dbfs:
        result.add("Audio signal", WARN, f"Silent window ({loudest:.0f} dBFS).")
    else:
        result.add("Audio signal", PASS, f"Signal present ({loudest:.0f} dBFS).")
    return result
