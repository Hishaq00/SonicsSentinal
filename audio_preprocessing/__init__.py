"""Audio input, validation, pre-processing, quality analysis and fingerprinting."""
from .loader import (
    SUPPORTED_EXTENSIONS,
    AudioDecodeError,
    DecodedAudio,
    decode_audio,
    decode_bytes,
    load_mono,
    resample,
    sha256_file,
    write_wav,
)
from .preprocess import (
    PreprocessConfig,
    PreprocessResult,
    Segment,
    fix_length,
    preprocess_audio,
    segment_audio,
    select_event_segments,
)
from .quality import analyze_quality
from .validation import ValidationResult, validate_audio_file, validate_live_window

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "AudioDecodeError",
    "DecodedAudio",
    "decode_audio",
    "decode_bytes",
    "load_mono",
    "resample",
    "sha256_file",
    "write_wav",
    "PreprocessConfig",
    "PreprocessResult",
    "Segment",
    "fix_length",
    "preprocess_audio",
    "segment_audio",
    "select_event_segments",
    "analyze_quality",
    "ValidationResult",
    "validate_audio_file",
    "validate_live_window",
]
