"""Audio-format, validation, integrity, silence and boundary tests."""
from __future__ import annotations

import numpy as np
import pytest

from audio_preprocessing.loader import (
    AudioDecodeError,
    convert_with_ffmpeg,
    decode_audio,
    find_ffmpeg,
    sha256_file,
    sniff_container,
)
from audio_preprocessing.validation import validate_audio_file
from config import load_default_settings

from .conftest import noise, tone, write

RULES = load_default_settings()["validation"]
needs_ffmpeg = pytest.mark.skipif(find_ffmpeg() is None, reason="FFmpeg not available")


def statuses(result):
    return {c["name"]: c["status"] for c in result.checks}


# ------------------------------------------------------------------ formats --
@pytest.mark.parametrize("subtype,bits", [("PCM_16", 16), ("PCM_24", 24), ("FLOAT", 32)])
def test_wav_subtypes_decode_with_bit_depth(wav_factory, subtype, bits):
    dec = decode_audio(wav_factory(f"t_{subtype}.wav", tone(), subtype=subtype))
    assert dec.format == "WAV" and dec.bit_depth == bits
    assert dec.sample_rate == 22050 and dec.channels == 1
    assert abs(dec.duration - 2.0) < 0.01


def test_flac_and_ogg_decode(tmp_path):
    import soundfile as sf

    y = tone(1.5)
    sf.write(tmp_path / "a.flac", y, 22050, format="FLAC")
    sf.write(tmp_path / "a.ogg", y, 22050, format="OGG", subtype="VORBIS")
    for name, fmt in (("a.flac", "FLAC"), ("a.ogg", "OGG")):
        dec = decode_audio(tmp_path / name)
        assert dec.format == fmt
        assert abs(dec.duration - 1.5) < 0.05


@needs_ffmpeg
@pytest.mark.parametrize("ext,args", [("mp3", ["-codec:a", "libmp3lame", "-b:a", "128k"]), ("m4a", ["-codec:a", "aac", "-b:a", "128k"])])
def test_compressed_formats_decode(wav_factory, tmp_path, ext, args):
    src = wav_factory("src.wav", tone(2.0))
    dst = tmp_path / f"out.{ext}"
    convert_with_ffmpeg(src, dst, args)
    dec = decode_audio(dst)
    assert dec.format == ext.upper() or (ext == "mp3" and dec.format == "MPEG")
    assert abs(dec.duration - 2.0) < 0.1
    assert dec.bit_depth is None  # compressed formats have no bit depth
    assert sniff_container(dst.read_bytes()[:64]) == ext


def test_stereo_file_reports_channels(tmp_path):
    y = np.stack([tone(), tone(freq=660)], axis=1)
    dec = decode_audio(write(tmp_path / "stereo.wav", y))
    assert dec.channels == 2
    assert dec.mono().ndim == 1


# ------------------------------------------------------------- validation --
def test_valid_file_passes_all_checks(wav_factory):
    res = validate_audio_file(wav_factory(), "clip.wav", RULES)
    assert res.ok, res.errors
    assert set(statuses(res)) >= {"Format", "File size", "File integrity", "Duration", "Sampling rate", "Channels", "Audio signal"}


def test_unsupported_extension_rejected(wav_factory):
    res = validate_audio_file(wav_factory(), "clip.aiff", RULES)
    assert not res.ok and statuses(res)["Format"] == "fail"


def test_fake_audio_content_rejected(tmp_path):
    p = tmp_path / "fake.wav"
    p.write_bytes(b"this is not audio at all" * 20)
    res = validate_audio_file(p, "fake.wav", RULES)
    assert not res.ok
    assert "not a recognised audio format" in res.errors[0]


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    assert not validate_audio_file(p, "empty.wav", RULES).ok


def test_truncated_wav_is_flagged_or_rejected(wav_factory, tmp_path):
    good = wav_factory("good.wav", tone(3.0))
    data = good.read_bytes()
    bad = tmp_path / "trunc.wav"
    bad.write_bytes(data[: len(data) // 2])  # header claims 3 s, only 1.5 s present
    res = validate_audio_file(bad, "trunc.wav", RULES)
    st = statuses(res)
    assert st.get("File integrity") in ("warn", "fail") or not res.ok


def test_silent_file_rejected(wav_factory):
    res = validate_audio_file(wav_factory("silent.wav", np.zeros(44100, dtype=np.float32)), "silent.wav", RULES)
    assert not res.ok and statuses(res)["Audio signal"] == "fail"


def test_near_silent_file_rejected(wav_factory):
    res = validate_audio_file(wav_factory("quiet.wav", noise(2.0, amp=1e-4)), "quiet.wav", RULES)
    assert not res.ok


def test_too_short_rejected_and_boundary_accepted(wav_factory):
    short = validate_audio_file(wav_factory("short.wav", tone(0.3)), "short.wav", RULES)
    assert not short.ok and statuses(short)["Duration"] == "fail"
    boundary = validate_audio_file(wav_factory("edge.wav", tone(RULES["min_duration_seconds"] + 0.01)), "edge.wav", RULES)
    assert statuses(boundary)["Duration"] == "pass"


def test_too_long_rejected(wav_factory):
    rules = dict(RULES, max_duration_seconds=5)
    res = validate_audio_file(wav_factory("long.wav", tone(6.0)), "long.wav", rules)
    assert statuses(res)["Duration"] == "fail"


def test_size_limit(wav_factory):
    rules = dict(RULES, max_file_mb=0.01)  # 10 KB
    res = validate_audio_file(wav_factory("big.wav", tone(2.0)), "big.wav", rules)
    assert not res.ok and statuses(res)["File size"] == "fail"


def test_low_sample_rate_rejected(tmp_path):
    p = write(tmp_path / "lo.wav", tone(2.0, sr=4000), sr=4000)
    res = validate_audio_file(p, "lo.wav", RULES)
    assert statuses(res)["Sampling rate"] == "fail"


def test_too_many_channels_rejected(tmp_path):
    y = np.stack([tone()] * 10, axis=1)
    res = validate_audio_file(write(tmp_path / "multi.wav", y), "multi.wav", RULES)
    assert statuses(res)["Channels"] == "fail"


def test_sha256_changes_with_content(wav_factory):
    a = sha256_file(wav_factory("a.wav", tone(freq=440)))
    b = sha256_file(wav_factory("b.wav", tone(freq=441)))
    assert a != b and len(a) == 64


def test_decode_missing_file_raises(tmp_path):
    with pytest.raises(AudioDecodeError):
        decode_audio(tmp_path / "missing.wav")
