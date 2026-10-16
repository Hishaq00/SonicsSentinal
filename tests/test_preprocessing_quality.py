"""Pre-processing, segmentation, noise reduction and audio-quality tests
(silence, clipping, noise, low signal, missing frames)."""
from __future__ import annotations

import numpy as np

from audio_preprocessing import PreprocessConfig, analyze_quality, preprocess_audio, segment_audio, select_event_segments
from audio_preprocessing.noise_reduction import spectral_gate
from audio_preprocessing.preprocess import fix_length, trim_silence

from .conftest import impulse_in_noise, noise, tone

SR = 22050


# ---------------------------------------------------------- pre-processing --
def test_stereo_to_mono_resample_and_normalise():
    y = np.stack([tone(2.0, sr=44100), tone(2.0, sr=44100) * 0.5], axis=1)
    out = preprocess_audio(y, 44100, PreprocessConfig())
    assert out.sr == 22050 and out.y.ndim == 1
    assert abs(np.max(np.abs(out.y)) - 0.95) < 1e-3
    steps = [s["step"] for s in out.steps]
    for name in ("Stereo-to-mono conversion", "DC offset removal", "Resampling", "Noise reduction",
                 "Silence trimming", "Amplitude normalization"):
        assert name in steps


def test_silence_trimming_records_offset():
    y = np.concatenate([np.zeros(SR), tone(1.0), np.zeros(SR)]).astype(np.float32)
    out = preprocess_audio(y, SR, PreprocessConfig(noise_reduction=False))
    assert 0.9 < out.trim_offset < 1.05
    assert out.duration < 1.2


def test_trim_not_applied_for_live_windows():
    y = np.concatenate([np.zeros(SR), tone(1.0)]).astype(np.float32)
    out = preprocess_audio(y, SR, PreprocessConfig(), trim=False)
    assert out.trim_offset == 0 and abs(out.duration - 2.0) < 0.01


def test_trim_silence_helper_keeps_signal():
    y = np.concatenate([np.zeros(5000), tone(0.5), np.zeros(5000)]).astype(np.float32)
    trimmed, start, end = trim_silence(y, 40)
    assert start > 3000 and end < len(y) - 3000 and len(trimmed) >= int(0.5 * SR) - 2048


def test_fix_length_pad_and_truncate():
    y = np.ones(100, dtype=np.float32)
    padded, was_padded = fix_length(y, 200)
    assert len(padded) == 200 and was_padded and padded[50:150].sum() == 100
    cut, was = fix_length(np.arange(300, dtype=np.float32), 200)
    assert len(cut) == 200 and not was


def test_segmentation_lengths_and_timestamps():
    y = noise(5.3)
    segs = segment_audio(y, SR, 2.0, 1.0)
    assert all(len(s.audio) == 2 * SR for s in segs)
    assert segs[0].start == 0.0 and abs(segs[-1].end - 5.3) < 0.01  # end-aligned tail segment
    assert [s.index for s in segs] == list(range(len(segs)))


def test_short_clip_becomes_one_padded_segment():
    segs = segment_audio(tone(0.7), SR, 2.0, 1.0)
    assert len(segs) == 1 and segs[0].padded and len(segs[0].audio) == 2 * SR


def test_event_segment_selection_drops_quiet_segments():
    y = impulse_in_noise(6.0, at=3.0)
    segs = segment_audio(y, SR, 2.0, 1.0)
    # burst adds ~12 dB over the -40 dBFS noise floor, so the window must be narrower than that
    kept = select_event_segments(segs, keep_within_db=8)
    assert 0 < len(kept) < len(segs)
    assert all(s.start <= 3.0 <= s.end for s in kept)


def test_noise_reduction_skips_stationary_signals():
    _, info = spectral_gate(tone(3.0), SR)
    assert info["applied"] is False and "stationary" in info["reason"]


def test_noise_reduction_attenuates_noise_floor():
    y = impulse_in_noise(4.0, at=2.0)
    out, info = spectral_gate(y, SR)
    assert info["applied"] is True
    quiet = slice(0, int(1.5 * SR))
    assert np.sqrt(np.mean(out[quiet] ** 2)) < np.sqrt(np.mean(y[quiet] ** 2))


# ------------------------------------------------------------------ quality --
def issue_codes(q):
    return {i["code"] for i in q["issues"]}


def test_quality_good_for_clean_event():
    q = analyze_quality(impulse_in_noise(3.0), SR)
    assert q["label"] in ("Good", "Acceptable")


def test_quality_silence_is_unusable():
    q = analyze_quality(np.zeros(SR * 2, dtype=np.float32), SR)
    assert q["label"] == "Unusable" and "silence" in issue_codes(q)


def test_quality_detects_severe_clipping():
    y = np.clip(tone(2.0, amp=3.0), -1.0, 1.0)
    q = analyze_quality(y, SR)
    assert "clipping" in issue_codes(q)
    assert q["metrics"]["clipping_ratio"] > 0.01


def test_quality_detects_excessive_noise():
    q = analyze_quality(noise(3.0, amp=0.3), SR)
    assert "noise" in issue_codes(q)
    assert q["label"] in ("Acceptable", "Poor", "Unusable")


def test_quality_low_signal():
    q = analyze_quality(tone(2.0, amp=0.004), SR)
    assert "low_signal" in issue_codes(q)


def test_quality_detects_dropouts():
    y = tone(3.0)
    y[SR:SR + 2000] = 0.0  # 90 ms of digital silence inside the signal
    q = analyze_quality(y, SR)
    assert q["metrics"]["dropouts"] >= 1 and "missing_frames" in issue_codes(q)


def test_quality_reports_truncated_stream():
    q = analyze_quality(tone(2.0), SR, expected_frames=4 * SR, decoded_frames=2 * SR)
    assert "missing_frames" in issue_codes(q)


def test_quality_noise_level_flag_for_background_escalation():
    q = analyze_quality(noise(3.0, amp=0.2), SR, {"noise_floor_limit_dbfs": -30})
    assert q["metrics"]["noise_level_exceeded"] is True
