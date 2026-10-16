"""Dataset tooling and confidence calibration: generators, augmentation, label corrections, temperature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from augmentation.transforms import time_shift
from config import class_names, load_classes
from python_models.evaluation import calibration_summary, fit_temperature, temper
from scripts.relabel_dataset import plan
from scripts.synthetic.optional_sources import OPTIONAL_GENERATORS
from scripts.synthetic.sources import VARIANT_GENERATORS

SR = 22050


def test_optional_categories_are_classes_with_generators():
    """Every procedural generator writes into a class the models are trained on."""
    trainable = set(class_names())
    assert set(OPTIONAL_GENERATORS) <= trainable
    assert {cls for cls, _ in VARIANT_GENERATORS.values()} <= trainable
    # A parent category (Animal Sound) is listed but never predicted, so it is not trainable.
    listed = load_classes()["classes"]
    assert len(trainable) == sum(1 for c in listed if not c.get("abstract"))
    assert {c["name"] for c in listed if c.get("abstract")}.isdisjoint(trainable)


@pytest.mark.parametrize("name", sorted(OPTIONAL_GENERATORS))
def test_optional_generators_give_clean_audio(name):
    rng = np.random.default_rng(1)
    y, info = OPTIONAL_GENERATORS[name](rng, 3.0, SR)
    assert np.isfinite(y).all() and 0.5 < np.max(np.abs(y)) <= 1.0
    assert len(y) <= int(3.0 * SR) + 1 and info["variant"]


@pytest.mark.parametrize("variant", ["empty-room", "everyday-clicks", "large-animals"])
def test_variant_generators_give_clean_audio(variant):
    cls, gen = VARIANT_GENERATORS[variant]
    assert cls in class_names()
    y, info = gen(np.random.default_rng(2), 3.0, SR)
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.5 and info["variant"].startswith(variant.split("-")[0])


def test_time_shift_keeps_short_clips_inside():
    rng = np.random.default_rng(0)
    y = np.ones(int(0.2 * SR), dtype=np.float32)  # shorter than the 0.5 s maximum shift
    for _ in range(50):
        out, _ = time_shift(y, SR, rng)
        assert out.shape == y.shape and out.sum() > 0


def test_relabel_plan_selects_the_documented_recordings():
    df = pd.DataFrame([
        {"audio_id": "DS-MCH-0001", "status": "original", "source": "urbansound8k", "class_label": "Machinery Fault",
         "notes": "UrbanSound8K fold1 air_conditioner", "variant": ""},
        {"audio_id": "DS-MCH-0002", "status": "original", "source": "urbansound8k", "class_label": "Machinery Fault",
         "notes": "UrbanSound8K fold2 drilling", "variant": ""},
        {"audio_id": "DS-MCH-0003", "status": "original", "source": "synthetic-procedural",
         "class_label": "Machinery Fault", "notes": "", "variant": "bearing-defect"},
        {"audio_id": "DS-BGN-0001", "status": "original", "source": "synthetic-procedural",
         "class_label": "Background Noise", "notes": "", "variant": "rain+door-close"},
        {"audio_id": "DS-MCH-0001-AUG1", "status": "augmented", "source": "urbansound8k",
         "class_label": "Machinery Fault", "notes": "UrbanSound8K fold1 air_conditioner", "variant": ""},
    ])
    moves = {df.at[i, "audio_id"]: (new_id, c["to_class"]) for i, new_id, c in plan(df)}
    assert moves == {"DS-MCH-0001": ("DS-BGN-0002", "Background Noise"),
                     "DS-MCH-0002": ("DS-DRL-0001", "Drilling or Grinder Sound"),
                     "DS-BGN-0001": ("DS-DOR-0001", "Door Impact")}


def test_temperature_scaling_keeps_predictions_and_calibrates():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3000, 6)) * 2.5
    p_true = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(6, p=p) for p in p_true])
    over = np.exp(3 * logits) / np.exp(3 * logits).sum(axis=1, keepdims=True)  # three times too confident
    t = fit_temperature(over, y)
    assert 2.6 < t < 3.4
    cal = temper(over, t)
    assert (cal.argmax(axis=1) == over.argmax(axis=1)).all()
    assert np.allclose(cal.sum(axis=1), 1.0)
    assert calibration_summary(cal, y)["ece"] < calibration_summary(over, y)["ece"] / 2
    assert np.allclose(temper(over, 1.0), over) and np.allclose(temper(over, None), over)


def test_live_rendering_matches_the_live_service_path():
    """A window from live_render must be byte-identical to what live_service computes for the same audio."""
    from audio_preprocessing import PreprocessConfig, preprocess_audio
    from config import load_default_settings
    from feature_extraction.live_render import LIVE_SAMPLE_RATE, live_windows, seed_from, simulate_microphone
    from audio_preprocessing.loader import resample

    settings = load_default_settings()
    cfg = PreprocessConfig.from_settings(settings)
    rng = np.random.default_rng(3)
    y = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(5 * SR)) / SR) + 0.02 * rng.standard_normal(int(5 * SR))).astype("float32")

    windows, mic = live_windows(y, SR, seed_from("DS-TST-0001"), cfg, 2.0, 1.0)
    assert windows and mic

    # live_service.process_window: preprocess_audio(window, trim=False) on the window it receives.
    mic_audio, _ = simulate_microphone(y, SR, seed_from("DS-TST-0001"))
    y22 = resample(mic_audio, SR, LIVE_SAMPLE_RATE) if SR != LIVE_SAMPLE_RATE else mic_audio
    for seg, rendered in windows:
        expected = preprocess_audio(seg.audio, LIVE_SAMPLE_RATE, cfg, trim=False).y
        assert np.array_equal(rendered, expected)
        assert len(rendered) == len(seg.audio)  # never trimmed
        assert np.array_equal(seg.audio, y22[round(seg.start * LIVE_SAMPLE_RATE):][: len(seg.audio)])


def test_microphone_simulation_is_deterministic_and_bounded():
    from feature_extraction.live_render import MICROPHONES, seed_from, simulate_microphone

    rng = np.random.default_rng(4)
    y = (0.5 * rng.standard_normal(int(2 * SR))).astype("float32")
    a, mic_a = simulate_microphone(y, SR, seed_from("DS-TST-0002"))
    b, mic_b = simulate_microphone(y, SR, seed_from("DS-TST-0002"))
    assert mic_a == mic_b and mic_a in MICROPHONES and np.array_equal(a, b)
    assert np.isfinite(a).all() and np.max(np.abs(a)) <= 1.0
    assert not np.array_equal(a, simulate_microphone(y, SR, seed_from("DS-TST-0003"))[0])
