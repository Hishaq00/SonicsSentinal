"""Feature-extraction, dataset-integrity and Python-model tests."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from config import DATASET_DIR, METADATA_CSV, MODELS_DIR, class_names
from feature_extraction.features import DEFAULT_CONFIG, extract_features, feature_groups, feature_names

from .conftest import dataset_file, impulse_in_noise, needs_dataset, needs_model, noise, tone

SR = 22050


# ---------------------------------------------------------------- features --
def test_feature_vector_shape_and_names():
    v = extract_features(tone(2.0))
    names = feature_names()
    assert v.shape == (len(names),) == (193,)
    assert len(set(names)) == len(names)
    assert sum(len(g) for g in feature_groups().values()) == len(names)


def test_features_are_finite_even_for_silence_and_short_input():
    for y in (np.zeros(2 * SR, dtype=np.float32), tone(0.05), noise(2.0)):
        assert np.all(np.isfinite(extract_features(y)))


def test_features_are_deterministic():
    y = noise(2.0, seed=5)
    assert np.array_equal(extract_features(y), extract_features(y))


def test_features_distinguish_tone_from_noise_and_impulse():
    names = feature_names()
    idx = {n: i for i, n in enumerate(names)}
    t, n, imp = extract_features(tone(2.0, 800)), extract_features(noise(2.0)), extract_features(impulse_in_noise(2.0, at=0.5))
    assert t[idx["flatness_mean"]] < n[idx["flatness_mean"]]  # tone is not noise-like
    assert t[idx["harmonic_ratio"]] > imp[idx["harmonic_ratio"]]  # impulse is percussive
    assert imp[idx["rms_crest"]] > t[idx["rms_crest"]]  # impulse has a high crest factor
    assert abs(2 ** t[idx["f0_median_log2"]] - 800) < 40  # YIN finds the pitch


# ----------------------------------------------------------------- dataset --
@needs_dataset
def test_dataset_is_balanced_and_split_70_15_15():
    """Split ratios hold per class, and augmentation levels the training rows out.

    The categories built from real recordings are small on purpose (ESC-50 ships 40 clips
    per category), so the guarantee is on the *training rows* after augmentation, which is
    what the models actually see.
    """
    with open(METADATA_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    originals = [r for r in rows if r["status"] == "original"]
    train_rows = {}
    for cls in class_names():
        c = [r for r in originals if r["class_label"] == cls]
        assert len(c) >= 40, f"{cls}: only {len(c)} recordings"
        splits = {s: sum(r["split"] == s for r in c) for s in ("train", "validation", "test")}
        assert splits["test"] >= 6, cls
        assert abs(splits["train"] / len(c) - 0.70) < 0.05, cls
        train_rows[cls] = sum(1 for r in rows if r["class_label"] == cls and r["split"] == "train")
    spread = max(train_rows.values()) / max(1, min(train_rows.values()))
    assert spread <= 5.5, f"training rows per class span {spread:.1f}x: {train_rows}"


@needs_dataset
def test_no_source_leaks_between_splits():
    with open(METADATA_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_root: dict[str, set] = {}
    by_group: dict[str, set] = {}
    for r in rows:
        by_root.setdefault(r["root_audio_id"] or r["audio_id"], set()).add(r["split"])
        if r["source_group"] and (r["source_group"].startswith("tts") or r.get("source") != "urbansound8k"):
            by_group.setdefault(r["source_group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_root.values()), "augmented copies must stay in their parent's split"
    assert all(len(s) == 1 for s in by_group.values()), "a TTS source group must not cross splits"
    assert not any(r["status"] == "augmented" and r["split"] != "train" for r in rows)


@needs_dataset
def test_audio_ids_unique_and_files_exist():
    with open(METADATA_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["audio_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows[:: max(1, len(rows) // 50)]:
        assert (DATASET_DIR / r["relative_path"]).exists()


# ------------------------------------------------------------ python model --
@needs_model
def test_active_model_card_is_complete():
    pointer = json.loads((MODELS_DIR / "active.json").read_text())
    card = json.loads((MODELS_DIR / (pointer["version"] + ".json")).read_text())
    assert card["classes"] == class_names()
    assert len(card["comparison"]) >= 3  # at least three models compared
    t = card["metrics"]["test"]["recording"]
    assert {"accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix", "per_class"} <= set(t)


@needs_model
def test_model_outputs_probabilities_for_all_classes():
    from python_models.inference import get_classifier

    clf = get_classifier()
    scores = clf.predict_segments([tone(2.0), noise(2.0)])
    for s in scores:
        assert list(s) == class_names()
        assert abs(sum(s.values()) - 1.0) < 1e-6
        assert all(0.0 <= v <= 1.0 for v in s.values())


@needs_model
@needs_dataset
@pytest.mark.parametrize("cls", ["Gunshot", "Alarm or Siren", "Glass Breaking", "Background Noise"])
def test_model_recognises_unseen_test_clips(cls):
    """At least 2 of the first 3 unseen test clips of the class are predicted correctly."""
    from audio_preprocessing import PreprocessConfig, decode_audio, preprocess_audio, segment_audio, select_event_segments
    from python_models.inference import get_classifier

    clf = get_classifier()
    hits = 0
    for n in range(3):
        path = dataset_file(cls, n)
        dec = decode_audio(path)
        pre = preprocess_audio(dec.samples, dec.sample_rate, PreprocessConfig())
        segs = select_event_segments(segment_audio(pre.y, pre.sr, 2.0, 1.0))
        scores = clf.predict_segments([s.audio for s in segs])
        mean = {c: np.mean([s[c] for s in scores]) for c in scores[0]}
        hits += max(mean, key=mean.get) == cls
    assert hits >= 2
