"""Shared fixtures: a throw-away app + database per test, signal generators,
dataset helpers and a login helper."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from config import DATASET_DIR, METADATA_CSV, MODELS_DIR

SR = 22050


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def tone(seconds=2.0, freq=440.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds=2.0, amp=0.1, sr=SR, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


def impulse_in_noise(seconds=3.0, sr=SR, at=1.0, seed=1):
    y = noise(seconds, 0.01, sr, seed)
    n = int(0.15 * sr)
    burst = np.random.default_rng(seed).standard_normal(n) * np.exp(-np.arange(n) / (0.02 * sr)) * 0.8
    s = int(at * sr)
    y[s:s + n] += burst.astype(np.float32)
    return np.clip(y, -1, 1)


def write(path: Path, y, sr=SR, subtype="PCM_16") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
    return path


@pytest.fixture
def wav_factory(tmp_path):
    def make(name="clip.wav", y=None, sr=SR, subtype="PCM_16"):
        return write(tmp_path / name, tone() if y is None else y, sr, subtype)
    return make


# --------------------------------------------------------------------------- #
# Model / dataset availability
# --------------------------------------------------------------------------- #
def model_available() -> bool:
    return (MODELS_DIR / "active.json").exists()


def dataset_rows(split="test"):
    if not METADATA_CSV.exists():
        return []
    with open(METADATA_CSV, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split and r["status"] == "original"
                and (DATASET_DIR / r["relative_path"]).exists()]


def dataset_file(cls: str, n: int = 0, split="test") -> Path | None:
    rows = [r for r in dataset_rows(split) if r["class_label"] == cls]
    return DATASET_DIR / rows[n]["relative_path"] if len(rows) > n else None


needs_model = pytest.mark.skipif(not model_available(), reason="no trained Python model")
needs_dataset = pytest.mark.skipif(not dataset_rows(), reason="dataset not generated")


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
@pytest.fixture
def app(tmp_path, monkeypatch):
    from src import create_app
    from src.app_config import TestConfig

    class Cfg(TestConfig):
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + (tmp_path / "test.db").as_posix()
        STORAGE_DIR = str(tmp_path / "storage")
        SEED_DEMO_USERS = True

    application = create_app(Cfg)
    yield application
    from src.extensions import db

    with application.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, username="admin", password=None):
    passwords = {"admin": "Admin@2026", "evaluator": "Evaluate@2026", "reviewer": "Review@2026",
                 "security": "Secure@2026", "maintenance": "Maintain@2026", "user": "User@2026"}
    return client.post("/login", data={"username": username, "password": password or passwords[username]})


@pytest.fixture
def as_admin(client):
    login(client, "admin")
    return client


@pytest.fixture
def no_gtm(app):
    """Make sure GTM is treated as not configured (single-model results)."""
    from src.extensions import db
    from src.services.settings_service import update_settings

    with app.app_context():
        update_settings({"gtm": {"enabled": False}}, "test")
        db.session.commit()
    return app


def upload(client, path: Path, name: str | None = None):
    with open(path, "rb") as fh:
        return client.post("/api/uploads", data={"file": (fh, name or path.name)}, content_type="multipart/form-data")


@pytest.fixture
def gtm_fixture_dir(tmp_path):
    """Copy of the GTM-format test fixture (random weights, NOT a Teachable Machine model)."""
    src = Path(__file__).parent / "fixtures" / "gtm_test_model"
    if not (src / "model.json").exists():
        pytest.skip("run tests/fixtures/make_gtm_test_model.py first")
    dst = tmp_path / "gtm"
    shutil.copytree(src, dst)
    return dst
