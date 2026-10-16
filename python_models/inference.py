"""Load the active Python model and classify audio segments.

The model file contains the fitted scikit-learn estimator *and* its model card
(classes, feature configuration, pre-processing settings, metrics). Inference
therefore always extracts features exactly the way the model was trained.

A hybrid model file (``python -m python_models.train_cnn``) also holds a log-mel
CNN (``python_models/cnn.py``) and the fusion weights chosen on the validation
split; its probabilities are averaged with the feature model's.

Model files trained since 27 Sep 2026 also hold a ``temperature`` fitted on the
validation split, so the confidence shown matches how often the model is right
(temperature scaling never changes the predicted class).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from config import MODELS_DIR
from feature_extraction.features import FeatureConfig, extract_features


class ModelNotAvailable(RuntimeError):
    pass


class PythonSoundClassifier:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._artifact: dict | None = None
        self._cnn = None
        self._loaded_key: tuple | None = None

    # ------------------------------------------------------------------ #
    def active_pointer(self) -> dict | None:
        path = self.models_dir / "active.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _ensure_loaded(self) -> dict:
        pointer = self.active_pointer()
        if not pointer:
            raise ModelNotAvailable("No trained Python model found. Run: python -m python_models.train")
        file = self.models_dir / pointer["file"]
        if not file.exists():
            raise ModelNotAvailable(f"Model file {file.name} is missing.")
        key = (str(file), file.stat().st_mtime)
        if self._artifact is None or self._loaded_key != key:
            with self._lock:
                if self._artifact is None or self._loaded_key != key:
                    import joblib

                    artifact = joblib.load(file)
                    if not isinstance(artifact, dict) or "model" not in artifact or "card" not in artifact:
                        raise ModelNotAvailable(f"{file.name} is not a SonicSentinel model file.")
                    self._artifact = artifact
                    self._cnn = None
                    if artifact.get("cnn"):
                        from .cnn import NumpyCNN

                        self._cnn = NumpyCNN(artifact["cnn"])
                    self._loaded_key = key
        return self._artifact

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    @property
    def card(self) -> dict:
        return self._ensure_loaded()["card"]

    @property
    def version(self) -> str:
        return self.card["version"]

    @property
    def classes(self) -> list[str]:
        return list(self.card["classes"])

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig.from_dict(self.card.get("feature_config"))

    def features(self, segments: list[np.ndarray]) -> np.ndarray:
        cfg = self.feature_config
        return np.vstack([extract_features(s, cfg) for s in segments])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities of the feature-based model only (for a hybrid file: before fusion with the CNN)."""
        model = self._ensure_loaded()["model"]
        return np.asarray(model.predict_proba(X), dtype=np.float64)

    def predict_segments(self, segments: list[np.ndarray]) -> list[dict[str, float]]:
        """Class -> probability for every segment (probabilities sum to 1)."""
        if not segments:
            return []
        artifact = self._ensure_loaded()
        proba = self.predict_proba(self.features(segments))
        if self._cnn is not None:
            from .cnn import fuse

            fusion = artifact["fusion"]
            proba = fuse(proba, self._cnn.predict_proba_segments(segments), fusion["cnn_weight"],
                         fusion.get("critical_weights"))
        if artifact.get("temperature"):
            from .evaluation import temper

            proba = temper(proba, artifact["temperature"])
        classes = self.classes
        return [{classes[j]: float(row[j]) for j in range(len(classes))} for row in proba]

    def list_models(self) -> list[dict]:
        cards = []
        for p in sorted(self.models_dir.glob("py-v*.json")):
            try:
                cards.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(cards, key=lambda c: c.get("trained_at", ""), reverse=True)

    def activate(self, version: str) -> None:
        file = self.models_dir / f"{version}.joblib"
        if not file.exists():
            raise ModelNotAvailable(f"Model {version} not found.")
        (self.models_dir / "active.json").write_text(
            json.dumps({"version": version, "file": file.name}, indent=2), encoding="utf-8"
        )
        self._artifact = None
        self._cnn = None
        self._loaded_key = None


_default: PythonSoundClassifier | None = None


def get_classifier() -> PythonSoundClassifier:
    global _default
    if _default is None:
        _default = PythonSoundClassifier()
    return _default
