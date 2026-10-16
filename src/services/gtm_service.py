"""Google Teachable Machine (GTM) integration.

The GTM audio model is a TensorFlow.js model. It runs in the browser (the
supported way to use an exported GTM audio model) and is loaded either from
``gtm_model/model/`` (model.json, metadata.json, weights.bin exported from
GTM) or from the hosted GTM link (https://teachablemachine.withgoogle.com/models/<id>/).

The browser computes GTM's own input spectrogram from the *same* segment the
Python model analysed, runs the model, and posts the class scores back. The
Python prediction is never sent to the browser before GTM has answered, so
GTM predicts independently.
"""
from __future__ import annotations

import json
import math
import re

from flask import url_for

from config import GTM_MODEL_DIR, class_names

from ..extensions import db
from ..models import ModelVersion, utcnow
from .settings_service import get_settings

GTM_PARAMS = {"sample_rate": 44100, "frames": 43, "bins": 232, "fft": 2048, "hop": 1024}


def local_files() -> dict:
    files = {p.name: p.stat().st_size for p in GTM_MODEL_DIR.glob("*") if p.is_file()} if GTM_MODEL_DIR.exists() else {}
    return files


def local_available() -> bool:
    files = local_files()
    return "model.json" in files and "metadata.json" in files and any(f.endswith(".bin") for f in files)


def read_local_metadata() -> dict | None:
    path = GTM_MODEL_DIR / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def labels_from_metadata(meta: dict | None) -> list[str]:
    if not meta:
        return []
    return list(meta.get("wordLabels") or meta.get("words") or meta.get("labels") or [])


def label_check(labels: list[str], expected: list[str] | None = None) -> dict:
    expected = expected or class_names()
    missing = [c for c in expected if c not in labels]
    extra = [c for c in labels if c not in expected]
    return {"ok": not missing and not extra, "missing": missing, "extra": extra}


def model_base_url() -> str | None:
    s = get_settings()["gtm"]
    if s.get("source") == "url" and s.get("url"):
        return s["url"]
    if local_available():
        return url_for("gtm.model_file", filename="x")[:-1]
    return None


def client_config(expected_labels: list[str] | None = None) -> dict:
    s = get_settings()["gtm"]
    base = model_base_url()
    meta = read_local_metadata() if s.get("source", "local") == "local" else {}
    ts = str((meta or {}).get("timeStamp") or "")
    m_name = str((meta or {}).get("modelName") or "gtm-model")
    v_label = s.get("version_label") or (f"{m_name}@{ts}" if ts else "")
    return {
        "enabled": bool(s.get("enabled")) and base is not None,
        "configured": base is not None,
        "source": s.get("source", "local"),
        "model_url": (base + "model.json") if base else None,
        "metadata_url": (base + "metadata.json") if base else None,
        "version_label": clean_version(v_label),
        "expected_labels": expected_labels or class_names(),
        "status_url": url_for("gtm.report_status"),
        **GTM_PARAMS,
    }


def is_enabled() -> bool:
    s = get_settings()["gtm"]
    return bool(s.get("enabled")) and model_base_url() is not None


def clean_version(text: str | None) -> str:
    text = re.sub(r"[^\w .:@+\-/]", "", str(text or ""))[:120].strip()
    return text or "gtm-unversioned"


def parse_scores(raw: dict, expected: list[str]) -> dict[str, float]:
    """Validate one GTM score dictionary: every expected class, numbers in [0, 1]."""
    if not isinstance(raw, dict):
        raise ValueError("GTM scores must be an object of class -> probability")
    scores: dict[str, float] = {}
    for label in expected:
        if label not in raw:
            raise ValueError(f"GTM result is missing the class '{label}'. Check that the GTM class names match.")
        try:
            value = float(raw[label])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid GTM score for '{label}'")
        if math.isnan(value) or value < -1e-6 or value > 1 + 1e-6:
            raise ValueError(f"GTM score for '{label}' is outside 0-1")
        scores[label] = min(1.0, max(0.0, value))
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("GTM scores sum to zero")
    if abs(total - 1.0) > 0.05:  # softmax outputs sum to 1; allow float noise only
        raise ValueError(f"GTM scores sum to {total:.3f}, expected 1.0")
    return {k: v / total for k, v in scores.items()}


def register_version(version: str, labels: list[str], source: str, username: str, notes: str = "") -> ModelVersion:
    mv = ModelVersion.query.filter_by(model_type="gtm", version=version).first()
    if mv is None:
        mv = ModelVersion(model_type="gtm", version=version, algorithm="Teachable Machine audio (speech-commands transfer learning)",
                          source=source, labels=labels, registered_by=username, notes=notes)
        db.session.add(mv)
    ModelVersion.query.filter(ModelVersion.model_type == "gtm", ModelVersion.version != version).update({"is_active": False})
    mv.is_active = True
    mv.labels = labels
    mv.source = source
    mv.registered_at = mv.registered_at or utcnow()
    db.session.flush()
    return mv
