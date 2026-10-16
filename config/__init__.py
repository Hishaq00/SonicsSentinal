"""Shared project paths and configuration loaders.

Every module (web app, training scripts, tests) reads class names and default
settings from here so there is exactly one source of truth.
"""
from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "audio_dataset"
MODELS_DIR = PROJECT_ROOT / "python_models" / "saved"
GTM_DIR = PROJECT_ROOT / "gtm_model"
# Folder holding the exported GTM files (model.json, metadata.json, weights.bin).
GTM_MODEL_DIR = Path(os.environ.get("GTM_MODEL_DIR") or (GTM_DIR / "model"))
GTM_SAMPLES_DIR = GTM_DIR / "training_samples"
RULES_DIR = PROJECT_ROOT / "alert_rules"
RULES_FILE = RULES_DIR / "alert_rules.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATABASE_DIR = PROJECT_ROOT / "database"
STORAGE_DIR = DATA_DIR / "storage"
METADATA_CSV = DATA_DIR / "dataset_metadata.csv"
FEATURES_DIR = DATA_DIR / "features"

SEVERITY_LEVELS = ["Informational", "Low", "Medium", "High", "Critical"]
QUALITY_LEVELS = ["Unusable", "Poor", "Acceptable", "Good"]
CONSISTENCY_STATUSES = [
    "Acceptable Match",
    "Weak Match",
    "Model Disagreement",
    "Uncertain Result",
    "Single-Model Result",
]
EVENT_STATUSES = [
    "Uploaded",
    "Classified",
    "Uncertain",
    "Alert Generated",
    "Manual Review",
    "Reviewed",
    "Closed",
]


@lru_cache(maxsize=1)
def load_classes() -> dict:
    with open(CONFIG_DIR / "classes.json", encoding="utf-8") as fh:
        return json.load(fh)


def class_names() -> list[str]:
    """Ordered list of the class names the models are trained on.

    A class marked ``abstract`` is a parent category (Animal Sound, whose children
    are Dog Bark, Cat, Bird ...). It is never predicted on its own, so it is not part
    of this list; ``class_info()`` still carries it so the interface can show the
    parent next to the species.
    """
    return [c["name"] for c in load_classes()["classes"] if not c.get("abstract")]


def class_info() -> dict[str, dict]:
    return {c["name"]: c for c in load_classes()["classes"]}


def class_parent(name: str) -> str | None:
    """The parent category of a class, e.g. "Animal Sound" for "Dog Bark"."""
    info = class_info().get(name) or {}
    return info.get("parent")


def unknown_label() -> str:
    return load_classes().get("unknown_label", "Unknown Sound")


def slugify_class(name: str) -> str:
    info = class_info().get(name)
    if info:
        return info["slug"]
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def class_code(name: str) -> str:
    info = class_info().get(name)
    if info and info.get("code"):
        return info["code"]
    letters = "".join(ch for ch in name.upper() if ch.isalpha())
    return (letters[:3] or "UNK").ljust(3, "X")


def class_from_folder(folder: str) -> str | None:
    """Map a folder name (class name or slug, any case) to a class name."""
    key = folder.strip().lower().replace("-", "_").replace(" ", "_")
    for c in load_classes()["classes"]:
        if key in (c["slug"], c["name"].lower().replace(" ", "_")):
            return c["name"]
    return None


@lru_cache(maxsize=1)
def _default_settings() -> dict:
    with open(CONFIG_DIR / "settings.default.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_default_settings() -> dict:
    """Return a deep copy so callers can modify it safely."""
    return copy.deepcopy(_default_settings())


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def severity_rank(severity: str) -> int:
    try:
        return SEVERITY_LEVELS.index(severity)
    except ValueError:
        return 0


def quality_rank(quality: str) -> int:
    try:
        return QUALITY_LEVELS.index(quality)
    except ValueError:
        return 0
