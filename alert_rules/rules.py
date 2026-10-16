"""Loading, validating and saving the configurable alert rules (JSON file)."""
from __future__ import annotations

import copy
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import QUALITY_LEVELS, RULES_FILE, SEVERITY_LEVELS, unknown_label

MANUAL_REVIEW_OPTIONS = ["never", "on_uncertain", "on_uncertain_or_unconfirmed", "always"]
ESCALATION_CONDITIONS = ["repeated", "noise_level"]
ROLE_KEYS = ["user", "reviewer", "security", "maintenance", "admin"]

DEFAULT_RULE = {
    "critical": False,
    "alert_enabled": False,
    "alert_type": "environment",
    "severity": "Low",
    "min_confidence": 0.60,
    "min_top_two_margin": 0.10,
    "required_consecutive_detections": 1,
    "repeat_window_seconds": 10,
    "require_model_agreement": False,
    "min_audio_quality": "Poor",
    "manual_review_condition": "on_uncertain",
    "recommended_action": "Review the event and take action if needed.",
    "responsible_roles": ["security"],
    "escalation": {
        "enabled": False,
        "condition": "repeated",
        "min_repeats": 3,
        "within_seconds": 30,
        "min_confidence": 0.8,
        "severity": "High",
        "generate_alert": False,
        "notify": "",
    },
}

_lock = threading.Lock()


def normalize_rule(rule: dict) -> dict:
    out = copy.deepcopy(DEFAULT_RULE)
    for k, v in rule.items():
        if k == "escalation" and isinstance(v, dict):
            out["escalation"].update(v)
        else:
            out[k] = v
    return out


def validate_rules(doc: dict) -> list[str]:
    errors: list[str] = []
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        return ["The file must contain a non-empty 'rules' list."]
    seen = set()
    for i, raw in enumerate(rules):
        r = normalize_rule(raw)
        name = r.get("category") or f"rule #{i + 1}"
        if not r.get("category"):
            errors.append(f"{name}: category is required.")
        if name in seen:
            errors.append(f"{name}: duplicate category.")
        seen.add(name)
        if r["severity"] not in SEVERITY_LEVELS:
            errors.append(f"{name}: severity must be one of {SEVERITY_LEVELS}.")
        for key in ("min_confidence", "min_top_two_margin"):
            try:
                val = float(r[key])
            except (TypeError, ValueError):
                errors.append(f"{name}: {key} must be a number.")
                continue
            if not 0.0 <= val <= 1.0:
                errors.append(f"{name}: {key} must be between 0 and 1.")
        try:
            if int(r["required_consecutive_detections"]) < 1 or int(r["required_consecutive_detections"]) > 10:
                errors.append(f"{name}: required_consecutive_detections must be 1-10.")
            if int(r["repeat_window_seconds"]) < 1:
                errors.append(f"{name}: repeat_window_seconds must be positive.")
        except (TypeError, ValueError):
            errors.append(f"{name}: repetition settings must be whole numbers.")
        if r["min_audio_quality"] not in QUALITY_LEVELS:
            errors.append(f"{name}: min_audio_quality must be one of {QUALITY_LEVELS}.")
        if r["manual_review_condition"] not in MANUAL_REVIEW_OPTIONS:
            errors.append(f"{name}: manual_review_condition must be one of {MANUAL_REVIEW_OPTIONS}.")
        esc = r["escalation"]
        if esc.get("condition") not in ESCALATION_CONDITIONS:
            errors.append(f"{name}: escalation condition must be one of {ESCALATION_CONDITIONS}.")
        if esc.get("severity") not in SEVERITY_LEVELS:
            errors.append(f"{name}: escalation severity must be one of {SEVERITY_LEVELS}.")
        bad_roles = [x for x in r.get("responsible_roles", []) if x not in ROLE_KEYS]
        if bad_roles:
            errors.append(f"{name}: unknown roles {bad_roles}.")
    return errors


def load_rules(path: Path | None = None) -> dict:
    # Resolved per call (not as a default argument) so tests can point RULES_FILE elsewhere.
    path = path or RULES_FILE
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["rules"] = [normalize_rule(r) for r in doc.get("rules", [])]
    return doc


def rules_by_category(doc: dict | None = None) -> dict[str, dict]:
    doc = doc or load_rules()
    table = {r["category"]: r for r in doc["rules"]}
    unknown = unknown_label()
    if unknown not in table:
        table[unknown] = normalize_rule({
            "category": unknown, "severity": "Low", "manual_review_condition": "always",
            "recommended_action": "The sound could not be classified reliably. Listen to it and label it manually.",
        })
    return table


def rule_for(category: str, table: dict[str, dict]) -> dict:
    if category in table:
        return table[category]
    return normalize_rule({"category": category})


def save_rules(doc: dict, user: str, path: Path | None = None) -> dict:
    """Validate, bump the version, keep the previous file in history/ and write atomically."""
    path = path or RULES_FILE
    errors = validate_rules(doc)
    if errors:
        raise ValueError("; ".join(errors))
    with _lock:
        current = load_rules(path) if path.exists() else {"version": 0}
        history = path.parent / "history"
        history.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copyfile(path, history / f"alert_rules_v{current.get('version', 0)}.json")
        new = {
            "version": int(current.get("version", 0)) + 1,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_by": user,
            "notes": doc.get("notes", current.get("notes", "")),
            "rules": [normalize_rule(r) for r in doc["rules"]],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
        tmp.replace(path)
    return new
