"""Runtime settings: defaults from config/settings.default.json, overrides stored
in the database (table app_settings) and edited on Admin > Settings."""
from __future__ import annotations

from flask import g, has_app_context

from audio_preprocessing.loader import SUPPORTED_EXTENSIONS
from config import deep_merge, load_default_settings

from ..extensions import db
from ..models import AppSetting

SETTINGS_KEY = "settings"

# (section, key, label, type, min, max, help)
SCHEMA: list[tuple] = [
    ("detection", "min_confidence", "Minimum confidence for an automatic result", float, 0.0, 1.0,
     "Below this fused confidence the result is marked uncertain and routed to review."),
    ("detection", "top_two_margin", "Top-two margin threshold", float, 0.0, 1.0,
     "Minimum difference between the highest and second-highest class confidence."),
    ("detection", "max_confidence_difference", "Max. confidence difference for an Acceptable Match", float, 0.0, 1.0,
     "|Python top confidence - GTM top confidence| allowed when both models agree."),
    ("detection", "unknown_threshold", "Unknown-sound threshold", float, 0.0, 1.0,
     "Fused confidence below this becomes 'Unknown Sound'."),
    ("detection", "overlap_threshold", "Overlapping-sound threshold", float, 0.05, 0.5,
     "Two or more event classes above this confidence are reported as overlapping sounds."),
    ("detection", "fusion_weight_python", "Python model weight in the fused score", float, 0.0, 1.0,
     "0.5 = both models count equally; GTM weight = 1 - this value."),
    ("detection", "fusion_weight_python_live", "Python model weight on the live monitor", float, 0.0, 1.0,
     "Same weighting for microphone windows, where the Python model is the more reliable of the two."),
    ("detection", "live_smoothing_windows", "Live windows averaged per result", int, 1, 6,
     "Live windows overlap, so the result is the mean of the last N. 1 = decide every window on its own."),
    ("detection", "confidence_high", "'High' confidence level from", float, 0.0, 1.0, ""),
    ("detection", "confidence_medium", "'Medium' confidence level from", float, 0.0, 1.0, ""),
    ("segmentation", "segment_seconds", "Upload segment duration (s)", float, 0.5, 5.0,
     "Long recordings are cut into fixed windows of this length."),
    ("segmentation", "segment_hop_seconds", "Upload segment hop (s)", float, 0.25, 5.0,
     "Distance between segment starts; smaller than the duration = overlapping segments."),
    ("segmentation", "live_window_seconds", "Live window duration (s)", float, 1.0, 3.0,
     "Microphone audio is analysed in windows of this length (1-3 s)."),
    ("segmentation", "live_hop_seconds", "Live window interval (s)", float, 0.5, 3.0,
     "A new live prediction is produced every interval."),
    ("preprocessing", "noise_reduction", "Noise reduction", bool, None, None,
     "Adaptive spectral gating (skipped automatically for stationary sounds)."),
    ("preprocessing", "trim_silence", "Trim leading / trailing silence", bool, None, None, ""),
    ("validation", "max_file_mb", "Maximum file size (MB)", float, 1, 30, ""),
    ("validation", "min_duration_seconds", "Minimum duration (s)", float, 0.1, 10, ""),
    ("validation", "max_duration_seconds", "Maximum duration (s)", float, 5, 3600, ""),
    ("validation", "min_sample_rate", "Minimum sampling rate (Hz)", int, 4000, 48000, ""),
    ("validation", "max_channels", "Maximum channels", int, 1, 32, ""),
    ("validation", "max_batch_files", "Maximum files per batch", int, 1, 200, ""),
    ("quality", "silence_dbfs", "Silence threshold (dBFS)", float, -90, -30, ""),
    ("quality", "low_signal_dbfs", "Low-signal threshold (dBFS)", float, -70, -10, ""),
    ("quality", "clipping_warning_ratio", "Clipping warning ratio", float, 0.0, 0.5, ""),
    ("quality", "clipping_severe_ratio", "Severe clipping ratio", float, 0.0, 0.5, ""),
    ("quality", "noise_floor_limit_dbfs", "Background-noise limit (dBFS)", float, -80, 0,
     "Background Noise above this level is escalated (Medium severity)."),
    ("live", "store_audio", "Store live audio", str, None, None,
     "events = only windows with a detection, all = every window, none = predictions only."),
    ("live", "repeat_window_seconds", "Repeated-detection period (s)", int, 1, 300, ""),
    ("live", "alert_cooldown_seconds", "Alert cool-down per category (s)", int, 0, 600,
     "Repeated detections inside this period update the existing alert instead of creating a new one."),
    ("retention", "audio_days", "Keep uploaded audio (days)", int, 1, 3650, ""),
    ("retention", "live_audio_days", "Keep live audio (days)", int, 1, 3650, ""),
    ("retention", "records_days", "Keep event records (days)", int, 1, 3650, ""),
    ("security", "max_failed_logins", "Failed logins before lock-out", int, 3, 20, ""),
    ("security", "lockout_minutes", "Lock-out duration (minutes)", int, 1, 1440, ""),
    ("monitoring", "failed_upload_limit", "Failed uploads before an admin alert", int, 2, 100, ""),
    ("monitoring", "critical_alert_limit", "Critical alerts per window before an admin alert", int, 2, 500, ""),
    ("monitoring", "low_confidence_min_predictions", "Min. predictions for a low-confidence spike", int, 5, 1000, ""),
]
CHOICES = {("live", "store_audio"): ["events", "all", "none"], ("gtm", "source"): ["local", "url"]}


def _stored() -> dict:
    row = db.session.get(AppSetting, SETTINGS_KEY)
    return dict(row.value or {}) if row else {}


def get_settings(fresh: bool = False) -> dict:
    """Merged settings, cached for the duration of a request."""
    if has_app_context() and not fresh and "ss_settings" in g:
        return g.ss_settings
    merged = deep_merge(load_default_settings(), _stored())
    if has_app_context():
        g.ss_settings = merged
    return merged


def coerce(section: str, key: str, raw, typ, lo, hi):
    if typ is bool:
        return raw in (True, "1", "true", "on", "yes")
    if typ is str:
        value = str(raw).strip()
        choices = CHOICES.get((section, key))
        if choices and value not in choices:
            raise ValueError(f"must be one of {', '.join(choices)}")
        return value
    try:
        value = typ(float(raw)) if typ is int else float(raw)
    except (TypeError, ValueError):
        raise ValueError("must be a number")
    if lo is not None and value < lo:
        raise ValueError(f"must be at least {lo}")
    if hi is not None and value > hi:
        raise ValueError(f"must be at most {hi}")
    return value


def update_settings(changes: dict, username: str) -> tuple[dict, list[str]]:
    """Validate and store a nested dict of changes. Returns (diff, errors)."""
    current = get_settings(fresh=True)
    stored = _stored()
    errors: list[str] = []
    diff: dict = {}
    schema = {(s, k): (t, lo, hi, label) for s, k, label, t, lo, hi, _ in SCHEMA}
    for section, values in changes.items():
        for key, raw in values.items():
            if section == "validation" and key == "allowed_formats":
                fmts = [f.strip().lower().lstrip(".") for f in (raw if isinstance(raw, list) else str(raw).split(","))]
                fmts = [f for f in fmts if f]
                bad = [f for f in fmts if f not in SUPPORTED_EXTENSIONS]
                if bad or not fmts:
                    errors.append(f"Allowed formats: unsupported {bad or 'empty list'} "
                                  f"(decoder supports {', '.join(SUPPORTED_EXTENSIONS)}).")
                    continue
                value = fmts
            elif section == "gtm":
                if key == "enabled":
                    value = raw in (True, "1", "true", "on", "yes")
                elif key == "source":
                    value = raw if raw in CHOICES[("gtm", "source")] else "local"
                elif key in ("url", "version_label"):
                    value = str(raw or "").strip()
                    if key == "url" and value and not value.startswith("https://"):
                        errors.append("GTM model URL must start with https://")
                        continue
                    if key == "url" and value and not value.endswith("/"):
                        value += "/"
                else:
                    continue
            elif (section, key) in schema:
                typ, lo, hi, label = schema[(section, key)]
                try:
                    value = coerce(section, key, raw, typ, lo, hi)
                except ValueError as exc:
                    errors.append(f"{label}: {exc}.")
                    continue
            else:
                continue
            if current.get(section, {}).get(key) != value:
                diff.setdefault(section, {})[key] = {"from": current.get(section, {}).get(key), "to": value}
                stored.setdefault(section, {})[key] = value
    merged = deep_merge(load_default_settings(), stored)
    det = merged["detection"]
    if det["confidence_medium"] > det["confidence_high"]:
        errors.append("'Medium' confidence level must not be higher than 'High'.")
    if merged["quality"]["clipping_warning_ratio"] > merged["quality"]["clipping_severe_ratio"]:
        errors.append("Clipping warning ratio must not exceed the severe ratio.")
    if errors:
        return {}, errors
    row = db.session.get(AppSetting, SETTINGS_KEY)
    if row is None:
        row = AppSetting(key=SETTINGS_KEY, value=stored, updated_by=username)
        db.session.add(row)
    else:
        row.value = stored
        row.updated_by = username
    db.session.flush()
    if has_app_context():
        g.pop("ss_settings", None)
    return diff, []


def schema_by_section() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for section, key, label, typ, lo, hi, help_text in SCHEMA:
        out.setdefault(section, []).append({
            "section": section, "key": key, "label": label, "type": typ.__name__,
            "min": lo, "max": hi, "help": help_text, "choices": CHOICES.get((section, key)),
        })
    return out
