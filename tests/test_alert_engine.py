"""Decision engine: model comparison, consistency status, low confidence,
overlapping sounds, unknown handling, severity, alert rules and repeated
detection (pure Python, no app or model needed)."""
from __future__ import annotations

import pytest

from alert_rules import (ALERT_GENERATED, ALERT_NOT_CONFIRMED, ALERT_PENDING, ALERT_SUPPRESSED, NO_ALERT, Context,
                         HistoryItem, decide, rules_by_category, summarize)
from config import class_names, load_default_settings

DET = load_default_settings()["detection"]
RULES = rules_by_category()
CLASSES = class_names()


def scores(top: str, conf: float, second: str | None = None, second_conf: float = 0.0) -> dict[str, float]:
    """Probability vector with ``conf`` on ``top`` (and optionally a runner-up); the rest is spread evenly."""
    s = {c: 0.0 for c in CLASSES}
    s[top] = conf
    if second:
        s[second] = second_conf
    rest = [c for c in CLASSES if s[c] == 0.0]
    left = max(0.0, 1.0 - conf - second_conf)
    for c in rest:
        s[c] = left / len(rest)
    return s


def run(py, gtm=None, quality="Good", metrics=None, ctx=None):
    return decide(py, gtm, quality, metrics, RULES, DET, ctx)


# --------------------------------------------------------------------------- #
# Model comparison (SRS step 11)
# --------------------------------------------------------------------------- #
def test_summary_top_three_and_margin():
    m = summarize(scores("Gunshot", 0.7, "Glass Breaking", 0.2))
    assert m.top == "Gunshot" and m.second == "Glass Breaking"
    assert m.margin == pytest.approx(0.5)
    assert [c for c, _ in m.top3][:2] == ["Gunshot", "Glass Breaking"] and len(m.top3) == 3


def test_confidence_difference_is_absolute_top_class_difference():
    d = run(scores("Gunshot", 0.9), scores("Gunshot", 0.75))
    assert d.class_match is True
    assert d.confidence_difference == pytest.approx(0.15)
    d2 = run(scores("Gunshot", 0.6), scores("Gunshot", 0.95))
    assert d2.confidence_difference == pytest.approx(0.35)


@pytest.mark.parametrize("py,gtm,expected", [
    (("Gunshot", 0.90), ("Gunshot", 0.85), "Acceptable Match"),
    (("Gunshot", 0.95), ("Gunshot", 0.62), "Weak Match"),       # same class, confidences 0.33 apart
    (("Gunshot", 0.90), ("Gunshot", 0.45), "Weak Match"),       # same class, GTM below threshold
    (("Gunshot", 0.85), ("Vehicle Horn", 0.80), "Model Disagreement"),
    (("Gunshot", 0.45), ("Vehicle Horn", 0.40), "Uncertain Result"),
])
def test_consistency_status(py, gtm, expected):
    assert run(scores(*py), scores(*gtm)).consistency == expected


def test_single_model_result_when_gtm_missing():
    d = run(scores("Vehicle Horn", 0.9))
    assert d.consistency == "Single-Model Result" and d.class_match is None and d.confidence_difference is None


def test_python_prediction_is_not_an_input_to_gtm_scores():
    """The engine only reads GTM's own vector: changing Python's scores never changes GTM's summary."""
    g = scores("Vehicle Horn", 0.8)
    a = run(scores("Gunshot", 0.9), dict(g))
    b = run(scores("Dog Bark", 0.9), dict(g))
    assert a.gtm.scores == b.gtm.scores and a.gtm.top == b.gtm.top == "Vehicle Horn"


# --------------------------------------------------------------------------- #
# Uncertainty: low confidence, similar top classes, overlap, unknown
# --------------------------------------------------------------------------- #
def test_low_confidence_is_uncertain_and_reviewed():
    d = run(scores("Vehicle Horn", 0.5, "Dog Bark", 0.2), scores("Vehicle Horn", 0.5, "Dog Bark", 0.2))
    assert d.confidence_level == "Low"
    assert "Low confidence" in d.uncertain_reasons and d.needs_review


def test_similar_top_classes_flagged():
    d = run(scores("Vehicle Horn", 0.5, "Alarm or Siren", 0.42), scores("Vehicle Horn", 0.5, "Alarm or Siren", 0.42))
    assert "Similar top-class confidence scores" in d.uncertain_reasons


def test_overlapping_sounds_detected():
    d = run(scores("Glass Breaking", 0.5, "Panic Scream", 0.4), scores("Glass Breaking", 0.5, "Panic Scream", 0.4))
    assert d.overlapping and set(d.overlap_classes) >= {"Glass Breaking", "Panic Scream"}
    assert "Overlapping sounds" in d.uncertain_reasons


def test_background_second_class_is_not_an_overlap():
    d = run(scores("Gunshot", 0.6, "Background Noise", 0.35), scores("Gunshot", 0.6, "Background Noise", 0.35))
    assert not d.overlapping


def test_very_low_confidence_becomes_unknown():
    flat = {c: 1 / len(CLASSES) for c in CLASSES}
    d = run(flat, flat)
    assert d.category == "Unknown Sound" and "Unsupported sound pattern" in d.uncertain_reasons
    assert d.alert_status == NO_ALERT and d.needs_review


def test_unusable_audio_is_never_classified():
    d = run(scores("Gunshot", 0.99), scores("Gunshot", 0.99), quality="Unusable")
    assert d.category == "Unknown Sound" and d.consistency == "Uncertain Result" and d.alert_status == NO_ALERT


# --------------------------------------------------------------------------- #
# Severity and alert rules (SRS steps 15-16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("category,severity", [
    ("Background Noise", "Informational"), ("Dog Bark", "Low"), ("Machinery Fault", "High"),
    ("Glass Breaking", "High"), ("Alarm or Siren", "High"), ("Aggression", "High"),
    ("Panic Scream", "Critical"), ("Person Asking for Help", "Critical"), ("Gunshot", "Critical"),
])
def test_default_severity_per_category(category, severity):
    assert run(scores(category, 0.9), scores(category, 0.9)).severity == severity


def test_glass_breaking_confident_agreement_generates_alert():
    d = run(scores("Glass Breaking", 0.92), scores("Glass Breaking", 0.88))
    assert d.alert_status == ALERT_GENERATED and not d.needs_review
    assert all(c["passed"] for c in d.alert_conditions.values())


def test_vehicle_horn_never_alerts():
    d = run(scores("Vehicle Horn", 0.95), scores("Vehicle Horn", 0.95))
    assert d.alert_status == NO_ALERT and d.severity == "Low"


def test_gunshot_requires_model_agreement():
    # Python is sure it is a gunshot, GTM hears a vehicle horn: no alert, routed to review
    d = run(scores("Gunshot", 0.95), scores("Vehicle Horn", 0.8),
            ctx=Context(source="upload", n_segments=3, run_length=3))
    assert d.consistency == "Model Disagreement"
    assert d.alert_conditions["model_agreement"]["passed"] is False and d.alert_status == ALERT_SUPPRESSED
    assert d.needs_review and "Different predictions from the two models" in d.review_reasons


def test_poor_quality_blocks_gunshot_alert():
    d = run(scores("Gunshot", 0.95), scores("Gunshot", 0.95), quality="Poor",
            ctx=Context(source="upload", n_segments=3, run_length=3))
    assert not d.alert_conditions["audio_quality"]["passed"] and d.alert_status == ALERT_SUPPRESSED
    assert "Poor audio quality" in d.review_reasons


def test_min_confidence_threshold_is_configurable():
    rules = {k: dict(v) for k, v in RULES.items()}
    py, gtm = scores("Aggression", 0.7), scores("Aggression", 0.7)
    assert decide(py, gtm, "Good", None, rules, DET).alert_status == ALERT_GENERATED
    rules["Aggression"]["min_confidence"] = 0.9
    assert decide(py, gtm, "Good", None, rules, DET).alert_status == ALERT_SUPPRESSED


# --------------------------------------------------------------------------- #
# Repeated detection (SRS step 15 / xl)
# --------------------------------------------------------------------------- #
def test_live_gunshot_pending_then_confirmed_in_next_window():
    py, gtm = scores("Gunshot", 0.9), scores("Gunshot", 0.9)
    first = run(py, gtm, ctx=Context(source="live", time_s=0.0, index=0))
    assert first.alert_status == ALERT_PENDING
    second = run(py, gtm, ctx=Context(source="live", time_s=2.0, index=1,
                                      history=[HistoryItem(0, 0.0, "Gunshot", 0.9)]))
    assert second.consecutive_count == 2 and second.alert_status == ALERT_GENERATED


def test_non_consecutive_history_does_not_count_as_consecutive():
    ctx = Context(source="live", time_s=60.0, index=5, history=[HistoryItem(1, 2.0, "Gunshot", 0.9)])
    d = run(scores("Gunshot", 0.9), scores("Gunshot", 0.9), ctx=ctx)
    assert d.consecutive_count == 1 and d.alert_status == ALERT_PENDING


def test_upload_single_window_not_confirmed():
    ctx = Context(source="upload", n_segments=4, run_length=1, repeat_count=1)
    d = run(scores("Machinery Fault", 0.9), scores("Machinery Fault", 0.9), ctx=ctx)
    assert d.alert_status == ALERT_NOT_CONFIRMED


def test_repeated_aggression_escalates_to_critical():
    hist = [HistoryItem(i, i * 2.0, "Aggression", 0.9) for i in range(3)]
    d = run(scores("Aggression", 0.9), scores("Aggression", 0.9),
            ctx=Context(source="live", time_s=6.0, index=3, history=hist))
    assert d.escalated and d.severity == "Critical" and "Security lead" in d.escalation_note


def test_background_noise_escalates_only_above_noise_limit():
    quiet = run(scores("Background Noise", 0.9), scores("Background Noise", 0.9), metrics={})
    loud = run(scores("Background Noise", 0.9), scores("Background Noise", 0.9),
               metrics={"noise_level_exceeded": True, "noise_floor_dbfs": -18.0})
    assert quiet.severity == "Informational" and quiet.alert_status == NO_ALERT
    assert loud.escalated and loud.severity == "Medium" and loud.alert_status == ALERT_GENERATED


# --------------------------------------------------------------------------- #
# Live smoothing: overlapping windows are decided together, not one by one
# --------------------------------------------------------------------------- #
def test_live_result_is_the_mean_of_the_recent_windows():
    det = {**DET, "live_smoothing_windows": 3}
    previous = [scores("Vehicle Horn", 0.9), scores("Vehicle Horn", 0.9)]
    ctx = Context(source="live", time_s=2.0, index=2, recent_scores=previous)
    d = decide(scores("Vehicle Horn", 0.6), scores("Vehicle Horn", 0.6), "Good", None, RULES, det, ctx)
    assert d.smoothing_windows == 3
    # the reported confidence sits between this window and the two before it
    assert d.window_scores["Vehicle Horn"] < d.fused_scores["Vehicle Horn"] < 0.9


def test_live_smoothing_keeps_one_odd_window_from_flipping_the_category():
    det = {**DET, "live_smoothing_windows": 3}
    horn = [scores("Vehicle Horn", 0.85), scores("Vehicle Horn", 0.85)]
    odd = scores("Dog Bark", 0.55, "Vehicle Horn", 0.3)
    alone = decide(odd, odd, "Good", None, RULES, det, Context(source="live", index=2))
    smoothed = decide(odd, odd, "Good", None, RULES, det,
                      Context(source="live", index=2, recent_scores=horn))
    assert alone.category == "Dog Bark"
    assert smoothed.category == "Vehicle Horn" and smoothed.smoothing_windows == 3


def test_live_smoothing_uses_only_the_windows_it_is_given():
    """An interrupted stream passes no recent windows, so the result is this window alone."""
    det = {**DET, "live_smoothing_windows": 3}
    d = decide(scores("Gunshot", 0.9), scores("Gunshot", 0.9), "Good", None, RULES, det,
               Context(source="live", index=7, recent_scores=[]))
    assert d.smoothing_windows == 1 and d.fused_scores == d.window_scores


def test_uploads_are_never_smoothed():
    det = {**DET, "live_smoothing_windows": 3}
    ctx = Context(source="upload", n_segments=3, run_length=3,
                  recent_scores=[scores("Vehicle Horn", 0.9), scores("Vehicle Horn", 0.9)])
    d = decide(scores("Dog Bark", 0.8), scores("Dog Bark", 0.8), "Good", None, RULES, det, ctx)
    assert d.smoothing_windows == 1 and d.category == "Dog Bark"


def test_live_path_has_its_own_fusion_weight():
    det = {**DET, "fusion_weight_python": 0.5, "fusion_weight_python_live": 0.9,
           "live_smoothing_windows": 1}
    py, gtm = scores("Gunshot", 0.9), scores("Vehicle Horn", 0.9)
    upload = decide(py, gtm, "Good", None, RULES, det, Context(source="upload"))
    live = decide(py, gtm, "Good", None, RULES, det, Context(source="live", index=0))
    assert live.window_scores["Gunshot"] > upload.window_scores["Gunshot"]
    assert live.category == "Gunshot"
