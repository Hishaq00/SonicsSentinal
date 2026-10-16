"""Manual review: confirm or correct the detected class, comment, override.

The original model outputs (Python scores, GTM scores, fused decision) are
never modified; the reviewer's decision is stored separately (reviews table +
reviewed_category / reviewed_severity on the record).
"""
from __future__ import annotations

from alert_rules import rules_by_category
from config import class_names, unknown_label

from ..extensions import db
from ..models import Review, utcnow
from . import alerts_service, audit

# Why a record lands in the manual-review queue (SRS step 17); used as queue filters.
REVIEW_REASONS = [
    "Different predictions from the two models",
    "Low confidence",
    "Poor audio quality",
    "Similar top-class confidence scores",
    "Overlapping sounds",
    "Unsupported sound pattern",
    "Critical event without sufficient model agreement",
    "Possible false alarm",
]

DECISIONS = {"confirmed": "Confirmed", "corrected": "Corrected", "false_alarm": "False alarm"}


def review_categories() -> list[str]:
    return class_names() + [unknown_label()]


def submit_review(record, reviewer, decision: str, category: str | None, comment: str = "",
                  recommended_action: str = "", override: bool = False) -> Review:
    if decision not in DECISIONS:
        raise ValueError("Choose confirm, correct or false alarm.")
    original = record.final_category
    if decision == "confirmed":
        category = original
    if not category or category not in review_categories():
        raise ValueError("Choose a valid sound category.")
    if decision == "corrected" and category == original:
        raise ValueError("The corrected category is the same as the detected one; use Confirm instead.")
    if decision == "false_alarm" and category == original:
        raise ValueError("For a false alarm choose what the sound actually was.")
    rules = rules_by_category()
    rule = rules.get(category, {})
    severity = rule.get("severity", "Low")
    is_override = bool(override) or category != original

    review = Review(record_id=record.id, reviewer_id=reviewer.id, decision=decision, original_category=original,
                    reviewed_category=category, reviewed_severity=severity, is_override=is_override,
                    comment=(comment or "").strip()[:2000], recommended_action=(recommended_action or "").strip()[:1000])
    db.session.add(review)
    record.reviewed_category = category
    record.reviewed_severity = severity
    record.reviewed_by = reviewer.id
    record.reviewed_at = utcnow()
    record.needs_review = False
    record.status = "Reviewed"
    if review.recommended_action:
        record.recommended_action = review.recommended_action

    # Alerts follow the reviewer: dismiss alerts for a rejected category, raise one for a confirmed critical event.
    open_alerts = [a for a in record.alerts if a.status in alerts_service.ACTIVE_STATES]
    for a in open_alerts:
        if a.category != category:
            a.status = "Dismissed"
            a.resolved_at = utcnow()
            if a.first_action_at is None:
                a.first_action_at = utcnow()
            db.session.add(alerts_service.AlertAction(alert_id=a.id, user_id=reviewer.id, action="dismiss",
                                                      comment=f"Reviewer override: sound is {category}."))
    has_alert_for_category = any(a.category == category for a in record.alerts)
    if rule.get("alert_enabled") and not has_alert_for_category and decision != "confirmed":
        alerts_service.create_manual_alert(record, category, severity, reviewer, rule)
    db.session.flush()
    audit.log("REVIEW", "record", record.audio_id,
              {"decision": decision, "from": original, "to": category, "comment": review.comment})
    if is_override:
        audit.log("OVERRIDE", "record", record.audio_id,
                  {"model_category": original, "reviewer_category": category, "severity": severity,
                   "python": record.python_top, "gtm": record.gtm_top})
    return review


def close_record(record, user, comment: str = "") -> None:
    if record.status == "Closed":
        raise ValueError("Event is already closed.")
    record.status = "Closed"
    record.closed_at = utcnow()
    for a in record.alerts:
        if a.status in alerts_service.ACTIVE_STATES:
            a.status = "Resolved"
            a.resolved_at = utcnow()
            if a.first_action_at is None:
                a.first_action_at = utcnow()
            db.session.add(alerts_service.AlertAction(alert_id=a.id, user_id=user.id, action="resolve",
                                                      comment="Event closed. " + (comment or "")))
    audit.log("EVENT_CLOSED", "record", record.audio_id, {"comment": comment})
    db.session.flush()
