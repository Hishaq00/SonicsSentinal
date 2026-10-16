"""Alert creation, acknowledgement, dismissal, escalation and history."""
from __future__ import annotations

from datetime import timedelta

from ..extensions import db
from ..models import Alert, AlertAction, AudioRecord, utcnow
from ..security import can, can_act_on_alert
from ..utils import next_code
from . import audit

ACTIVE_STATES = ("Active", "Acknowledged", "Escalated")
TRANSITIONS = {
    "acknowledge": {"Active", "Escalated"},
    "escalate": {"Active", "Acknowledged"},
    "dismiss": {"Active", "Acknowledged", "Escalated"},
    "resolve": {"Acknowledged", "Escalated"},
    "reopen": {"Dismissed", "Resolved"},
    "comment": {"Active", "Acknowledged", "Escalated", "Dismissed", "Resolved"},
}
RESULT_STATE = {"acknowledge": "Acknowledged", "escalate": "Escalated", "dismiss": "Dismissed",
                "resolve": "Resolved", "reopen": "Active"}


def _message(decision, source: str) -> str:
    agree = {
        "Acceptable Match": "both models agree",
        "Weak Match": "models agree (weakly)",
        "Model Disagreement": "models disagree",
        "Uncertain Result": "uncertain",
        "Single-Model Result": "Python model only",
    }.get(decision.consistency, decision.consistency)
    where = "live microphone" if source == "live" else "uploaded audio"
    return f"{decision.category} detected in {where} - {decision.confidence * 100:.0f}% confidence, {agree}."


def create_alert(record: AudioRecord, decision, segment_index: int | None = None, session=None,
                 cooldown_seconds: int = 0) -> Alert:
    """Create an alert, or update an open one for the same live session/category during the cool-down."""
    if session is not None and cooldown_seconds > 0:
        since = utcnow() - timedelta(seconds=cooldown_seconds)
        existing = Alert.query.filter(Alert.session_id == session.id, Alert.category == decision.category,
                                      Alert.status.in_(ACTIVE_STATES), Alert.last_seen_at >= since).first()
        if existing:
            existing.occurrences = (existing.occurrences or 1) + 1
            existing.last_seen_at = utcnow()
            existing.confidence = max(existing.confidence or 0, decision.confidence)
            if decision.escalated and existing.severity != decision.severity:
                existing.severity = decision.severity
                existing.auto_escalated = True
                existing.escalated_to = (existing.escalated_to or "") or decision.escalation_note
            db.session.flush()
            return existing
    alert = Alert(
        alert_code=next_code("ALR"),
        record_id=record.id,
        session_id=session.id if session else None,
        segment_index=segment_index,
        category=decision.category,
        severity=decision.severity,
        alert_type=decision.alert_type,
        source=record.source,
        confidence=decision.confidence,
        consistency=decision.consistency,
        message=_message(decision, record.source),
        recommended_action=decision.recommended_action,
        conditions=decision.alert_conditions,
        responsible_roles=decision.responsible_roles,
        auto_escalated=decision.escalated,
        escalated_to=decision.escalation_note or None,
    )
    db.session.add(alert)
    db.session.flush()
    audit.log("ALERT_CREATED", "alert", alert.alert_code,
              {"record": record.audio_id, "category": alert.category, "severity": alert.severity,
               "confidence": round(alert.confidence or 0, 3)}, user=record.user)
    return alert


def create_manual_alert(record: AudioRecord, category: str, severity: str, reviewer, rule: dict) -> Alert:
    alert = Alert(
        alert_code=next_code("ALR"),
        record_id=record.id,
        category=category,
        severity=severity,
        alert_type=rule.get("alert_type", ""),
        source=record.source,
        confidence=None,
        consistency="Reviewer confirmed",
        message=f"{category} confirmed by reviewer {reviewer.username} on {record.audio_id}.",
        recommended_action=rule.get("recommended_action", ""),
        conditions={},
        responsible_roles=rule.get("responsible_roles", []),
    )
    db.session.add(alert)
    db.session.flush()
    audit.log("ALERT_CREATED", "alert", alert.alert_code, {"record": record.audio_id, "category": category,
                                                           "manual": True})
    return alert


def apply_action(alert: Alert, user, action: str, comment: str = "", escalate_to: str = "") -> Alert:
    if action not in TRANSITIONS:
        raise ValueError("Unknown action.")
    if not can_act_on_alert(user, alert):
        raise PermissionError("Your role is not responsible for this alert category.")
    if alert.status not in TRANSITIONS[action]:
        raise ValueError(f"Cannot {action} an alert that is {alert.status}.")
    comment = (comment or "").strip()[:1000]
    if action == "escalate" and not escalate_to.strip():
        raise ValueError("Choose who the alert is escalated to.")
    if action == "dismiss" and not comment:
        raise ValueError("Give a reason when dismissing an alert (for example 'false alarm: door slam').")
    act = AlertAction(alert_id=alert.id, user_id=user.id, action=action, comment=comment,
                      escalate_to=escalate_to.strip()[:160] or None)
    db.session.add(act)
    if action in RESULT_STATE:
        alert.status = RESULT_STATE[action]
    if action == "escalate":
        alert.escalated_to = escalate_to.strip()[:160]
    if action in ("resolve", "dismiss"):
        alert.resolved_at = utcnow()
    if action != "comment" and alert.first_action_at is None:
        alert.first_action_at = utcnow()
    if action == "dismiss" and alert.record and alert.record.status not in ("Reviewed", "Closed"):
        rec = alert.record
        rec.needs_review = True
        rec.review_reasons = list(dict.fromkeys((rec.review_reasons or []) + ["Possible false alarm (alert dismissed)"]))
        rec.status = "Manual Review"
    db.session.flush()
    audit.log("ALERT_ACTION", "alert", alert.alert_code,
              {"action": action, "status": alert.status, "comment": comment, "escalate_to": escalate_to})
    return alert


def visible_alerts(user):
    """Alerts the user may see: operators see all, normal users only their own recordings."""
    q = Alert.query.join(AudioRecord, Alert.record_id == AudioRecord.id).filter(AudioRecord.is_evaluation.is_(False))
    if not can(user, "view_all_records"):
        q = q.filter(AudioRecord.user_id == user.id)
    return q


def feed(user, since_id: int = 0, limit: int = 20) -> list[Alert]:
    return (visible_alerts(user).filter(Alert.id > since_id, Alert.status == "Active")
            .order_by(Alert.id.desc()).limit(limit).all())
