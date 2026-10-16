"""Audit trail: logins, uploads, microphone sessions, predictions, alerts,
reviews, overrides, exports, model updates and settings changes."""
from __future__ import annotations

from flask import has_request_context, request
from flask_login import current_user

from ..extensions import db
from ..models import AuditLog

ACTIONS = {
    "LOGIN": "Signed in",
    "LOGIN_FAILED": "Failed sign-in",
    "LOGOUT": "Signed out",
    "REGISTER": "Registered",
    "PROFILE_UPDATE": "Updated profile",
    "PASSWORD_CHANGE": "Changed password",
    "UPLOAD": "Uploaded audio",
    "UPLOAD_REJECTED": "Upload rejected",
    "PREDICTION": "Prediction stored",
    "MIC_START": "Started microphone session",
    "MIC_STOP": "Stopped microphone session",
    "MIC_STATUS": "Microphone status changed",
    "MIC_CONSENT": "Accepted microphone notice",
    "ALERT_CREATED": "Alert generated",
    "ALERT_ACTION": "Alert action",
    "REVIEW": "Manual review",
    "OVERRIDE": "Decision override",
    "EVENT_CLOSED": "Event closed",
    "EXPORT": "Data export",
    "REPORT": "Report downloaded",
    "MODEL_UPDATE": "Model update",
    "SETTINGS_CHANGE": "Settings changed",
    "RULES_CHANGE": "Alert rules changed",
    "USER_UPDATE": "User account changed",
    "RETENTION_RUN": "Retention clean-up",
    "EVALUATION": "Model comparison run",
    "SYSTEM_ALERT_RESOLVED": "System alert resolved",
}


def log(action: str, target_type: str | None = None, target_id=None, details: dict | None = None,
        user=None, username: str | None = None, commit: bool = False) -> AuditLog:
    if user is None and has_request_context() and getattr(current_user, "is_authenticated", False):
        user = current_user
    entry = AuditLog(
        user_id=getattr(user, "id", None),
        username=username or getattr(user, "username", None),
        action=action,
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        details=details or {},
    )
    if has_request_context():
        fwd = request.headers.get("X-Forwarded-For", "")
        entry.ip_address = (fwd.split(",")[0].strip() if fwd else request.remote_addr) or ""
        entry.user_agent = (request.user_agent.string or "")[:200]
    db.session.add(entry)
    if commit:
        db.session.commit()
    return entry
