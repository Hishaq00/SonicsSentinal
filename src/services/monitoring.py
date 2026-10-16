"""Monitoring and anomaly alerts for administrators.

Raised for: repeated failed uploads, model failures, unusual low-confidence
spikes, excessive critical alerts, duplicate files and failed login attempts.
An open system alert of the same kind (and subject) is updated instead of
creating a new one, so a burst produces one entry with a counter.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func

from ..extensions import db
from ..models import Alert, AudioRecord, AuditLog, SystemAlert, utcnow
from .settings_service import get_settings

KIND_LABELS = {
    "failed_uploads": "Repeated failed uploads",
    "model_failure": "Model failure",
    "low_confidence_spike": "Low-confidence spike",
    "excessive_critical": "Excessive critical alerts",
    "duplicate_file": "Duplicate file",
    "failed_logins": "Failed login attempts",
}


def raise_alert(kind: str, message: str, details: dict | None = None, severity: str = "Medium",
                subject: str | None = None, dedupe_minutes: int = 60) -> SystemAlert:
    since = utcnow() - timedelta(minutes=dedupe_minutes)
    q = SystemAlert.query.filter(SystemAlert.kind == kind, SystemAlert.resolved.is_(False),
                                 SystemAlert.last_seen_at >= since)
    for existing in q.all():
        if (existing.details or {}).get("subject") == subject:
            existing.count = (existing.count or 1) + 1
            existing.last_seen_at = utcnow()
            existing.message = message
            db.session.flush()
            return existing
    alert = SystemAlert(kind=kind, message=message, severity=severity,
                        details={**(details or {}), "subject": subject})
    db.session.add(alert)
    db.session.flush()
    return alert


def on_failed_upload(user) -> None:
    cfg = get_settings()["monitoring"]
    since = utcnow() - timedelta(minutes=int(cfg.get("failed_upload_window_minutes", 10)))
    n = AuditLog.query.filter(AuditLog.action == "UPLOAD_REJECTED", AuditLog.user_id == user.id,
                              AuditLog.created_at >= since).count()
    if n >= int(cfg.get("failed_upload_limit", 5)):
        raise_alert("failed_uploads", f"{user.username} had {n} rejected uploads in "
                    f"{cfg.get('failed_upload_window_minutes', 10)} minutes.",
                    {"user": user.username, "count": n}, "Medium", subject=user.username)


def on_failed_login(username: str, ip: str) -> None:
    cfg = get_settings()
    window = int(cfg["security"].get("lockout_minutes", 15))
    since = utcnow() - timedelta(minutes=window)
    n_user = AuditLog.query.filter(AuditLog.action == "LOGIN_FAILED", AuditLog.username == username,
                                   AuditLog.created_at >= since).count()
    n_ip = AuditLog.query.filter(AuditLog.action == "LOGIN_FAILED", AuditLog.ip_address == ip,
                                 AuditLog.created_at >= since).count()
    limit = int(cfg["security"].get("max_failed_logins", 5))
    if n_user >= limit or n_ip >= limit * 2:
        raise_alert("failed_logins", f"{max(n_user, n_ip)} failed sign-in attempts for '{username}' "
                    f"(IP {ip}) in {window} minutes.",
                    {"username": username, "ip": ip, "count": max(n_user, n_ip)}, "High", subject=f"{username}|{ip}")


def on_model_failure(model: str, error: str) -> None:
    raise_alert("model_failure", f"{model} model failure: {error[:180]}", {"model": model, "error": error[:500]},
                "High", subject=model, dedupe_minutes=30)


def on_duplicate(record: AudioRecord, original: AudioRecord, near: bool = False) -> None:
    what = "near-duplicate (re-encoded, trimmed or volume-changed copy)" if near else "identical file"
    raise_alert("duplicate_file", f"{record.audio_id} is a {what} of {original.audio_id}.",
                {"record": record.audio_id, "original": original.audio_id, "near": near}, "Low",
                subject=record.audio_id)


def check_low_confidence_spike() -> None:
    cfg = get_settings()
    mon = cfg["monitoring"]
    since = utcnow() - timedelta(minutes=int(mon.get("low_confidence_window_minutes", 30)))
    base = AudioRecord.query.filter(AudioRecord.classified_at >= since, AudioRecord.final_confidence.isnot(None),
                                    AudioRecord.is_evaluation.is_(False))
    total = base.count()
    if total < int(mon.get("low_confidence_min_predictions", 20)):
        return
    low = base.filter(AudioRecord.final_confidence < float(cfg["detection"]["min_confidence"])).count()
    ratio = low / total
    if ratio >= float(cfg["detection"].get("low_confidence_spike_ratio", 0.5)):
        raise_alert("low_confidence_spike", f"{low} of the last {total} predictions ({ratio:.0%}) were below the "
                    f"confidence threshold. Check microphones, audio quality or model drift.",
                    {"low": low, "total": total}, "Medium", subject="global")


def check_excessive_critical() -> None:
    mon = get_settings()["monitoring"]
    minutes = int(mon.get("critical_alert_window_minutes", 10))
    since = utcnow() - timedelta(minutes=minutes)
    n = db.session.query(func.count(Alert.id)).filter(Alert.severity == "Critical", Alert.created_at >= since).scalar()
    if n >= int(mon.get("critical_alert_limit", 10)):
        raise_alert("excessive_critical", f"{n} critical alerts in the last {minutes} minutes. "
                    f"Possible real incident, false-alarm loop or tampering.", {"count": n}, "High", subject="global")


def after_prediction() -> None:
    check_low_confidence_spike()
    check_excessive_critical()
