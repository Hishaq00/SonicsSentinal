"""Data retention: administrators configure how long audio and records are kept.

* uploaded audio older than ``retention.audio_days`` is deleted (record kept),
* live-window audio older than ``retention.live_audio_days`` is deleted,
* event records older than ``retention.records_days`` are deleted entirely.
Runs automatically once a day (lazily, on the first request after 24 h) and
on demand from Admin > Settings or ``flask --app wsgi purge``.
"""
from __future__ import annotations

from datetime import timedelta

from ..extensions import db
from ..models import AppSetting, AudioRecord, utcnow
from . import audit, storage
from .settings_service import get_settings

LAST_RUN_KEY = "retention_last_run"


def run_retention(username: str = "system") -> dict:
    cfg = get_settings()["retention"]
    now = utcnow()
    summary = {"audio_deleted": 0, "live_audio_deleted": 0, "records_deleted": 0}

    cutoff_records = now - timedelta(days=int(cfg["records_days"]))
    old = AudioRecord.query.filter(AudioRecord.created_at < cutoff_records, AudioRecord.is_evaluation.is_(False)).all()
    for rec in old:
        storage.delete_all(rec)
        db.session.delete(rec)
        summary["records_deleted"] += 1

    cutoff_audio = now - timedelta(days=int(cfg["audio_days"]))
    for rec in AudioRecord.query.filter(AudioRecord.created_at < cutoff_audio, AudioRecord.source != "live",
                                        AudioRecord.audio_deleted.is_(False), AudioRecord.is_evaluation.is_(False)).all():
        storage.delete_audio(rec)
        summary["audio_deleted"] += 1

    cutoff_live = now - timedelta(days=int(cfg["live_audio_days"]))
    for rec in AudioRecord.query.filter(AudioRecord.created_at < cutoff_live, AudioRecord.source == "live",
                                        AudioRecord.audio_deleted.is_(False)).all():
        storage.delete_audio(rec)
        summary["live_audio_deleted"] += 1

    row = db.session.get(AppSetting, LAST_RUN_KEY)
    if row is None:
        db.session.add(AppSetting(key=LAST_RUN_KEY, value={"at": now.isoformat(), **summary}, updated_by=username))
    else:
        row.value = {"at": now.isoformat(), **summary}
        row.updated_by = username
    audit.log("RETENTION_RUN", "system", None, {**summary, "settings": cfg}, username=username)
    db.session.commit()
    return summary


def last_run() -> dict | None:
    row = db.session.get(AppSetting, LAST_RUN_KEY)
    return row.value if row else None


def run_if_due() -> dict | None:
    info = last_run()
    if info:
        from datetime import datetime

        try:
            if utcnow() - datetime.fromisoformat(info["at"]) < timedelta(hours=24):
                return None
        except (KeyError, ValueError):
            pass
    return run_retention("scheduler")
