"""Database models (SQLAlchemy). SQLite by default; any SQLAlchemy URL works.

Tables: users, audio_records, segments, alerts, alert_actions, reviews,
mic_sessions, batches, model_versions, audit_logs, system_alerts,
app_settings, evaluation_runs, id_counters.
"""
from __future__ import annotations

from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import Index
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


ROLES = {
    "user": "Normal User",
    "reviewer": "Audio Reviewer",
    "security": "Security Operator",
    "maintenance": "Maintenance Operator",
    "admin": "Administrator",
}


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    user_code = db.Column(db.String(16), unique=True, nullable=False, index=True)
    username = db.Column(db.String(40), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    phone = db.Column(db.String(30), default="")
    organization = db.Column(db.String(120), default="")
    department = db.Column(db.String(120), default="")
    job_title = db.Column(db.String(120), default="")
    active_account = db.Column(db.Boolean, default=True, nullable=False)
    mic_consent_at = db.Column(db.DateTime)
    preferences = db.Column(db.JSON, default=dict)
    failed_logins = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime)
    last_login_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    @property
    def is_active(self) -> bool:  # Flask-Login
        return bool(self.active_account)

    @property
    def role_label(self) -> str:
        return ROLES.get(self.role, self.role)

    @property
    def initials(self) -> str:
        parts = [p for p in (self.full_name or self.username).split() if p]
        return "".join(p[0] for p in parts[:2]).upper() or "?"

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def pref(self, key: str, default=None):
        return (self.preferences or {}).get(key, default)


class Batch(db.Model):
    __tablename__ = "batches"
    id = db.Column(db.Integer, primary_key=True)
    batch_code = db.Column(db.String(24), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    file_count = db.Column(db.Integer, default=0)
    processed = db.Column(db.Integer, default=0)
    failed = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=utcnow)
    user = db.relationship("User")


class MicSession(db.Model):
    __tablename__ = "mic_sessions"
    id = db.Column(db.Integer, primary_key=True)
    session_code = db.Column(db.String(24), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    status = db.Column(db.String(16), default="active")  # active, paused, stopped, disconnected
    window_seconds = db.Column(db.Float, default=2.0)
    hop_seconds = db.Column(db.Float, default=1.0)
    device_label = db.Column(db.String(200), default="")
    location = db.Column(db.String(120), default="")
    windows_processed = db.Column(db.Integer, default=0)
    silent_windows = db.Column(db.Integer, default=0)
    detections = db.Column(db.Integer, default=0)
    alerts = db.Column(db.Integer, default=0)
    avg_latency_ms = db.Column(db.Float, default=0.0)
    started_at = db.Column(db.DateTime, default=utcnow)
    last_window_at = db.Column(db.DateTime)
    ended_at = db.Column(db.DateTime)
    user = db.relationship("User")


class EvaluationRun(db.Model):
    __tablename__ = "evaluation_runs"
    id = db.Column(db.Integer, primary_key=True)
    run_code = db.Column(db.String(24), unique=True, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    status = db.Column(db.String(16), default="running")
    per_class = db.Column(db.Integer, default=10)
    total = db.Column(db.Integer, default=0)
    completed = db.Column(db.Integer, default=0)
    python_version = db.Column(db.String(64))
    gtm_version = db.Column(db.String(120))
    summary = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=utcnow)
    finished_at = db.Column(db.DateTime)
    user = db.relationship("User")


class AudioRecord(db.Model):
    __tablename__ = "audio_records"
    id = db.Column(db.Integer, primary_key=True)
    audio_id = db.Column(db.String(24), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    source = db.Column(db.String(12), default="upload")  # upload, batch, live, evaluation
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"))
    session_id = db.Column(db.Integer, db.ForeignKey("mic_sessions.id"), index=True)
    evaluation_run_id = db.Column(db.Integer, db.ForeignKey("evaluation_runs.id"), index=True)
    window_index = db.Column(db.Integer)
    dataset_audio_id = db.Column(db.String(32))
    actual_class = db.Column(db.String(40))

    original_filename = db.Column(db.String(255), default="")
    stored_path = db.Column(db.String(400))
    processed_path = db.Column(db.String(400))
    spectrogram_path = db.Column(db.String(400))
    waveform_path = db.Column(db.String(400))
    audio_deleted = db.Column(db.Boolean, default=False)

    format = db.Column(db.String(16))
    codec = db.Column(db.String(32))
    file_size = db.Column(db.Integer)
    duration = db.Column(db.Float)
    sample_rate = db.Column(db.Integer)
    channels = db.Column(db.Integer)
    bit_depth = db.Column(db.Integer)
    sha256 = db.Column(db.String(64), index=True)
    fingerprint = db.Column(db.LargeBinary)
    fp_descriptor = db.Column(db.LargeBinary)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("audio_records.id"))
    near_duplicate_of_id = db.Column(db.Integer, db.ForeignKey("audio_records.id"))
    near_duplicate_ber = db.Column(db.Float)

    validation = db.Column(db.JSON)
    metadata_extra = db.Column(db.JSON)
    preprocessing = db.Column(db.JSON)
    quality_label = db.Column(db.String(12), index=True)
    quality_score = db.Column(db.Integer)
    quality = db.Column(db.JSON)

    status = db.Column(db.String(20), default="Uploaded", index=True)
    gtm_status = db.Column(db.String(12), default="pending")  # pending, completed, unavailable, failed

    python_top = db.Column(db.String(40))
    python_confidence = db.Column(db.Float)
    python_scores = db.Column(db.JSON)
    python_margin = db.Column(db.Float)
    gtm_top = db.Column(db.String(40))
    gtm_confidence = db.Column(db.Float)
    gtm_scores = db.Column(db.JSON)
    gtm_margin = db.Column(db.Float)
    class_match = db.Column(db.Boolean)
    confidence_difference = db.Column(db.Float)
    margin_difference = db.Column(db.Float)
    consistency = db.Column(db.String(24), index=True)

    final_category = db.Column(db.String(40), index=True)
    final_confidence = db.Column(db.Float, index=True)
    confidence_level = db.Column(db.String(8))
    severity = db.Column(db.String(16), index=True)
    alert_status = db.Column(db.String(24))
    needs_review = db.Column(db.Boolean, default=False, index=True)
    review_reasons = db.Column(db.JSON)
    uncertain_reasons = db.Column(db.JSON)
    overlapping = db.Column(db.Boolean, default=False)
    overlap_classes = db.Column(db.JSON)
    recommended_action = db.Column(db.Text)
    decision = db.Column(db.JSON)
    events = db.Column(db.JSON)
    python_model_version = db.Column(db.String(64))
    gtm_model_version = db.Column(db.String(120))

    reviewed_category = db.Column(db.String(40))
    reviewed_severity = db.Column(db.String(16))
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    reviewed_at = db.Column(db.DateTime)

    error_message = db.Column(db.Text)
    processing_ms = db.Column(db.Integer)
    is_evaluation = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    classified_at = db.Column(db.DateTime)
    closed_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", foreign_keys=[user_id])
    reviewer = db.relationship("User", foreign_keys=[reviewed_by])
    session = db.relationship("MicSession")
    batch = db.relationship("Batch")
    segments = db.relationship("Segment", back_populates="record", cascade="all, delete-orphan",
                               order_by="Segment.index")
    alerts = db.relationship("Alert", back_populates="record", cascade="all, delete-orphan")
    reviews = db.relationship("Review", back_populates="record", cascade="all, delete-orphan",
                              order_by="Review.created_at")
    duplicate_of = db.relationship("AudioRecord", remote_side=[id], foreign_keys=[duplicate_of_id])
    near_duplicate_of = db.relationship("AudioRecord", remote_side=[id], foreign_keys=[near_duplicate_of_id])

    __table_args__ = (Index("ix_records_created_category", "created_at", "final_category"),)

    @property
    def effective_category(self) -> str | None:
        return self.reviewed_category or self.final_category

    @property
    def effective_severity(self) -> str | None:
        return self.reviewed_severity or self.severity

    @property
    def display_name(self) -> str:
        if self.source == "live":
            return f"Live window #{self.window_index or 0}"
        return self.original_filename or self.audio_id


class Segment(db.Model):
    __tablename__ = "segments"
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_records.id"), nullable=False, index=True)
    index = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.Float, nullable=False)  # seconds in the original recording
    end_time = db.Column(db.Float, nullable=False)
    proc_start = db.Column(db.Float, nullable=False)  # seconds in the pre-processed audio
    proc_end = db.Column(db.Float, nullable=False)
    rms_dbfs = db.Column(db.Float)
    is_event = db.Column(db.Boolean, default=True)
    silent = db.Column(db.Boolean, default=False)
    quality_label = db.Column(db.String(12))
    quality = db.Column(db.JSON)
    python_scores = db.Column(db.JSON)
    python_top = db.Column(db.String(40))
    python_confidence = db.Column(db.Float)
    gtm_scores = db.Column(db.JSON)
    gtm_top = db.Column(db.String(40))
    gtm_confidence = db.Column(db.Float)
    gtm_window_start = db.Column(db.Float)
    final_category = db.Column(db.String(40))
    final_confidence = db.Column(db.Float)
    consistency = db.Column(db.String(24))
    severity = db.Column(db.String(16))
    alert_status = db.Column(db.String(24))
    needs_review = db.Column(db.Boolean, default=False)
    decision = db.Column(db.JSON)
    record = db.relationship("AudioRecord", back_populates="segments")


class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    alert_code = db.Column(db.String(24), unique=True, nullable=False)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_records.id"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("mic_sessions.id"))
    segment_index = db.Column(db.Integer)
    category = db.Column(db.String(40), nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=False, index=True)
    alert_type = db.Column(db.String(20))
    source = db.Column(db.String(12))
    confidence = db.Column(db.Float)
    consistency = db.Column(db.String(24))
    message = db.Column(db.String(300))
    recommended_action = db.Column(db.Text)
    conditions = db.Column(db.JSON)
    responsible_roles = db.Column(db.JSON)
    status = db.Column(db.String(16), default="Active", index=True)
    auto_escalated = db.Column(db.Boolean, default=False)
    escalated_to = db.Column(db.String(160))
    occurrences = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    last_seen_at = db.Column(db.DateTime, default=utcnow)
    first_action_at = db.Column(db.DateTime)
    resolved_at = db.Column(db.DateTime)
    record = db.relationship("AudioRecord", back_populates="alerts")
    actions = db.relationship("AlertAction", back_populates="alert", cascade="all, delete-orphan",
                              order_by="AlertAction.created_at")

    @property
    def response_seconds(self) -> float | None:
        if self.first_action_at and self.created_at:
            return (self.first_action_at - self.created_at).total_seconds()
        return None


class AlertAction(db.Model):
    __tablename__ = "alert_actions"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey("alerts.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    action = db.Column(db.String(16), nullable=False)  # acknowledge, dismiss, escalate, resolve, comment
    comment = db.Column(db.Text, default="")
    escalate_to = db.Column(db.String(160))
    created_at = db.Column(db.DateTime, default=utcnow)
    alert = db.relationship("Alert", back_populates="actions")
    user = db.relationship("User")


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey("audio_records.id"), nullable=False, index=True)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    decision = db.Column(db.String(16), nullable=False)  # confirmed, corrected, false_alarm
    original_category = db.Column(db.String(40))
    reviewed_category = db.Column(db.String(40))
    reviewed_severity = db.Column(db.String(16))
    is_override = db.Column(db.Boolean, default=False)
    comment = db.Column(db.Text, default="")
    recommended_action = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=utcnow)
    record = db.relationship("AudioRecord", back_populates="reviews")
    reviewer = db.relationship("User")


class ModelVersion(db.Model):
    __tablename__ = "model_versions"
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(10), nullable=False)  # python, gtm
    version = db.Column(db.String(120), nullable=False)
    algorithm = db.Column(db.String(80))
    source = db.Column(db.String(400))
    labels = db.Column(db.JSON)
    metrics = db.Column(db.JSON)
    is_active = db.Column(db.Boolean, default=False)
    registered_at = db.Column(db.DateTime, default=utcnow)
    registered_by = db.Column(db.String(40))
    notes = db.Column(db.Text, default="")
    __table_args__ = (db.UniqueConstraint("model_type", "version"),)


class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    username = db.Column(db.String(40))
    action = db.Column(db.String(40), nullable=False, index=True)
    target_type = db.Column(db.String(30))
    target_id = db.Column(db.String(40))
    details = db.Column(db.JSON)
    ip_address = db.Column(db.String(64))
    user_agent = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=utcnow, index=True)


class SystemAlert(db.Model):
    __tablename__ = "system_alerts"
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(40), nullable=False, index=True)
    severity = db.Column(db.String(16), default="Medium")
    message = db.Column(db.String(300), nullable=False)
    details = db.Column(db.JSON)
    count = db.Column(db.Integer, default=1)
    resolved = db.Column(db.Boolean, default=False, index=True)
    resolved_by = db.Column(db.String(40))
    resolved_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=utcnow)
    last_seen_at = db.Column(db.DateTime, default=utcnow)


class AppSetting(db.Model):
    __tablename__ = "app_settings"
    key = db.Column(db.String(60), primary_key=True)
    value = db.Column(db.JSON)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
    updated_by = db.Column(db.String(40))


class IdCounter(db.Model):
    __tablename__ = "id_counters"
    key = db.Column(db.String(30), primary_key=True)
    value = db.Column(db.Integer, default=0, nullable=False)
