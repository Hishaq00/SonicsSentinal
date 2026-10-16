"""Live microphone monitoring page and its JSON API."""
from __future__ import annotations

import json

from flask import Blueprint, abort, jsonify, render_template, request
from flask_login import current_user

from config import class_info
from python_models.inference import ModelNotAvailable, get_classifier

from ..extensions import db
from ..models import MicSession, utcnow
from ..security import permission_required
from ..services import audit, gtm_service, live_service
from ..services.settings_service import get_settings

bp = Blueprint("live", __name__)
MAX_WINDOW_BYTES = 3 * 1024 * 1024


def _own_session(code: str) -> MicSession:
    session = MicSession.query.filter_by(session_code=code).first()
    if session is None:
        abort(404)
    if session.user_id != current_user.id:
        abort(403)
    return session


@bp.route("/live")
@permission_required("live_monitor")
def monitor():
    settings = get_settings()
    clf = get_classifier()
    sessions = (MicSession.query.filter_by(user_id=current_user.id).order_by(MicSession.started_at.desc()).limit(5).all())
    return render_template(
        "live/monitor.html",
        segmentation=settings["segmentation"],
        live=settings["live"],
        python_ready=clf.available,
        python_version=clf.version if clf.available else None,
        classes=clf.classes if clf.available else [],
        class_info=class_info(),
        gtm=gtm_service.client_config(clf.classes if clf.available else None),
        consent=current_user.mic_consent_at is not None,
        sessions=sessions,
    )


@bp.route("/api/live/consent", methods=["POST"])
@permission_required("live_monitor")
def consent():
    if current_user.mic_consent_at is None:
        current_user.mic_consent_at = utcnow()
        audit.log("MIC_CONSENT", "user", current_user.user_code)
        db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/live/sessions", methods=["POST"])
@permission_required("live_monitor")
def start():
    if current_user.mic_consent_at is None:
        return jsonify(ok=False, error="Please read and accept the microphone notice first."), 400
    if not get_classifier().available:
        return jsonify(ok=False, error="The Python model is not available. Ask an administrator to train it."), 503
    data = request.get_json(silent=True) or {}
    # Close sessions the browser left open (tab closed without stopping).
    for old in MicSession.query.filter_by(user_id=current_user.id, ended_at=None).all():
        live_service.stop_session(old, "disconnected")
    session = live_service.start_session(current_user, float(data.get("window") or 0), float(data.get("hop") or 0),
                                         str(data.get("device") or ""), str(data.get("location") or ""))
    return jsonify(ok=True, session=session.session_code, window=session.window_seconds, hop=session.hop_seconds), 201


@bp.route("/api/live/sessions/<code>/status", methods=["POST"])
@permission_required("live_monitor")
def status(code):
    session = _own_session(code)
    if session.ended_at:
        return jsonify(ok=False, error="Session already ended."), 409
    try:
        live_service.set_status(session, (request.get_json(silent=True) or {}).get("status", ""))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, status=session.status)


@bp.route("/api/live/sessions/<code>/stop", methods=["POST"])
@permission_required("live_monitor")
def stop(code):
    session = _own_session(code)
    reason = (request.get_json(silent=True) or {}).get("reason", "stopped")
    live_service.stop_session(session, reason if reason in ("stopped", "disconnected") else "stopped")
    return jsonify(ok=True, windows=session.windows_processed, detections=session.detections, alerts=session.alerts)


@bp.route("/api/live/sessions/<code>/windows", methods=["POST"])
@permission_required("live_monitor")
def window(code):
    session = _own_session(code)
    if session.ended_at:
        return jsonify(ok=False, error="Session already ended."), 409
    audio = request.files.get("audio")
    if audio is None:
        return jsonify(ok=False, error="No audio window received."), 400
    data = audio.read(MAX_WINDOW_BYTES + 1)
    if len(data) > MAX_WINDOW_BYTES:
        return jsonify(ok=False, error="Window too large."), 413
    try:
        meta = json.loads(request.form.get("meta") or "{}")
    except json.JSONDecodeError:
        return jsonify(ok=False, error="Invalid window metadata."), 400
    try:
        result = live_service.process_window(session, data, meta.get("gtm"), int(meta.get("window_index", 0)),
                                             meta.get("client_time"))
    except ModelNotAvailable as exc:
        db.session.rollback()
        return jsonify(ok=False, error=str(exc)), 503
    except ValueError as exc:
        db.session.rollback()
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, **result)
