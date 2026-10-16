"""Audio upload (single and batch), analysis results, audio preview, GTM hand-off,
manual review, closing events and the PDF report."""
from __future__ import annotations

import io
import json
import os
import uuid

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from alert_rules import rules_by_category
from config import class_info, class_names
from python_models.inference import get_classifier

from ..extensions import db
from ..models import AudioRecord, Batch
from ..security import can, can_view_record, permission_required
from ..services import audit, gtm_service, pipeline, reports, review_service, storage
from ..services.settings_service import get_settings
from ..utils import next_code

bp = Blueprint("analysis", __name__)


def _get_record(audio_id: str) -> AudioRecord:
    record = AudioRecord.query.filter_by(audio_id=audio_id).first()
    if record is None:
        abort(404)
    if not can_view_record(current_user, record):
        abort(403)
    return record


@bp.route("/analyze")
@permission_required("upload")
def analyze():
    settings = get_settings()
    clf = get_classifier()
    return render_template(
        "analysis/analyze.html",
        validation=settings["validation"],
        segmentation=settings["segmentation"],
        can_batch=can(current_user, "batch_upload"),
        python_ready=clf.available,
        python_version=clf.version if clf.available else None,
        gtm=gtm_service.client_config(clf.classes if clf.available else None),
        recent=(AudioRecord.query.filter_by(user_id=current_user.id).filter(AudioRecord.source.in_(("upload", "batch")))
                .order_by(AudioRecord.created_at.desc()).limit(6).all()),
    )


@bp.route("/api/batches", methods=["POST"])
@permission_required("batch_upload")
def create_batch():
    count = request.get_json(silent=True, force=True) or {}
    n = int(count.get("files", 0))
    limit = int(get_settings()["validation"]["max_batch_files"])
    if n < 1 or n > limit:
        return jsonify(ok=False, error=f"A batch can contain 1-{limit} files."), 400
    batch = Batch(batch_code=next_code("BAT"), user_id=current_user.id, file_count=n)
    db.session.add(batch)
    db.session.commit()
    return jsonify(ok=True, batch=batch.batch_code, url=url_for("analysis.batch_detail", code=batch.batch_code)), 201


@bp.route("/api/uploads", methods=["POST"])
@permission_required("upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(ok=False, error="No file was sent."), 400
    batch = None
    if request.form.get("batch"):
        if not can(current_user, "batch_upload"):
            abort(403)
        batch = Batch.query.filter_by(batch_code=request.form["batch"], user_id=current_user.id).first()
        if batch is None:
            return jsonify(ok=False, error="Unknown batch."), 400
    ext = os.path.splitext(file.filename)[1].lower()[:8]
    tmp = storage.tmp_dir() / f"{uuid.uuid4().hex}{secure_filename(ext) if ext else ''}"
    file.save(tmp)
    try:
        record = pipeline.analyze_file(tmp, file.filename, current_user, source="batch" if batch else "upload", batch=batch)
    except pipeline.UploadRejected as exc:
        if batch:
            batch.failed = (batch.failed or 0) + 1
            db.session.commit()
        return jsonify(ok=False, error=str(exc), checks=exc.details.get("checks", [])), 400
    except pipeline.ProcessingError as exc:
        db.session.rollback()
        return jsonify(ok=False, error=str(exc)), 503
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    if batch:
        batch.processed = (batch.processed or 0) + 1
        db.session.commit()
    return jsonify(ok=True, record=pipeline.record_summary(record, include_scores=False),
                   gtm_job=pipeline.gtm_job(record) if record.gtm_status == "pending" else None), 201


@bp.route("/api/records/<audio_id>")
@login_required
def record_json(audio_id):
    record = _get_record(audio_id)
    data = pipeline.record_summary(record)
    data["segments"] = [{"index": s.index, "start": s.start_time, "end": s.end_time, "category": s.final_category,
                         "confidence": s.final_confidence, "python": s.python_top, "gtm": s.gtm_top,
                         "silent": s.silent} for s in record.segments]
    data["events"] = record.events or []
    return jsonify(data)


@bp.route("/api/records/<audio_id>/gtm-job")
@login_required
def record_gtm_job(audio_id):
    record = _get_record(audio_id)
    return jsonify(pipeline.gtm_job(record))


@bp.route("/api/records/<audio_id>/gtm", methods=["POST"])
@login_required
def submit_gtm(audio_id):
    record = _get_record(audio_id)
    if record.user_id != current_user.id and not (record.is_evaluation and can(current_user, "evaluation")):
        abort(403)
    payload = request.get_json(silent=True) or {}
    try:
        pipeline.receive_gtm(record, payload)
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, record=pipeline.record_summary(record))


@bp.route("/records/<audio_id>")
@login_required
def record_detail(audio_id):
    record = _get_record(audio_id)
    rules = rules_by_category()
    waveform = None
    wpath = storage.resolve(record.waveform_path)
    if wpath:
        waveform = json.loads(wpath.read_text(encoding="utf-8"))
    siblings = []
    if record.session_id:
        siblings = (AudioRecord.query.filter(AudioRecord.session_id == record.session_id,
                                             AudioRecord.final_category != "Background Noise")
                    .order_by(AudioRecord.window_index.desc()).limit(12).all())
    clf = get_classifier()
    return render_template(
        "analysis/record.html",
        record=record,
        waveform=waveform,
        rules=rules,
        rule=rules.get(record.final_category or "", {}),
        classes=class_names(),
        class_info=class_info(),
        review_categories=review_service.review_categories(),
        can_review=can(current_user, "review"),
        can_close=can(current_user, "close_event"),
        gtm=gtm_service.client_config(clf.classes if clf.available else None),
        siblings=siblings,
    )


@bp.route("/records/<audio_id>/audio/<kind>")
@login_required
def record_audio(audio_id, kind):
    record = _get_record(audio_id)
    if kind not in ("original", "processed"):
        abort(404)
    path = storage.resolve(record.stored_path if kind == "original" else record.processed_path)
    if path is None:
        abort(404)
    mimetypes = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
                 ".m4a": "audio/mp4"}
    return send_file(path, mimetype=mimetypes.get(path.suffix.lower(), "application/octet-stream"),
                     conditional=True, max_age=0)


@bp.route("/records/<audio_id>/spectrogram.png")
@login_required
def record_spectrogram(audio_id):
    record = _get_record(audio_id)
    path = storage.resolve(record.spectrogram_path)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/png", max_age=3600)


@bp.route("/records/<audio_id>/report.pdf")
@login_required
def record_report(audio_id):
    record = _get_record(audio_id)
    try:
        pdf = reports.record_pdf(record)
    except Exception as exc:  # report failure must show a clear message
        flash(f"The report could not be generated: {exc}", "error")
        return redirect(url_for("analysis.record_detail", audio_id=audio_id))
    audit.log("REPORT", "record", record.audio_id, {"format": "pdf"})
    db.session.commit()
    return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True,
                     download_name=f"SonicSentinel_{record.audio_id}.pdf")


@bp.route("/records/<audio_id>/review", methods=["POST"])
@permission_required("review")
def submit_review(audio_id):
    record = _get_record(audio_id)
    if record.gtm_status == "pending":
        flash("Wait until both models have finished before reviewing.", "error")
        return redirect(url_for("analysis.record_detail", audio_id=audio_id))
    try:
        review_service.submit_review(
            record, current_user, request.form.get("decision", ""), request.form.get("category"),
            request.form.get("comment", ""), request.form.get("recommended_action", ""),
            bool(request.form.get("override")))
        db.session.commit()
        flash("Review saved. The original model outputs are preserved.", "success")
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "error")
    nxt = request.form.get("next")
    if nxt == "queue":
        return redirect(url_for("review.queue"))
    return redirect(url_for("analysis.record_detail", audio_id=audio_id))


@bp.route("/records/<audio_id>/close", methods=["POST"])
@permission_required("close_event")
def close_record(audio_id):
    record = _get_record(audio_id)
    try:
        review_service.close_record(record, current_user, request.form.get("comment", ""))
        db.session.commit()
        flash(f"{record.audio_id} closed.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    return redirect(url_for("analysis.record_detail", audio_id=audio_id))


@bp.route("/batches/<code>")
@permission_required("batch_upload")
def batch_detail(code):
    batch = Batch.query.filter_by(batch_code=code).first_or_404()
    if batch.user_id != current_user.id and not can(current_user, "view_all_records"):
        abort(403)
    records = AudioRecord.query.filter_by(batch_id=batch.id).order_by(AudioRecord.id).all()
    counts: dict[str, int] = {}
    for r in records:
        counts[r.effective_category or "Pending"] = counts.get(r.effective_category or "Pending", 0) + 1
    return render_template("analysis/batch.html", batch=batch, records=records, counts=counts,
                           class_info=class_info())
