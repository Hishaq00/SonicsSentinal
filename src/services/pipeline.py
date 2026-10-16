"""Upload analysis pipeline.

analyze_file()  - runs on upload (server side):
    validate -> store securely -> metadata -> SHA-256 duplicate check ->
    fingerprint near-duplicate check -> audio quality -> waveform/spectrogram ->
    pre-processing -> fixed-duration segmentation -> Python classification.
The browser then runs the GTM model on the *same* pre-processed segments
(gtm_job) and posts its scores (receive_gtm).
finalize()      - combines both models with the decision engine, stores the
    final result and raises alerts.
If GTM is disabled or not configured the record is finalised immediately as a
single-model result.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from werkzeug.utils import secure_filename

from alert_rules import ALERT_GENERATED, SegmentInput, decide_recording, rules_by_category
from audio_preprocessing import PreprocessConfig, analyze_quality, preprocess_audio, segment_audio, write_wav
from audio_preprocessing.fingerprint import compute_fingerprint
from audio_preprocessing.loader import sha256_file
from audio_preprocessing.validation import validate_audio_file
from python_models.inference import ModelNotAvailable, get_classifier

from ..extensions import db
from ..models import AudioRecord, Segment, utcnow
from ..utils import next_code
from . import alerts_service, audit, duplicates, gtm_service, monitoring, storage, visuals
from .settings_service import get_settings

SILENT_SEGMENT_DBFS = -60.0  # after peak normalisation
EVENT_WITHIN_DB = 15.0


class UploadRejected(Exception):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ProcessingError(Exception):
    pass


def _display_name(name: str) -> str:
    cleaned = secure_filename(name) or "audio"
    return cleaned[:200]


def analyze_file(path: Path, original_name: str, user, *, source: str = "upload", batch=None,
                 evaluation_run=None, dataset_row: dict | None = None, keep_source: bool = False) -> AudioRecord:
    t0 = time.perf_counter()
    settings = get_settings()
    validation = validate_audio_file(path, original_name, settings["validation"], settings["quality"]["silence_dbfs"])
    if not validation.ok:
        audit.log("UPLOAD_REJECTED", "file", _display_name(original_name), {"errors": validation.errors}, user=user)
        if source in ("upload", "batch"):
            monitoring.on_failed_upload(user)
        db.session.commit()
        raise UploadRejected(" ".join(validation.errors), validation.to_dict())

    decoded = validation.decoded
    classifier = get_classifier()
    try:
        classes = classifier.classes
        py_version = classifier.version
    except ModelNotAvailable as exc:
        monitoring.on_model_failure("Python", str(exc))
        db.session.commit()
        raise ProcessingError(str(exc))

    audio_id = next_code("EV" if source == "evaluation" else "SS")
    folder = storage.record_dir(audio_id)
    ext = Path(original_name).suffix.lower() or ".wav"
    stored = folder / f"original{ext}"
    if keep_source:
        shutil.copyfile(path, stored)
    else:
        shutil.move(str(path), stored)

    record = AudioRecord(
        audio_id=audio_id,
        user_id=user.id,
        source=source,
        batch_id=batch.id if batch else None,
        evaluation_run_id=evaluation_run.id if evaluation_run else None,
        is_evaluation=source == "evaluation",
        dataset_audio_id=(dataset_row or {}).get("audio_id"),
        actual_class=(dataset_row or {}).get("class_label"),
        original_filename=_display_name(original_name),
        stored_path=storage.rel(stored),
        format=decoded.format,
        codec=decoded.codec,
        file_size=stored.stat().st_size,
        duration=round(decoded.duration, 3),
        sample_rate=decoded.sample_rate,
        channels=decoded.channels,
        bit_depth=decoded.bit_depth,
        sha256=sha256_file(stored),
        validation=validation.to_dict(),
        metadata_extra={"decoder": decoded.decoder, "warnings": decoded.warnings,
                        "expected_frames": decoded.expected_frames, "decoded_frames": decoded.frames},
        status="Uploaded",
        python_model_version=py_version,
    )
    db.session.add(record)
    db.session.flush()

    mono = decoded.mono()
    sr = decoded.sample_rate

    # Duplicate and near-duplicate detection (user uploads only).
    if source in ("upload", "batch"):
        fp, desc = compute_fingerprint(mono, sr)
        record.fingerprint, record.fp_descriptor = fp, desc
        dup = duplicates.find_exact(record.sha256, record.id)
        if dup:
            record.duplicate_of_id = dup.id
            monitoring.on_duplicate(record, dup)
        else:
            near, ber = duplicates.find_near(fp, desc, record.id)
            if near:
                record.near_duplicate_of_id = near.id
                record.near_duplicate_ber = round(ber, 4)
                monitoring.on_duplicate(record, near, near=True)

    quality = analyze_quality(mono, sr, settings["quality"], decoded.warnings, decoded.expected_frames,
                              decoded.frames, decoded.bit_depth)
    record.quality_label, record.quality_score, record.quality = quality["label"], quality["score"], quality

    wave_path, spec_path = visuals.write_visuals(mono, sr, folder)
    record.waveform_path, record.spectrogram_path = storage.rel(wave_path), storage.rel(spec_path)

    # Pre-processing (identical to training) and segmentation.
    pre_cfg = PreprocessConfig.from_settings(settings)
    pre = preprocess_audio(decoded.samples, sr, pre_cfg)
    steps = list(pre.steps)
    if decoded.decoder == "ffmpeg" or decoded.format not in ("WAV", "WAVEX"):
        steps.append({"step": "Audio-format conversion",
                      "detail": f"{decoded.format}/{decoded.codec} decoded with {decoded.decoder} and stored as "
                                f"16-bit PCM WAV at {pre.sr} Hz for analysis."})
    processed = folder / "processed.wav"
    write_wav(processed, pre.y, pre.sr)
    record.processed_path = storage.rel(processed)

    seg_s = float(settings["segmentation"]["segment_seconds"])
    hop_s = float(settings["segmentation"]["segment_hop_seconds"])
    segs = segment_audio(pre.y, pre.sr, seg_s, hop_s)
    steps.append({"step": "Fixed-duration segmentation",
                  "detail": f"{len(segs)} segment(s) of {seg_s:g} s every {hop_s:g} s"
                            + (" (short clip centred and zero-padded)." if segs and segs[0].padded else ".")})
    record.preprocessing = {"steps": steps, "trim_offset": round(pre.trim_offset, 4), "sample_rate": pre.sr,
                            "duration": round(pre.duration, 3), "config": pre_cfg.to_dict(),
                            "segment_seconds": seg_s, "segment_hop_seconds": hop_s}

    loudest = max(s.rms_dbfs for s in segs)
    active = [s for s in segs if s.rms_dbfs > SILENT_SEGMENT_DBFS]
    try:
        probs = classifier.predict_segments([s.audio for s in active])
    except Exception as exc:  # model crash must be visible to admins
        monitoring.on_model_failure("Python", str(exc))
        record.error_message = f"Python model failure: {exc}"
        db.session.commit()
        raise ProcessingError(f"Python model failure: {exc}")
    prob_by_index = {s.index: p for s, p in zip(active, probs)}

    offset = pre.trim_offset
    for s in segs:
        o0 = offset + s.start
        o1 = min(offset + s.end, decoded.duration)
        chunk = mono[int(o0 * sr):max(int(o1 * sr), int(o0 * sr) + 1)]
        q = analyze_quality(chunk, sr, settings["quality"], compact=True)
        scores = prob_by_index.get(s.index)
        top = max(scores, key=scores.get) if scores else None
        db.session.add(Segment(
            record_id=record.id, index=s.index, start_time=round(o0, 3), end_time=round(o1, 3),
            proc_start=round(s.start, 3), proc_end=round(s.end, 3), rms_dbfs=round(s.rms_dbfs, 1),
            silent=scores is None, is_event=scores is not None and s.rms_dbfs >= loudest - EVENT_WITHIN_DB,
            quality_label=q["label"], quality={"score": q["score"], "metrics": q["metrics"],
                                               "issues": [i["label"] for i in q["issues"]]},
            python_scores=scores, python_top=top, python_confidence=scores[top] if scores else None,
        ))
    record.status = "Classified"
    record.gtm_status = "pending" if gtm_service.is_enabled() else "unavailable"
    record.processing_ms = int((time.perf_counter() - t0) * 1000)
    db.session.flush()
    audit.log("UPLOAD", "record", audio_id, {"file": record.original_filename, "source": source,
                                             "duration": record.duration, "quality": record.quality_label,
                                             "duplicate_of": record.duplicate_of.audio_id if record.duplicate_of else None},
              user=user)
    if record.gtm_status == "unavailable":
        finalize(record, None, None, "unavailable")
    db.session.commit()
    return record


def gtm_job(record: AudioRecord) -> dict:
    """What the browser needs to run GTM on the same segments (no Python results included)."""
    from flask import url_for

    return {
        "audio_id": record.audio_id,
        "status": record.gtm_status,
        "audio_url": url_for("analysis.record_audio", audio_id=record.audio_id, kind="processed"),
        "segments": [{"index": s.index, "start": s.proc_start, "end": s.proc_end} for s in record.segments if not s.silent],
        "submit_url": url_for("analysis.submit_gtm", audio_id=record.audio_id),
        "gtm": gtm_service.client_config(get_classifier().classes if get_classifier().available else None),
    }


def receive_gtm(record: AudioRecord, payload: dict) -> AudioRecord:
    """Validate the browser's GTM output and finalise the record."""
    if record.gtm_status not in ("pending",):
        raise ValueError(f"GTM result already received (status {record.gtm_status}).")
    expected = get_classifier().classes
    if payload.get("error"):
        monitoring.on_model_failure("GTM", str(payload["error"]))
        finalize(record, None, None, "failed", note=f"GTM error: {str(payload['error'])[:200]}")
        return record
    labels = payload.get("labels") or []
    check = gtm_service.label_check(labels, expected)
    if not check["ok"]:
        msg = f"GTM class names do not match the Python model (missing {check['missing']}, extra {check['extra']})."
        monitoring.on_model_failure("GTM", msg)
        finalize(record, None, None, "failed", note=msg)
        return record
    by_index: dict[int, dict] = {}
    windows: dict[int, float] = {}
    for item in payload.get("segments") or []:
        idx = int(item.get("index"))
        by_index[idx] = gtm_service.parse_scores(item.get("scores"), expected)
        if item.get("window_start") is not None:
            windows[idx] = float(item["window_start"])
    needed = {s.index for s in record.segments if not s.silent}
    if not needed.issubset(by_index):
        raise ValueError(f"GTM results missing for segments {sorted(needed - set(by_index))}.")
    version = gtm_service.clean_version(payload.get("model_version"))
    finalize(record, by_index, version, "completed", windows)
    return record


def finalize(record: AudioRecord, gtm_by_index: dict[int, dict] | None, gtm_version: str | None,
             gtm_status: str, windows: dict[int, float] | None = None, note: str | None = None) -> AudioRecord:
    settings = get_settings()
    rules = rules_by_category()
    det = settings["detection"]
    segments = list(record.segments)
    inputs = []
    for s in segments:
        gtm_scores = (gtm_by_index or {}).get(s.index)
        s.gtm_scores = gtm_scores
        if gtm_scores:
            s.gtm_top = max(gtm_scores, key=gtm_scores.get)
            s.gtm_confidence = gtm_scores[s.gtm_top]
            s.gtm_window_start = (windows or {}).get(s.index)
        inputs.append(SegmentInput(s.index, s.start_time, s.end_time, s.python_scores, gtm_scores,
                                   s.quality_label or record.quality_label, (s.quality or {}).get("metrics", {}),
                                   bool(s.is_event)))
    result = decide_recording(inputs, rules, det, record.quality_label, (record.quality or {}).get("metrics", {}),
                              record.python_model_version, gtm_version)
    for s, d in zip(segments, result["segment_decisions"]):
        if d is None:
            continue
        s.final_category, s.final_confidence = d.category, round(d.confidence, 4)
        s.consistency, s.severity, s.alert_status, s.needs_review = d.consistency, d.severity, d.alert_status, d.needs_review
        s.decision = _compact(d)
    d = result["decision"]
    record.gtm_status = gtm_status
    record.gtm_model_version = gtm_version
    if d is None:
        record.status = "Classified"
        record.final_category = "Silence"
        record.error_message = "No classifiable audio in any segment."
        db.session.flush()
        return record
    apply_decision(record, d, result.get("events"))
    if note:
        record.error_message = note
    record.python_scores = _round_scores(result.get("python_recording_scores"))
    record.gtm_scores = _round_scores(result.get("gtm_recording_scores"))
    db.session.flush()
    if not record.is_evaluation:
        if d.alert_status == ALERT_GENERATED:
            alerts_service.create_alert(record, d)
        for extra in result["extra_alerts"]:
            alerts_service.create_alert(record, extra["decision"], segment_index=extra["segment_index"])
        audit.log("PREDICTION", "record", record.audio_id, {
            "category": d.category, "confidence": round(d.confidence, 3), "python": d.python.top,
            "gtm": d.gtm.top if d.gtm else None, "consistency": d.consistency, "severity": d.severity,
            "alert": d.alert_status, "review": d.needs_review,
            "python_version": record.python_model_version, "gtm_version": gtm_version}, user=record.user)
        monitoring.after_prediction()
    db.session.flush()
    return record


def apply_decision(record: AudioRecord, d, events=None) -> None:
    record.python_top, record.python_confidence = d.python.top, round(d.python.confidence, 4)
    record.python_margin = round(d.python.margin, 4)
    record.python_scores = _round_scores(d.python.scores)
    if d.gtm:
        record.gtm_top, record.gtm_confidence = d.gtm.top, round(d.gtm.confidence, 4)
        record.gtm_margin = round(d.gtm.margin, 4)
        record.gtm_scores = _round_scores(d.gtm.scores)
    record.class_match = d.class_match
    record.confidence_difference = None if d.confidence_difference is None else round(d.confidence_difference, 4)
    record.margin_difference = None if d.margin_difference is None else round(d.margin_difference, 4)
    record.consistency = d.consistency
    record.final_category = d.category
    record.final_confidence = round(d.confidence, 4)
    record.confidence_level = d.confidence_level
    record.severity = d.severity
    record.alert_status = d.alert_status
    record.needs_review = d.needs_review
    record.review_reasons = d.review_reasons
    record.uncertain_reasons = d.uncertain_reasons
    record.overlapping = d.overlapping
    record.overlap_classes = d.overlap_classes
    record.recommended_action = d.recommended_action
    record.decision = _compact(d)
    if events is not None:
        record.events = events
    record.status = d.event_status
    record.classified_at = utcnow()


def _round_scores(scores):
    if not scores:
        return None
    return {k: round(float(v), 4) for k, v in scores.items()}


def _compact(d) -> dict:
    return {
        "category": d.category,
        "confidence": round(d.confidence, 4),
        "confidence_level": d.confidence_level,
        "consistency": d.consistency,
        "class_match": d.class_match,
        "confidence_difference": None if d.confidence_difference is None else round(d.confidence_difference, 4),
        "margin_difference": None if d.margin_difference is None else round(d.margin_difference, 4),
        "python": {"top": d.python.top, "confidence": round(d.python.confidence, 4), "second": d.python.second,
                   "margin": round(d.python.margin, 4), "top3": d.python.top3},
        "gtm": None if not d.gtm else {"top": d.gtm.top, "confidence": round(d.gtm.confidence, 4),
                                       "second": d.gtm.second, "margin": round(d.gtm.margin, 4), "top3": d.gtm.top3},
        "fused_scores": _round_scores(d.fused_scores),
        "fused_margin": round(d.fused_margin, 4),
        "smoothing_windows": d.smoothing_windows,
        "window_scores": _round_scores(d.window_scores) if d.smoothing_windows > 1 else None,
        "overlapping": d.overlapping,
        "overlap_classes": d.overlap_classes,
        "uncertain": d.uncertain,
        "uncertain_reasons": d.uncertain_reasons,
        "severity": d.severity,
        "escalated": d.escalated,
        "escalation_note": d.escalation_note,
        "alert_enabled": d.alert_enabled,
        "alert_status": d.alert_status,
        "alert_conditions": d.alert_conditions,
        "consecutive_count": d.consecutive_count,
        "repeat_count": d.repeat_count,
        "needs_review": d.needs_review,
        "review_reasons": d.review_reasons,
        "recommended_action": d.recommended_action,
        "event_status": d.event_status,
    }


def finalize_stale(minutes: int = 5) -> int:
    """Records whose GTM result never arrived (browser closed) are finalised as single-model results."""
    cutoff = utcnow() - timedelta(minutes=minutes)
    stale = AudioRecord.query.filter(AudioRecord.gtm_status == "pending", AudioRecord.created_at < cutoff).limit(50).all()
    for rec in stale:
        finalize(rec, None, None, "timeout", note="GTM result was not received from the browser in time; Python-only result.")
    if stale:
        db.session.commit()
    return len(stale)


def record_summary(record: AudioRecord, include_scores: bool = True) -> dict:
    """JSON view of a record (used by the upload page, APIs and tests)."""
    from flask import url_for

    data = {
        "audio_id": record.audio_id,
        "url": url_for("analysis.record_detail", audio_id=record.audio_id),
        "filename": record.original_filename,
        "status": record.status,
        "gtm_status": record.gtm_status,
        "duration": record.duration,
        "quality": record.quality_label,
        "category": record.effective_category,
        "confidence": record.final_confidence,
        "confidence_level": record.confidence_level,
        "severity": record.effective_severity,
        "consistency": record.consistency,
        "alert_status": record.alert_status,
        "needs_review": record.needs_review,
        "review_reasons": record.review_reasons or [],
        "recommended_action": record.recommended_action,
        "duplicate_of": record.duplicate_of.audio_id if record.duplicate_of else None,
        "near_duplicate_of": record.near_duplicate_of.audio_id if record.near_duplicate_of else None,
        "processing_ms": record.processing_ms,
    }
    if include_scores and record.gtm_status != "pending":
        data.update({
            "python": {"top": record.python_top, "confidence": record.python_confidence, "scores": record.python_scores},
            "gtm": {"top": record.gtm_top, "confidence": record.gtm_confidence, "scores": record.gtm_scores},
            "confidence_difference": record.confidence_difference,
            "class_match": record.class_match,
        })
    return data


def dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
