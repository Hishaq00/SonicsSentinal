"""Live microphone monitoring.

The browser captures the microphone continuously and sends one window every
``live_hop_seconds`` (default: 2 s windows every 1 s, i.e. 50 % overlap). For
each window the browser has already run the GTM model on the same audio, so the
request carries the raw window (16-bit WAV) plus GTM's class scores. The server
then validates, pre-processes, runs the Python model, compares the models,
applies the alert rules using the recent windows of this session (repeated
detection) and stores the result.

Because the windows overlap, the decision is not taken on one window alone: the
fused scores of the last ``detection.live_smoothing_windows`` windows are
averaged (``alert_rules.engine.smooth_live``). Silence, a pause, a gap in the
window indices or a long delay ends that stretch, so a result never mixes audio
from before an interruption.

Privacy: audio is only written to disk according to ``live.store_audio``
(default: only windows with a detection). Background-noise windows are kept
as predictions only.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from alert_rules import ALERT_GENERATED, ALERT_PENDING, Context, HistoryItem, decide, rules_by_category
from alert_rules.engine import BACKGROUND
from audio_preprocessing import PreprocessConfig, analyze_quality, decode_bytes, preprocess_audio, write_wav
from audio_preprocessing.validation import validate_live_window
from config import unknown_label
from python_models.inference import get_classifier

from ..extensions import db
from ..models import AudioRecord, MicSession, Segment, utcnow
from ..utils import next_code
from . import alerts_service, audit, gtm_service, monitoring, storage, visuals
from .pipeline import apply_decision
from .settings_service import get_settings


@dataclass
class _Pending:
    record_id: int
    category: str
    index: int


_lock = threading.Lock()
_history: dict[int, deque] = {}
_pending: dict[int, _Pending] = {}
_counts: dict[int, dict[str, int]] = {}
# Fused scores of the most recent windows per session, for the live smoothing in the decision
# engine: (window index, client time, scores). Cleared whenever the stream breaks.
_recent: dict[int, deque] = {}
MAX_SMOOTHING_GAP_S = 4.0  # a longer pause between two windows starts a new stretch of audio


def _recent_scores(session_id: int, window_index: int, now_s: float, count: int) -> list[dict]:
    """Fused scores of the windows immediately before this one (oldest first).

    Only windows from the same uninterrupted stretch count: consecutive window
    indices and no long pause, so a result never averages across a silence, a
    pause or a reconnect.
    """
    if count <= 1:
        return []
    with _lock:
        items = list(_recent.get(session_id, ()))
    out: list[dict] = []
    expected = window_index - 1
    previous_time = now_s
    for index, time_s, scores in reversed(items):
        if index != expected or not (0 <= previous_time - time_s <= MAX_SMOOTHING_GAP_S):
            break
        out.append(scores)
        expected -= 1
        previous_time = time_s
        if len(out) >= count - 1:
            break
    return list(reversed(out))


def start_session(user, window_s: float, hop_s: float, device_label: str = "", location: str = "") -> MicSession:
    s = get_settings()["segmentation"]
    window_s = float(min(3.0, max(1.0, window_s or s["live_window_seconds"])))
    hop_s = float(min(window_s, max(0.5, hop_s or s["live_hop_seconds"])))
    session = MicSession(session_code=next_code("MIC"), user_id=user.id, window_seconds=window_s,
                         hop_seconds=hop_s, device_label=(device_label or "")[:200], location=(location or "")[:120])
    db.session.add(session)
    db.session.flush()
    with _lock:
        _history[session.id] = deque(maxlen=120)
        _recent[session.id] = deque(maxlen=8)
        _pending.pop(session.id, None)
        _counts[session.id] = {}
    audit.log("MIC_START", "mic_session", session.session_code,
              {"window_s": window_s, "hop_s": hop_s, "device": session.device_label, "location": session.location})
    db.session.commit()
    return session


def set_status(session: MicSession, status: str) -> MicSession:
    if status not in ("active", "paused", "disconnected"):
        raise ValueError("invalid status")
    if session.status != status:
        audit.log("MIC_STATUS", "mic_session", session.session_code, {"from": session.status, "to": status})
        session.status = status
        if status != "active":  # a pause or a drop-out ends the current stretch of audio
            with _lock:
                _recent.pop(session.id, None)
        db.session.commit()
    return session


def stop_session(session: MicSession, reason: str = "stopped") -> MicSession:
    if session.ended_at is None:
        session.status = "stopped" if reason == "stopped" else reason
        session.ended_at = utcnow()
        with _lock:
            counts = _counts.pop(session.id, {})
            _history.pop(session.id, None)
            _recent.pop(session.id, None)
            pending = _pending.pop(session.id, None)
        if pending:
            _expire_pending(pending)
        audit.log("MIC_STOP", "mic_session", session.session_code, {
            "reason": reason, "windows": session.windows_processed, "silent": session.silent_windows,
            "detections": session.detections, "alerts": session.alerts, "categories": counts,
            "avg_latency_ms": round(session.avg_latency_ms or 0, 1)})
        db.session.commit()
    return session


def _expire_pending(p: _Pending) -> None:
    """A critical detection that was never confirmed becomes a possible false alarm."""
    rec = db.session.get(AudioRecord, p.record_id)
    if rec is None or rec.alert_status != ALERT_PENDING:
        return
    rules = rules_by_category()
    rule = rules.get(p.category, {})
    rec.alert_status = "Not Confirmed"
    if rule.get("manual_review_condition") == "on_uncertain_or_unconfirmed":
        rec.needs_review = True
        rec.review_reasons = list(dict.fromkeys((rec.review_reasons or []) + ["Possible false alarm"]))
        rec.status = "Manual Review"
        rec.recommended_action = ("Not confirmed in the next window. Listen to the segment and confirm or "
                                  "reject the detection.")


def process_window(session: MicSession, wav_bytes: bytes, gtm_payload: dict | None, window_index: int,
                   client_time: float | None = None) -> dict:
    t0 = time.perf_counter()
    settings = get_settings()
    det = settings["detection"]
    rules = rules_by_category()
    classifier = get_classifier()
    classes = classifier.classes

    decoded = decode_bytes(wav_bytes, ".wav")
    live_silence_threshold = float(settings["quality"].get("live_silence_dbfs", -60.0))
    validation = validate_live_window(decoded, session.window_seconds, live_silence_threshold)
    if not validation.ok:
        raise ValueError(" ".join(validation.errors))
    mono = decoded.mono()
    sr = decoded.sample_rate
    session.windows_processed = (session.windows_processed or 0) + 1
    session.last_window_at = utcnow()
    if session.status != "active":
        session.status = "active"
    level = float(20 * np.log10(max(np.sqrt(np.mean(mono ** 2)), 1e-10)))

    # Only treat as silence if there is genuinely no audio signal (< live_silence_threshold or pure zeros)
    silent = (np.max(np.abs(mono)) < 1e-4 or level < live_silence_threshold)
    if silent:
        session.silent_windows = (session.silent_windows or 0) + 1
        with _lock:
            _recent.pop(session.id, None)  # silence ends the stretch the result is smoothed over
            pending = _pending.pop(session.id, None)
        if pending:
            _expire_pending(pending)
        db.session.commit()
        return {"window_index": window_index, "status": "silence", "level_dbfs": round(level, 1),
                "latency_ms": int((time.perf_counter() - t0) * 1000)}

    quality_cfg = dict(settings["quality"])
    quality_cfg["silence_dbfs"] = live_silence_threshold
    quality = analyze_quality(mono, sr, quality_cfg, compact=True)
    pre = preprocess_audio(decoded.samples, sr, PreprocessConfig.from_settings(settings), trim=False)
    python_scores = classifier.predict_segments([pre.y])[0]

    gtm_scores, gtm_version, gtm_note = None, None, None
    if gtm_payload and gtm_payload.get("scores"):
        labels = gtm_payload.get("labels") or list(gtm_payload["scores"])
        check = gtm_service.label_check(labels, classes)
        if check["ok"]:
            gtm_scores = gtm_service.parse_scores(gtm_payload["scores"], classes)
            gtm_version = gtm_service.clean_version(gtm_payload.get("model_version"))
        else:
            gtm_note = f"GTM labels mismatch: missing {check['missing']}"
    elif gtm_payload and gtm_payload.get("error"):
        gtm_note = f"GTM error: {str(gtm_payload['error'])[:150]}"

    now_s = float(client_time) if client_time else time.time()
    with _lock:
        history = list(_history.setdefault(session.id, deque(maxlen=120)))
    ctx = Context(source="live", time_s=now_s, index=window_index, history=history,
                  recent_scores=_recent_scores(session.id, window_index, now_s,
                                               int(det.get("live_smoothing_windows", 1) or 1)))
    d = decide(python_scores, gtm_scores, quality["label"], quality["metrics"], rules, det, ctx,
               classifier.version, gtm_version)
    with _lock:
        _recent.setdefault(session.id, deque(maxlen=8)).append((window_index, now_s, d.window_scores))

    # Decide whether to keep the audio (privacy setting).
    store_mode = settings["live"].get("store_audio", "events")
    is_event = d.category not in (BACKGROUND,) or d.alert_status != "No Alert" or d.needs_review
    keep_audio = store_mode == "all" or (store_mode == "events" and is_event)

    record = AudioRecord(
        audio_id=next_code("LV"), user_id=session.user_id, source="live", session_id=session.id,
        window_index=window_index, original_filename=f"{session.session_code} window {window_index}",
        format="WAV", codec=decoded.codec, file_size=len(wav_bytes), duration=round(decoded.duration, 3),
        sample_rate=sr, channels=decoded.channels, bit_depth=decoded.bit_depth,
        validation=validation.to_dict(), quality_label=quality["label"], quality_score=quality["score"],
        quality=quality, python_model_version=classifier.version, gtm_model_version=gtm_version,
        gtm_status="completed" if gtm_scores else ("failed" if gtm_note else "unavailable"),
        error_message=gtm_note,
        preprocessing={"steps": pre.steps, "trim_offset": 0.0, "sample_rate": pre.sr,
                       "duration": round(pre.duration, 3), "segment_seconds": session.window_seconds,
                       "segment_hop_seconds": session.hop_seconds},
    )
    db.session.add(record)
    db.session.flush()
    if keep_audio:
        folder = storage.record_dir(record.audio_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "original.wav").write_bytes(wav_bytes)
        write_wav(folder / "processed.wav", pre.y, pre.sr)
        record.stored_path = storage.rel(folder / "original.wav")
        record.processed_path = storage.rel(folder / "processed.wav")
        record.waveform_path = storage.rel(folder / "waveform.json")
        record.spectrogram_path = storage.rel(folder / "spectrogram.png")
        threading.Thread(target=visuals.write_visuals, args=(mono, sr, folder), daemon=True).start()
    apply_decision(record, d, [{"category": d.category, "start": 0.0, "end": record.duration,
                                "segments": [0], "confidence": round(d.confidence, 4), "severity": d.severity,
                                "alert": d.alert_status == ALERT_GENERATED}])
    db.session.add(Segment(
        record_id=record.id, index=0, start_time=0.0, end_time=record.duration, proc_start=0.0,
        proc_end=round(pre.duration, 3), rms_dbfs=round(level, 1), is_event=True, silent=False,
        quality_label=quality["label"], quality={"score": quality["score"], "metrics": quality["metrics"],
                                                 "issues": [i["label"] for i in quality["issues"]]},
        python_scores=record.python_scores, python_top=d.python.top, python_confidence=round(d.python.confidence, 4),
        gtm_scores=record.gtm_scores, gtm_top=d.gtm.top if d.gtm else None,
        gtm_confidence=round(d.gtm.confidence, 4) if d.gtm else None,
        gtm_window_start=(gtm_payload or {}).get("window_start"),
        final_category=d.category, final_confidence=round(d.confidence, 4), consistency=d.consistency,
        severity=d.severity, alert_status=d.alert_status, needs_review=d.needs_review, decision=record.decision,
    ))

    # Repeated-detection bookkeeping.
    with _lock:
        _history.setdefault(session.id, deque(maxlen=120)).append(
            HistoryItem(window_index, now_s, d.category, d.confidence))
        counts = _counts.setdefault(session.id, {})
        counts[d.category] = counts.get(d.category, 0) + 1
        previous = _pending.pop(session.id, None)
        if d.alert_status == ALERT_PENDING:
            _pending[session.id] = _Pending(record.id, d.category, window_index)
    if previous and not (previous.category == d.category and d.alert_status == ALERT_GENERATED):
        _expire_pending(previous)
    if previous and previous.category == d.category and d.alert_status == ALERT_GENERATED:
        prev = db.session.get(AudioRecord, previous.record_id)
        if prev is not None:
            prev.alert_status = "Confirmed by next window"

    new_alert = None
    if d.alert_status == ALERT_GENERATED:
        before = session.alerts or 0
        alert = alerts_service.create_alert(record, d, session=session,
                                            cooldown_seconds=int(settings["live"].get("alert_cooldown_seconds", 15)))
        if alert.record_id == record.id:
            session.alerts = before + 1
            new_alert = alert
        else:
            new_alert = alert
    if d.category not in (BACKGROUND, unknown_label()):
        session.detections = (session.detections or 0) + 1
        audit.log("PREDICTION", "record", record.audio_id, {
            "session": session.session_code, "window": window_index, "category": d.category,
            "confidence": round(d.confidence, 3), "python": d.python.top, "gtm": d.gtm.top if d.gtm else None,
            "consistency": d.consistency, "alert": d.alert_status,
            "python_version": classifier.version, "gtm_version": gtm_version})

    latency = (time.perf_counter() - t0) * 1000
    n = session.windows_processed or 1
    session.avg_latency_ms = ((session.avg_latency_ms or 0) * (n - 1) + latency) / n
    record.processing_ms = int(latency)
    if n % 10 == 0:
        monitoring.after_prediction()
    db.session.commit()

    return {
        "window_index": window_index,
        "status": "classified",
        "audio_id": record.audio_id,
        "record_url": f"/records/{record.audio_id}",
        "level_dbfs": round(level, 1),
        "quality": quality["label"],
        "quality_issues": [i["label"] for i in quality["issues"]],
        "python": {"top": d.python.top, "confidence": round(d.python.confidence, 4), "top3": d.python.top3},
        "gtm": None if not d.gtm else {"top": d.gtm.top, "confidence": round(d.gtm.confidence, 4), "top3": d.gtm.top3},
        "gtm_note": gtm_note,
        "class_match": d.class_match,
        "consistency": d.consistency,
        "smoothing_windows": d.smoothing_windows,
        "confidence_difference": None if d.confidence_difference is None else round(d.confidence_difference, 4),
        "category": d.category,
        "confidence": round(d.confidence, 4),
        "confidence_level": d.confidence_level,
        "severity": d.severity,
        "alert_status": d.alert_status,
        "consecutive": d.consecutive_count,
        "needs_review": d.needs_review,
        "review_reasons": d.review_reasons,
        "overlapping": d.overlap_classes,
        "recommended_action": d.recommended_action,
        "alert": None if not new_alert else {"code": new_alert.alert_code, "message": new_alert.message,
                                             "severity": new_alert.severity, "occurrences": new_alert.occurrences},
        "latency_ms": int(latency),
        "stored_audio": keep_audio,
    }
